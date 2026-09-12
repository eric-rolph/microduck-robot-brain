"""Local PyTorch PPO training and verification runner for Microduck in MuJoCo.

Validates the standardized 61-D observation contract, BAM M6 actuator dynamics,
reward convergence, advantage estimation, and ONNX policy export.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from microduck_brain.sim.env import MicroduckMuJoCoEnv


class ActorCritic(nn.Module):
    """Actor-Critic network for 61-D observation to 14-D joint actions."""

    def __init__(self, obs_dim: int = 61, action_dim: int = 14) -> None:
        super().__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        val = self.critic(obs)
        return dist, val

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, val = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, val.squeeze(-1)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, val = self.forward(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, val.squeeze(-1), entropy


class MicroduckPPO:
    """PPO trainer for the Microduck MuJoCo environment."""

    def __init__(
        self,
        num_envs: int = 4,
        steps_per_env: int = 64,
        device: str = "cpu",
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> None:
        self.num_envs = num_envs
        self.steps_per_env = steps_per_env
        self.device = torch.device(device)
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lam = lam

        # Initialize parallel environments
        self.envs = [MicroduckMuJoCoEnv(use_backlash=True) for _ in range(num_envs)]
        self.model = ActorCritic(obs_dim=61, action_dim=14).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def collect_rollouts(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, float],
    ]:
        """Collect rollouts across parallel environments."""
        total_steps = self.num_envs * self.steps_per_env
        obs_buf = torch.zeros((self.steps_per_env, self.num_envs, 61), device=self.device)
        act_buf = torch.zeros((self.steps_per_env, self.num_envs, 14), device=self.device)
        logp_buf = torch.zeros((self.steps_per_env, self.num_envs), device=self.device)
        rew_buf = torch.zeros((self.steps_per_env, self.num_envs), device=self.device)
        done_buf = torch.zeros((self.steps_per_env, self.num_envs), device=self.device)
        val_buf = torch.zeros((self.steps_per_env, self.num_envs), device=self.device)

        current_obs = [env.reset() for env in self.envs]
        # Command forward walk
        for env in self.envs:
            env.set_command(lin_vel_x=0.15, ang_vel_z=0.0)

        ep_returns = []
        cur_returns = np.zeros(self.num_envs)

        with torch.no_grad():
            for t in range(self.steps_per_env):
                obs_tensor = torch.tensor(
                    np.array(current_obs), dtype=torch.float32, device=self.device
                )
                action, logp, val = self.model.get_action(obs_tensor)

                act_np = action.cpu().numpy()
                obs_buf[t] = obs_tensor
                act_buf[t] = action
                logp_buf[t] = logp
                val_buf[t] = val

                next_obs = []
                for i, env in enumerate(self.envs):
                    o, r, term, trunc, info = env.step(act_np[i])
                    done = term or trunc
                    rew_buf[t, i] = r
                    done_buf[t, i] = float(done)
                    cur_returns[i] += r

                    if done:
                        ep_returns.append(cur_returns[i])
                        cur_returns[i] = 0.0
                        o = env.reset()
                        env.set_command(lin_vel_x=0.15, ang_vel_z=0.0)

                    next_obs.append(o)

                current_obs = next_obs

            # Final values for GAE
            last_obs_tensor = torch.tensor(
                np.array(current_obs), dtype=torch.float32, device=self.device
            )
            _, _, last_val = self.model.get_action(last_obs_tensor)

        # Generalized Advantage Estimation (GAE)
        adv_buf = torch.zeros_like(rew_buf)
        last_gae = 0.0
        for t in reversed(range(self.steps_per_env)):
            if t == self.steps_per_env - 1:
                next_non_terminal = 1.0 - done_buf[t]
                next_value = last_val
            else:
                next_non_terminal = 1.0 - done_buf[t]
                next_value = val_buf[t + 1]

            delta = rew_buf[t] + self.gamma * next_value * next_non_terminal - val_buf[t]
            adv_buf[t] = last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae

        ret_buf = adv_buf + val_buf

        # Flatten batch dimensions
        flat_obs = obs_buf.reshape(total_steps, 61)
        flat_act = act_buf.reshape(total_steps, 14)
        flat_logp = logp_buf.reshape(total_steps)
        flat_adv = adv_buf.reshape(total_steps)
        flat_ret = ret_buf.reshape(total_steps)

        # Normalize advantages
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        stats = {
            "mean_reward": float(rew_buf.mean().item()),
            "mean_return": float(np.mean(ep_returns)) if ep_returns else float(cur_returns.mean()),
        }
        return flat_obs, flat_act, flat_logp, flat_adv, flat_ret, stats

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_logp: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        epochs: int = 4,
        batch_size: int = 64,
    ) -> dict[str, float]:
        """Perform PPO clipped policy and value updates."""
        total_steps = obs.shape[0]
        policy_losses = []
        value_losses = []
        kls = []

        for _ in range(epochs):
            indices = torch.randperm(total_steps, device=self.device)
            for start in range(0, total_steps, batch_size):
                batch_idx = indices[start : start + batch_size]
                b_obs = obs[batch_idx]
                b_act = actions[batch_idx]
                b_old_logp = old_logp[batch_idx]
                b_adv = advantages[batch_idx]
                b_ret = returns[batch_idx]

                new_logp, val, entropy = self.model.evaluate_actions(b_obs, b_act)

                # Ratio
                ratio = torch.exp(new_logp - b_old_logp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * ((val - b_ret) ** 2).mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                kls.append(approx_kl.item())

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "approx_kl": float(np.mean(kls)),
        }

    def train_smoke(self, iterations: int = 5) -> list[dict[str, float]]:
        """Run smoke training loop and report progress."""
        history = []
        for it in range(1, iterations + 1):
            obs, act, logp, adv, ret, collect_stats = self.collect_rollouts()
            update_stats = self.update(obs, act, logp, adv, ret)

            metrics = {
                "iteration": it,
                "mean_reward": collect_stats["mean_reward"],
                "mean_return": collect_stats["mean_return"],
                "policy_loss": update_stats["policy_loss"],
                "value_loss": update_stats["value_loss"],
                "approx_kl": update_stats["approx_kl"],
            }
            history.append(metrics)
            print(
                f"[PPO Iter {it:2d}/{iterations}] "
                f"Reward: {metrics['mean_reward']:6.2f} | "
                f"Return: {metrics['mean_return']:6.1f} | "
                f"PolLoss: {metrics['policy_loss']:+7.4f} | "
                f"ValLoss: {metrics['value_loss']:6.4f} | "
                f"KL: {metrics['approx_kl']:6.4f}"
            )

        return history

    def export_onnx(self, output_path: str | Path) -> None:
        """Export trained actor MLP to standalone ONNX file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        dummy_input = torch.zeros((1, 61), dtype=torch.float32, device=self.device)

        class ActorWrapper(nn.Module):
            def __init__(self, actor_net: nn.Module) -> None:
                super().__init__()
                self.actor_net = actor_net

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.actor_net(x)

        wrapper = ActorWrapper(self.model.actor).to(self.device)

        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={"obs": {0: "batch_size"}, "action": {0: "batch_size"}},
            opset_version=17,
        )
        print(f"Exported policy ONNX model to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Microduck MuJoCo PPO training smoke runner")
    parser.add_argument("--iterations", type=int, default=5, help="Number of PPO iterations")
    parser.add_argument("--num-envs", type=int, default=4, help="Parallel environments")
    parser.add_argument("--steps-per-env", type=int, default=64, help="Steps per rollout")
    parser.add_argument("--export-onnx", type=str, default="models/microduck_walk.onnx")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    print(f"Starting Microduck PPO training run on {args.device}...")
    trainer = MicroduckPPO(
        num_envs=args.num_envs,
        steps_per_env=args.steps_per_env,
        device=args.device,
    )
    history = trainer.train_smoke(iterations=args.iterations)

    if args.export_onnx:
        trainer.export_onnx(args.export_onnx)

    print("Training run completed successfully.")


if __name__ == "__main__":
    main()
