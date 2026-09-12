"""
Export trained Isaac Lab / RSL-RL locomotion policy to ONNX.
Wraps the actor network and empirical observation normalizer into a single forward pass.
"""

from __future__ import annotations
import argparse
import os
import torch
import torch.nn as nn


class PolicyExportWrapper(nn.Module):
    """
    Wraps actor network and empirical observation normalizer into a single forward pass.
    """

    def __init__(
        self,
        actor_module: nn.Module,
        obs_mean: torch.Tensor,
        obs_var: torch.Tensor,
        clip_obs: float = 100.0,
    ) -> None:
        super().__init__()
        self.actor = actor_module
        self.register_buffer("obs_mean", obs_mean)
        self.register_buffer("obs_std", torch.sqrt(obs_var + 1e-8))
        self.clip_obs = clip_obs

    def forward(self, raw_obs: torch.Tensor) -> torch.Tensor:
        norm_obs = torch.clamp(
            (raw_obs - self.obs_mean) / self.obs_std,
            -self.clip_obs,
            self.clip_obs,
        )
        return self.actor(norm_obs)


def export_rsl_rl_to_onnx(
    checkpoint_path: str,
    output_onnx_path: str,
    num_obs: int = 48,
    num_actions: int = 12,
) -> None:
    """
    Loads checkpoint, wraps actor and normalizer, and exports to ONNX.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    actor = checkpoint["model"]["actor"]
    obs_mean = checkpoint["obs_norm"]["mean"]
    obs_var = checkpoint["obs_norm"]["var"]

    wrapper = PolicyExportWrapper(actor, obs_mean, obs_var)
    wrapper.eval()

    dummy_input = torch.zeros((1, num_obs), dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["joint_actions"],
        dynamic_axes=None,  # Fixed batch=1 for zero-allocation RT execution
    )
    print(f"Policy exported to: {output_onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Isaac Lab policy to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default="microduck_locomotion.onnx", help="Output ONNX path")
    parser.add_argument("--num-obs", type=int, default=48, help="Number of observations")
    parser.add_argument("--num-actions", type=int, default=12, help="Number of actuator actions")
    args = parser.parse_args()

    export_rsl_rl_to_onnx(
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output,
        num_obs=args.num_obs,
        num_actions=args.num_actions,
    )
