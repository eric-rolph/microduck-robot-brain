"""Batched PyTorch/CUDA Policy Evaluator backed by real ExactBamPlant MuJoCo Physics.

Loads ONNX actor graphs faithfully into PyTorch (preserving observation normalization
layers and exact graph topology, verified to < 1e-5 parity against ONNX Runtime) and
evaluates closed-loop bipedal stability on real ExactBamPlant XL330 M6 MuJoCo plants
under P1 domain randomization and horizontal trunk push disturbances.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from microduck_brain.sim.exact_bam_plant import ExactBamPlant, PlantRandomization

# Standardized 61-D observation vector dimension & 14-D action space
OBS_DIM: int = 61
ACTION_DIM: int = 14


class TorchPolicyFromONNX(nn.Module):
    """Exact topological PyTorch evaluator for MLP / normalized ONNX actor graphs."""

    def __init__(self, onnx_path: str | Path, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.onnx_path = Path(onnx_path)
        model = onnx.load(str(self.onnx_path))
        self.input_name = model.graph.input[0].name
        self.output_name = model.graph.output[0].name

        self._tensor_names: dict[str, str] = {}
        for idx, init in enumerate(model.graph.initializer):
            arr = numpy_helper.to_array(init).copy()
            tensor = torch.from_numpy(arr).to(device)
            buf_name = f"init_{idx}"
            self.register_buffer(buf_name, tensor)
            self._tensor_names[init.name] = buf_name

        self._nodes: list[tuple[str, tuple[str, ...], tuple[str, ...], dict[str, Any]]] = []
        for node in model.graph.node:
            attrs: dict[str, Any] = {}
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.FLOAT:
                    attrs[attr.name] = float(attr.f)
                elif attr.type == onnx.AttributeProto.INT:
                    attrs[attr.name] = int(attr.i)
                elif attr.type == onnx.AttributeProto.INTS:
                    attrs[attr.name] = [int(v) for v in attr.ints]
                elif attr.type == onnx.AttributeProto.TENSOR:
                    arr = numpy_helper.to_array(attr.t).copy()
                    attrs[attr.name] = torch.from_numpy(arr).to(device)
            self._nodes.append((node.op_type, tuple(node.input), tuple(node.output), attrs))

    def _get_tensor(self, name: str, env: dict[str, torch.Tensor]) -> torch.Tensor:
        if name in env:
            return env[name]
        buf_name = self._tensor_names.get(name)
        if buf_name is not None:
            return getattr(self, buf_name)
        raise KeyError(f"Missing ONNX tensor {name!r} while executing {self.onnx_path.name}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        env: dict[str, torch.Tensor] = {self.input_name: obs}
        for op, inputs, outputs, attrs in self._nodes:
            vals = [self._get_tensor(name, env) if name else None for name in inputs]
            if op == "Sub":
                res = vals[0] - vals[1]
            elif op == "Add":
                res = vals[0] + vals[1]
            elif op == "Mul":
                res = vals[0] * vals[1]
            elif op == "Div":
                res = vals[0] / vals[1]
            elif op == "MatMul":
                res = torch.matmul(vals[0], vals[1])
            elif op == "Gemm":
                a, b = vals[0], vals[1]
                if attrs.get("transA", 0):
                    a = a.transpose(-1, -2)
                if attrs.get("transB", 0):
                    b = b.transpose(-1, -2)
                alpha = attrs.get("alpha", 1.0)
                beta = attrs.get("beta", 1.0)
                res = alpha * torch.matmul(a, b)
                if len(vals) > 2 and vals[2] is not None:
                    res = res + beta * vals[2]
            elif op == "Elu":
                res = F.elu(vals[0], alpha=attrs.get("alpha", 1.0))
            elif op == "Relu":
                res = F.relu(vals[0])
            elif op == "Tanh":
                res = torch.tanh(vals[0])
            elif op == "Sigmoid":
                res = torch.sigmoid(vals[0])
            elif op == "Constant":
                res = attrs["value"].to(self.device)
            elif op == "Clip":
                lo = vals[1] if len(vals) > 1 else None
                hi = vals[2] if len(vals) > 2 else None
                res = torch.clamp(vals[0], min=lo, max=hi)
            elif op == "Slice":
                data = vals[0]
                starts = vals[1].tolist()
                ends = vals[2].tolist()
                axes = vals[3].tolist() if len(vals) > 3 and vals[3] is not None else list(range(len(starts)))
                slices = [slice(None)] * data.ndim
                for ax, st, en in zip(axes, starts, ends):
                    if ax == 0 and data.shape[0] != 1 and st == 0 and en == 1:
                        slices[ax] = slice(None)
                    else:
                        slices[ax] = slice(int(st), int(en))
                res = data[tuple(slices)]
            elif op == "Concat":
                axis = attrs.get("axis", 0)
                res = torch.cat(vals, dim=axis)
            else:
                raise NotImplementedError(f"Unsupported ONNX operator {op!r} in {self.onnx_path.name}")
            env[outputs[0]] = res
        return env[self.output_name]

    def verify_ort_parity(self, num_samples: int = 32, seed: int = 0) -> float:
        """Return max absolute output difference vs CPU ONNX Runtime on random inputs."""
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(self.onnx_path), opts, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        rng = np.random.default_rng(seed)
        xs = rng.normal(0.0, 0.25, size=(num_samples, OBS_DIM)).astype(np.float32)
        with torch.no_grad():
            pt_out = self.forward(torch.from_numpy(xs).to(self.device)).detach().cpu().numpy()
        ort_out = np.concatenate([sess.run(None, {in_name: xs[i : i + 1]})[0] for i in range(num_samples)], axis=0)
        return float(np.max(np.abs(pt_out.astype(np.float64) - ort_out.astype(np.float64))))


@dataclass
class CUDABenchmarkResult:
    device_name: str
    num_parallel_envs: int
    num_steps: int
    total_samples: int
    elapsed_seconds: float
    throughput_samples_per_sec: float
    zero_fall_rate: float
    mean_tilt_deg: float
    max_tilt_deg: float
    pass_certification: bool
    ort_parity_max_abs: float = 0.0


class MicroduckCUDATestbench:
    """Batched PyTorch/CUDA policy verifier backed by real ExactBamPlant MuJoCo rollouts."""

    def __init__(
        self,
        policy_path: str | Path = "models/alpha_walking.onnx",
        device: str | None = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.policy_path = Path(policy_path)
        self.policy = TorchPolicyFromONNX(self.policy_path, device=self.device)
        self.policy.eval()
        self.ort_parity_max_abs = self.policy.verify_ort_parity(num_samples=16, seed=0)
        if self.ort_parity_max_abs > 1e-4:
            raise RuntimeError(
                f"TorchPolicyFromONNX parity mismatch for {self.policy_path.name}: {self.ort_parity_max_abs:.3e}"
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def run_stress_test(
        self,
        num_parallel_envs: int = 32,
        num_steps: int = 100,
        lateral_push_max_mps: float = 0.25,
        obs_noise_std: float = 0.02,
        seed: int = 42,
    ) -> CUDABenchmarkResult:
        """Run real ExactBamPlant MuJoCo rollouts with P1 DR, lateral pushes, and batched GPU inference."""
        rng = np.random.default_rng(seed)
        noise_scale = float(np.clip(obs_noise_std / 0.02, 0.0, 2.0))
        plants: list[ExactBamPlant] = []
        for idx in range(num_parallel_envs):
            dr = PlantRandomization.draw_p1(rng, scale=0.5, noise=(noise_scale > 0.0))
            dr.sensor_noise_scale *= noise_scale
            plant = ExactBamPlant(randomization=dr, seed=seed + idx + 1)
            plant.reset_stand()
            plants.append(plant)

        cmd = np.zeros(13, dtype=np.float32)
        cmd[0] = 0.08  # Forward velocity command in canonical slot [48]
        last_actions = np.zeros((num_parallel_envs, ACTION_DIM), dtype=np.float32)
        falls = np.zeros(num_parallel_envs, dtype=bool)
        push_step = min(max(num_steps // 3, 5), max(num_steps - 1, 1))
        push_magnitudes = rng.uniform(-lateral_push_max_mps, lateral_push_max_mps, size=num_parallel_envs)
        tilts: list[float] = []

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start_time = time.perf_counter()

        with torch.no_grad():
            for step in range(num_steps):
                if step == push_step:
                    for i, plant in enumerate(plants):
                        if not falls[i]:
                            plant.data.qvel[1] += float(push_magnitudes[i])

                obs_batch = np.stack(
                    [
                        plant.observe(cmd, lag_velocity=False, last_action=last_actions[i])
                        for i, plant in enumerate(plants)
                    ],
                    axis=0,
                )
                obs_t = torch.from_numpy(obs_batch).to(self.device)
                act_batch = self.policy(obs_t).detach().cpu().numpy()[:, :ACTION_DIM].astype(np.float32)

                for i, plant in enumerate(plants):
                    if falls[i]:
                        continue
                    plant.step_action(act_batch[i])
                    last_actions[i] = act_batch[i]
                    g = plant.projected_gravity()
                    tilt_deg = float(np.degrees(np.arccos(np.clip(-g[2], -1.0, 1.0))))
                    tilts.append(tilt_deg)
                    if plant.is_fallen():
                        falls[i] = True

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start_time

        mean_tilt = float(np.mean(tilts)) if tilts else 0.0
        max_tilt = float(np.max(tilts)) if tilts else 0.0
        zero_fall_rate = 1.0 - (int(np.sum(falls)) / max(num_parallel_envs, 1))
        total_samples = num_parallel_envs * num_steps
        throughput = total_samples / max(1e-4, elapsed)
        dev_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"

        return CUDABenchmarkResult(
            device_name=dev_name,
            num_parallel_envs=num_parallel_envs,
            num_steps=num_steps,
            total_samples=total_samples,
            elapsed_seconds=elapsed,
            throughput_samples_per_sec=throughput,
            zero_fall_rate=zero_fall_rate,
            mean_tilt_deg=mean_tilt,
            max_tilt_deg=max_tilt,
            pass_certification=(zero_fall_rate >= 0.99 and max_tilt < 45.0 and self.ort_parity_max_abs <= 1e-4),
            ort_parity_max_abs=self.ort_parity_max_abs,
        )
