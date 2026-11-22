"""Self-contained Pollen 61-D -> 14-D ONNX Skill Compiler and Phase Feature Runtime.

Implements the unified 61-D deployment contract for physics-aware ReActor skills
(arXiv:2605.06593v1 adapted to the 14-DoF Microduck bipedal robot):
  * Observation (61-D):
      [0:3]   gyro (rad/s)
      [3:6]   projected_gravity (unit vector in trunk frame)
      [6:20]  joint_pos - DEFAULT_POSE (14-D, rad)
      [20:34] joint_vel (14-D, rad/s, 1-control lag)
      [34:48] last_action (14-D, default-pose-relative action from step k-1)
      [48:61] command / phase_features (13-D)
  * Output (14-D):
      actions (default-pose-relative joint position targets: q_target = DEFAULT_POSE + actions)

When wrapping a ``residual61`` policy (trained with ``q_target = commands[k] + raw_action``
and ``obs[34:48] = last_raw_action``), this compiler embeds ``delta_c[k] = commands[k] - DEFAULT_POSE``
directly inside the ONNX graph as a constant lookup table indexed by the normalized phase
progress ``obs[:, 50]``. It translates ``obs[:, 34:48]`` from ``official61`` default-relative
history back into ``residual61`` raw residual history before the inner network and adds
``delta_c[k]`` to the network output, producing a 100% self-contained ``61 -> 14`` ONNX file
that requires zero external ``.npz`` files at runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

PHASE_SCHEMA_FOURIER_V1 = "microduck_finite_phase_fourier_v1"
PHASE_SCHEMA_FINITE_V1 = "microduck_finite_phase_v1"

OFFICIAL_DEFAULT_POSE = np.array(
    [
        0.0,
        -0.0872665,
        -0.457924,
        -0.004940,
        0.452984,
        0.3490659,
        0.3490659,
        0.0,
        0.0,
        0.0,
        0.0872665,
        0.457924,
        0.004940,
        -0.452984,
    ],
    dtype=np.float32,
)

OFFICIAL_JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase_features_fourier_v1(progress: float | np.ndarray, active: float | np.ndarray = 1.0) -> np.ndarray:
    """Compute the 13-D Fourier phase command (``microduck_finite_phase_fourier_v1``).

    Layout (13 slots):
      [0]  cos(2*pi*p)
      [1]  sin(2*pi*p)
      [2]  p  (normalized progress in [0, 1])
      [3]  cos(4*pi*p)
      [4]  sin(4*pi*p)
      [5]  cos(8*pi*p)
      [6]  sin(8*pi*p)
      [7]  cos(16*pi*p)
      [8]  sin(16*pi*p)
      [9]  cos(32*pi*p)
      [10] sin(32*pi*p)
      [11] active flag psi_t in [0, 1]
      [12] 0.0
    """
    scalar = np.ndim(progress) == 0
    p = np.atleast_1d(np.asarray(progress, dtype=np.float64))
    p = np.clip(p, 0.0, 1.0)
    a = np.broadcast_to(np.asarray(active, dtype=np.float64), p.shape)
    cols = [np.cos(2.0 * math.pi * p), np.sin(2.0 * math.pi * p), p]
    for freq in (2, 4, 8, 16):
        cols.append(np.cos(2.0 * math.pi * freq * p))
        cols.append(np.sin(2.0 * math.pi * freq * p))
    cols.append(a)
    cols.append(np.zeros_like(p))
    out = np.stack(cols, axis=-1).astype(np.float32)
    return out[0] if scalar else out


def phase_features_finite_v1(progress: float | np.ndarray) -> np.ndarray:
    """Compute the 13-D simple phase command (``microduck_finite_phase_v1``).

    Layout (13 slots):
      [0]    cos(2*pi*p)
      [1]    sin(2*pi*p)
      [2]    p  (normalized progress in [0, 1])
      [3:13] 0.0
    """
    scalar = np.ndim(progress) == 0
    p = np.atleast_1d(np.asarray(progress, dtype=np.float64))
    p = np.clip(p, 0.0, 1.0)
    out = np.zeros((len(p), 13), dtype=np.float32)
    out[:, 0] = np.cos(2.0 * math.pi * p).astype(np.float32)
    out[:, 1] = np.sin(2.0 * math.pi * p).astype(np.float32)
    out[:, 2] = p.astype(np.float32)
    return out[0] if scalar else out


def compute_phase_command(
    frame_index: int,
    total_frames: int,
    *,
    phase_schema: str = PHASE_SCHEMA_FOURIER_V1,
    active: float = 1.0,
) -> np.ndarray:
    """Compute the 13-D command vector for frame ``frame_index`` out of ``total_frames``."""
    denom = max(int(total_frames) - 1, 1)
    clamped_idx = min(max(int(frame_index), 0), max(int(total_frames) - 1, 0))
    progress = float(clamped_idx) / float(denom)
    if phase_schema == PHASE_SCHEMA_FINITE_V1:
        return phase_features_finite_v1(progress)
    if phase_schema == PHASE_SCHEMA_FOURIER_V1:
        return phase_features_fourier_v1(progress, active=active)
    raise ValueError(f"Unsupported phase_schema: {phase_schema!r}")


def transition_envelope(
    frame_index: int,
    total_frames: int,
    *,
    entry_ramp_frames: int = 0,
    exit_ramp_frames: int = 0,
) -> float:
    """ReActor-style smooth transition indicator psi_t in [0, 1] (arXiv:2605.06593v1 Eq. 3)."""
    if total_frames <= 1:
        return 1.0
    k = min(max(int(frame_index), 0), total_frames - 1)
    psi = 1.0
    if entry_ramp_frames > 0 and k < entry_ramp_frames:
        u = float(k + 1) / float(entry_ramp_frames + 1)
        psi = min(psi, u * u * (3.0 - 2.0 * u))
    rem = (total_frames - 1) - k
    if exit_ramp_frames > 0 and rem < exit_ramp_frames:
        u = float(rem + 1) / float(exit_ramp_frames + 1)
        psi = min(psi, u * u * (3.0 - 2.0 * u))
    return float(np.clip(psi, 0.0, 1.0))


def read_reactor_onnx_metadata(onnx_path: str | Path) -> dict[str, Any]:
    """Read embedded ReActor skill metadata from an ONNX file's custom metadata properties."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    raw = {p.key: p.value for p in model.metadata_props}
    parsed: dict[str, Any] = {}
    for k, v in raw.items():
        if k in ("frames", "control_steps", "pre_hold_frames", "source_frames", "post_hold_frames"):
            try:
                parsed[k] = int(v)
                continue
            except ValueError:
                pass
        elif k in ("fps", "control_dt_s"):
            try:
                parsed[k] = float(v)
                continue
            except ValueError:
                pass
        elif k in ("default_pose", "joint_names"):
            try:
                parsed[k] = json.loads(v)
                continue
            except json.JSONDecodeError:
                pass
        parsed[k] = v
    return parsed


def _set_model_metadata(model: onnx.ModelProto, metadata: dict[str, Any]) -> None:
    del model.metadata_props[:]
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = str(key)
        if isinstance(value, (list, tuple, dict)):
            entry.value = json.dumps(value)
        else:
            entry.value = str(value)


def _prefix_inner_graph(graph: onnx.GraphProto, prefix: str = "_inner_") -> tuple[str, str]:
    """Rename all internal tensors and nodes in ``graph`` in-place and return (in_name, out_name)."""
    old_in = graph.input[0].name
    old_out = graph.output[0].name
    new_in = f"{prefix}{old_in}"
    new_out = f"{prefix}{old_out}"

    def rename(name: str) -> str:
        if not name:
            return name
        if name == old_in:
            return new_in
        if name == old_out:
            return new_out
        return f"{prefix}{name}"

    for init in graph.initializer:
        init.name = rename(init.name)
    for vi in graph.value_info:
        vi.name = rename(vi.name)
    for node in graph.node:
        node.name = f"{prefix}{node.name}" if node.name else ""
        for i, inp in enumerate(node.input):
            node.input[i] = rename(inp)
        for i, out in enumerate(node.output):
            node.output[i] = rename(out)
    return new_in, new_out


def compile_self_contained_pollen61_onnx(
    source_onnx_path: str | Path,
    output_onnx_path: str | Path,
    *,
    motion_npz_path: str | Path | None = None,
    source_contract: str = "official61",
    phase_schema: str = PHASE_SCHEMA_FOURIER_V1,
    frames: int | None = None,
    skill_name: str = "skill",
    default_pose: np.ndarray = OFFICIAL_DEFAULT_POSE,
) -> dict[str, Any]:
    """Compile any 61-input ReActor ONNX policy into a self-contained ``61 -> 14`` ``official61`` ONNX.

    Supports:
      * ``source_contract="residual61"``: embeds ``delta_c[k] = commands[k] - default_pose`` from
        ``motion_npz_path`` as a constant tensor inside the ONNX graph, converts ``obs[:, 34:48]``
        from ``official61`` default-relative history to ``residual61`` raw residual history, and
        adds ``delta_c[k]`` to the output.
      * ``source_contract="official61"``: slices 20-D training-wheels outputs to 14-D if needed,
        and embeds frame/phase metadata directly into the ONNX file.
    """
    source_onnx_path = Path(source_onnx_path)
    output_onnx_path = Path(output_onnx_path)
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(source_onnx_path))
    graph = model.graph
    in_dims = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    out_dims = [d.dim_value for d in graph.output[0].type.tensor_type.shape.dim]
    if len(in_dims) != 2 or in_dims[1] != 61:
        raise ValueError(f"Expected 61-D input ONNX, got shape {in_dims}")
    out_width = int(out_dims[1])
    if out_width < 14:
        raise ValueError(f"Expected >= 14-D output ONNX, got shape {out_dims}")

    pre_hold_frames = 0
    source_frames = 0
    post_hold_frames = 0
    fps = 50.0
    delta_c: np.ndarray | None = None
    ref_sha256 = ""

    if motion_npz_path is not None:
        motion_npz_path = Path(motion_npz_path)
        ref_sha256 = sha256_file(motion_npz_path)
        with np.load(motion_npz_path) as raw:
            if "source_frame_index" in raw:
                sfi = np.asarray(raw["source_frame_index"], dtype=np.int64)
                inferred_frames = int(len(sfi))
                src_idx = np.flatnonzero(sfi >= 0)
                if len(src_idx) > 0:
                    pre_hold_frames = int(src_idx[0])
                    source_frames = int(len(src_idx))
                    post_hold_frames = int(inferred_frames - int(src_idx[-1]) - 1)
            elif "qpos" in raw:
                inferred_frames = int(len(raw["qpos"]))
                source_frames = inferred_frames
            else:
                raise ValueError(f"Cannot infer frame count from {motion_npz_path}")

            if frames is None:
                frames = inferred_frames
            if "fps" in raw:
                fps = float(np.asarray(raw["fps"]).reshape(-1)[0])

            if source_contract == "residual61":
                if "commands" not in raw:
                    raise ValueError(f"residual61 requires 'commands' in {motion_npz_path}")
                cmds = np.asarray(raw["commands"], dtype=np.float64)
                if cmds.ndim != 2 or cmds.shape[1] != 14:
                    raise ValueError(f"Expected (T, 14) commands, got {cmds.shape}")
                def_pose = np.asarray(default_pose, dtype=np.float64).reshape(1, 14)
                idx = np.minimum(np.arange(frames), len(cmds) - 1)
                delta_c = (cmds[idx] - def_pose).astype(np.float32)

    if frames is None or frames < 2:
        raise ValueError("frames must be >= 2 (provide frames or motion_npz_path)")
    if source_frames == 0:
        source_frames = frames

    if source_contract == "residual61":
        assert delta_c is not None
        inner_in, inner_out = _prefix_inner_graph(graph, prefix="_inner_")
        inner_nodes = list(graph.node)
        del graph.node[:]

        # Constant initializers for residual61 -> official61 graph wrapper
        inits = [
            numpy_helper.from_array(delta_c, name="_wrap_delta_c"),
            numpy_helper.from_array(np.array([0, 0], dtype=np.int64), name="_wrap_s_prefix_start"),
            numpy_helper.from_array(np.array([1, 34], dtype=np.int64), name="_wrap_s_prefix_end"),
            numpy_helper.from_array(np.array([0, 34], dtype=np.int64), name="_wrap_s_last_start"),
            numpy_helper.from_array(np.array([1, 48], dtype=np.int64), name="_wrap_s_last_end"),
            numpy_helper.from_array(np.array([0, 48], dtype=np.int64), name="_wrap_s_suffix_start"),
            numpy_helper.from_array(np.array([1, 61], dtype=np.int64), name="_wrap_s_suffix_end"),
            numpy_helper.from_array(np.array([0, 50], dtype=np.int64), name="_wrap_s_prog_start"),
            numpy_helper.from_array(np.array([1, 51], dtype=np.int64), name="_wrap_s_prog_end"),
            numpy_helper.from_array(np.array([0, 1], dtype=np.int64), name="_wrap_axes_01"),
            numpy_helper.from_array(np.array([[float(frames - 1)]], dtype=np.float32), name="_wrap_f_max_f32"),
            numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="_wrap_zero_f32"),
            numpy_helper.from_array(np.array(float(frames - 1), dtype=np.float32), name="_wrap_max_scalar_f32"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), name="_wrap_shape_1"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), name="_wrap_one_i64"),
            numpy_helper.from_array(np.array(0, dtype=np.int64), name="_wrap_zero_i64"),
            numpy_helper.from_array(np.array(frames - 1, dtype=np.int64), name="_wrap_max_scalar_i64"),
        ]
        if out_width > 14:
            inits.extend([
                numpy_helper.from_array(np.array([0, 0], dtype=np.int64), name="_wrap_s_act_start"),
                numpy_helper.from_array(np.array([1, 14], dtype=np.int64), name="_wrap_s_act_end"),
            ])
        graph.initializer.extend(inits)

        pre_nodes = [
            helper.make_node(
                "Slice",
                inputs=["obs", "_wrap_s_prefix_start", "_wrap_s_prefix_end", "_wrap_axes_01"],
                outputs=["_wrap_obs_prefix"],
                name="_wrap_slice_prefix",
            ),
            helper.make_node(
                "Slice",
                inputs=["obs", "_wrap_s_last_start", "_wrap_s_last_end", "_wrap_axes_01"],
                outputs=["_wrap_obs_last"],
                name="_wrap_slice_last",
            ),
            helper.make_node(
                "Slice",
                inputs=["obs", "_wrap_s_suffix_start", "_wrap_s_suffix_end", "_wrap_axes_01"],
                outputs=["_wrap_obs_suffix"],
                name="_wrap_slice_suffix",
            ),
            helper.make_node(
                "Slice",
                inputs=["obs", "_wrap_s_prog_start", "_wrap_s_prog_end", "_wrap_axes_01"],
                outputs=["_wrap_obs_prog"],
                name="_wrap_slice_prog",
            ),
            helper.make_node(
                "Mul",
                inputs=["_wrap_obs_prog", "_wrap_f_max_f32"],
                outputs=["_wrap_prog_scaled"],
                name="_wrap_mul_prog",
            ),
            helper.make_node(
                "Round",
                inputs=["_wrap_prog_scaled"],
                outputs=["_wrap_prog_round"],
                name="_wrap_round_prog",
            ),
            helper.make_node(
                "Clip",
                inputs=["_wrap_prog_round", "_wrap_zero_f32", "_wrap_max_scalar_f32"],
                outputs=["_wrap_prog_clip"],
                name="_wrap_clip_prog",
            ),
            helper.make_node(
                "Cast",
                inputs=["_wrap_prog_clip"],
                outputs=["_wrap_k_2d"],
                to=TensorProto.INT64,
                name="_wrap_cast_k",
            ),
            helper.make_node(
                "Reshape",
                inputs=["_wrap_k_2d", "_wrap_shape_1"],
                outputs=["_wrap_k_1d"],
                name="_wrap_reshape_k",
            ),
            helper.make_node(
                "Sub",
                inputs=["_wrap_k_1d", "_wrap_one_i64"],
                outputs=["_wrap_k_minus_1"],
                name="_wrap_sub_k",
            ),
            helper.make_node(
                "Clip",
                inputs=["_wrap_k_minus_1", "_wrap_zero_i64", "_wrap_max_scalar_i64"],
                outputs=["_wrap_k_prev_1d"],
                name="_wrap_clip_k_prev",
            ),
            helper.make_node(
                "Greater",
                inputs=["_wrap_k_2d", "_wrap_zero_i64"],
                outputs=["_wrap_k_gt_0"],
                name="_wrap_gt_k",
            ),
            helper.make_node(
                "Cast",
                inputs=["_wrap_k_gt_0"],
                outputs=["_wrap_is_pos_f32"],
                to=TensorProto.FLOAT,
                name="_wrap_cast_pos",
            ),
            helper.make_node(
                "Gather",
                inputs=["_wrap_delta_c", "_wrap_k_1d"],
                outputs=["_wrap_curr_offset"],
                axis=0,
                name="_wrap_gather_curr",
            ),
            helper.make_node(
                "Gather",
                inputs=["_wrap_delta_c", "_wrap_k_prev_1d"],
                outputs=["_wrap_prev_offset"],
                axis=0,
                name="_wrap_gather_prev",
            ),
            helper.make_node(
                "Sub",
                inputs=["_wrap_obs_last", "_wrap_prev_offset"],
                outputs=["_wrap_last_diff"],
                name="_wrap_sub_last",
            ),
            helper.make_node(
                "Mul",
                inputs=["_wrap_last_diff", "_wrap_is_pos_f32"],
                outputs=["_wrap_inner_last_raw"],
                name="_wrap_mul_last",
            ),
            helper.make_node(
                "Concat",
                inputs=["_wrap_obs_prefix", "_wrap_inner_last_raw", "_wrap_obs_suffix"],
                outputs=[inner_in],
                axis=1,
                name="_wrap_concat_obs",
            ),
        ]

        post_nodes = []
        raw_14_name = inner_out
        if out_width > 14:
            raw_14_name = "_wrap_inner_raw_14"
            post_nodes.append(
                helper.make_node(
                    "Slice",
                    inputs=[inner_out, "_wrap_s_act_start", "_wrap_s_act_end", "_wrap_axes_01"],
                    outputs=[raw_14_name],
                    name="_wrap_slice_act14",
                )
            )
        post_nodes.append(
            helper.make_node(
                "Add",
                inputs=["_wrap_curr_offset", raw_14_name],
                outputs=["actions"],
                name="_wrap_add_offset",
            )
        )

        graph.node.extend(pre_nodes)
        graph.node.extend(inner_nodes)
        graph.node.extend(post_nodes)

    elif out_width > 14:
        # official61 with extra outputs (e.g. 61 -> 20 training-wheels wrench): slice to 14
        old_out = graph.output[0].name
        raw_out = "_inner_actions_full"
        for node in graph.node:
            for i, out in enumerate(node.output):
                if out == old_out:
                    node.output[i] = raw_out
        graph.initializer.extend([
            numpy_helper.from_array(np.array([0, 0], dtype=np.int64), name="_wrap_s_act_start"),
            numpy_helper.from_array(np.array([1, 14], dtype=np.int64), name="_wrap_s_act_end"),
            numpy_helper.from_array(np.array([0, 1], dtype=np.int64), name="_wrap_axes_01"),
        ])
        graph.node.append(
            helper.make_node(
                "Slice",
                inputs=[raw_out, "_wrap_s_act_start", "_wrap_s_act_end", "_wrap_axes_01"],
                outputs=[old_out],
                name="_wrap_slice_act14",
            )
        )

    # Ensure output tensor type is explicitly [1, 14]
    del graph.input[:]
    del graph.output[:]
    graph.input.append(helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61]))
    graph.output.append(helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14]))

    meta = {
        "schema": "microduck-pollen61-self-contained-v1",
        "skill_name": skill_name,
        "contract": "official61",
        "source_contract": source_contract,
        "phase_schema": phase_schema,
        "frames": int(frames),
        "control_steps": int(frames - 1),
        "pre_hold_frames": int(pre_hold_frames),
        "source_frames": int(source_frames),
        "post_hold_frames": int(post_hold_frames),
        "fps": float(fps),
        "control_dt_s": round(1.0 / float(fps), 6),
        "default_pose": [round(float(x), 7) for x in default_pose],
        "joint_names": list(OFFICIAL_JOINT_NAMES),
        "source_onnx_sha256": sha256_file(source_onnx_path),
        "reference_sha256": ref_sha256,
    }
    _set_model_metadata(model, meta)
    onnx.checker.check_model(model)
    onnx.save(model, str(output_onnx_path))
    meta["output_onnx"] = str(output_onnx_path)
    meta["output_onnx_sha256"] = sha256_file(output_onnx_path)
    return meta


# Minimum kinematic motion-fidelity floors so stand-in-place policies cannot pass as working skills
SKILL_FIDELITY_FLOORS: dict[str, dict[str, float]] = {
    "bow": {"min_forward_pitch_deg": 12.0, "min_mean_joint_speed_rad_s": 0.03},
    "bow_walk_entry": {"min_forward_pitch_deg": 12.0, "min_mean_joint_speed_rad_s": 0.03},
    "kick": {"min_foot_lift_mm": 40.0, "min_mean_joint_speed_rad_s": 0.10},
    "karate_kicks": {"min_foot_lift_mm": 40.0, "min_mean_joint_speed_rad_s": 0.10},
    "jazzwalk": {"min_foot_lift_mm": 25.0, "min_mean_joint_speed_rad_s": 0.10},
    "wave": {"min_head_ptp_rad": 0.50, "min_mean_joint_speed_rad_s": 0.06},
    "byebye": {"min_head_ptp_rad": 0.50, "min_mean_joint_speed_rad_s": 0.06},
    "call": {"min_head_ptp_rad": 0.65, "min_mean_joint_speed_rad_s": 0.075},
}


def evaluate_skill_fidelity(
    onnx_path: str | Path,
    *,
    skill_name: str | None = None,
) -> dict[str, Any]:
    """Run a closed-loop ExactBamPlant rollout and check real kinematic motion-fidelity floors.

    Prevents stand-in-place or severely under-moving policies from passing merely because
    they do not fall.
    """
    from .mocap_engine import ReactorSkillPlayer
    from .sim.exact_bam_plant import ExactBamPlant

    path = Path(onnx_path)
    plant = ExactBamPlant()
    player = ReactorSkillPlayer(path, blend_entry_frames=0)
    resolved_name = skill_name or player.skill_name

    plant.reset_stand()
    player.start()
    qs: list[np.ndarray] = []
    pitches_deg: list[float] = []
    rolls_deg: list[float] = []
    lf_z_mm: list[float] = []
    rf_z_mm: list[float] = []
    fell = False
    fell_at = -1

    for k in range(player.control_steps):
        gyro = plant.data.sensordata[plant.gyro_adr : plant.gyro_adr + 3]
        grav = plant.projected_gravity()
        q = plant.data.qpos[plant.qadr]
        dq = plant.data.qvel[plant.dadr]
        done, target_q, _ = player.step_closed_loop(gyro, grav, q, dq)
        plant.step_control(target_q)

        qs.append(plant.data.qpos[plant.qadr].copy())
        g = plant.projected_gravity()
        pitches_deg.append(float(np.degrees(np.arctan2(-g[0], np.hypot(g[1], g[2])))))
        rolls_deg.append(float(np.degrees(np.arctan2(g[1], -g[2]))))
        feet = plant.data.site_xpos[plant.foot_sites]
        lf_z_mm.append(float(feet[0, 2] * 1000.0))
        rf_z_mm.append(float(feet[1, 2] * 1000.0))

        if plant.is_fallen():
            fell = True
            fell_at = k
            break
        if done:
            break

    q_arr = np.asarray(qs, dtype=np.float64)
    q_ptp = np.ptp(q_arr, axis=0) if len(q_arr) > 1 else np.zeros(14)
    mean_spd = float(np.mean(np.abs(np.diff(q_arr, axis=0)) / 0.02)) if len(q_arr) > 1 else 0.0
    max_forward_pitch = float(-min(pitches_deg)) if pitches_deg else 0.0
    max_foot_z = max(max(lf_z_mm, default=0.0), max(rf_z_mm, default=0.0))
    max_head_ptp = float(np.max(q_ptp[5:9]))
    max_leg_ptp = float(np.max(np.r_[q_ptp[:5], q_ptp[9:]]))

    floors = SKILL_FIDELITY_FLOORS.get(
        resolved_name, {"min_mean_joint_speed_rad_s": 0.04, "min_joint_ptp_rad": 0.30}
    )
    failures: list[str] = []
    if fell:
        failures.append(f"fell at step {fell_at}")
    if "min_forward_pitch_deg" in floors and max_forward_pitch < floors["min_forward_pitch_deg"]:
        failures.append(
            f"forward pitch {max_forward_pitch:.1f}deg < {floors['min_forward_pitch_deg']:.1f}deg"
        )
    if "min_foot_lift_mm" in floors and max_foot_z < floors["min_foot_lift_mm"]:
        failures.append(f"max foot lift {max_foot_z:.1f}mm < {floors['min_foot_lift_mm']:.1f}mm")
    if "min_head_ptp_rad" in floors and max_head_ptp < floors["min_head_ptp_rad"]:
        failures.append(f"max head ptp {max_head_ptp:.3f}rad < {floors['min_head_ptp_rad']:.3f}rad")
    if "min_joint_ptp_rad" in floors and max(max_leg_ptp, max_head_ptp) < floors["min_joint_ptp_rad"]:
        failures.append(
            f"max joint ptp {max(max_leg_ptp, max_head_ptp):.3f}rad < {floors['min_joint_ptp_rad']:.3f}rad"
        )
    if "min_mean_joint_speed_rad_s" in floors and mean_spd < floors["min_mean_joint_speed_rad_s"]:
        failures.append(
            f"mean joint speed {mean_spd:.3f}rad/s < {floors['min_mean_joint_speed_rad_s']:.3f}rad/s"
        )

    return {
        "skill_name": resolved_name,
        "steps_executed": len(qs),
        "fell": fell,
        "fell_at": fell_at,
        "max_forward_pitch_deg": round(max_forward_pitch, 2),
        "min_pitch_deg": round(min(pitches_deg, default=0.0), 2),
        "max_pitch_deg": round(max(pitches_deg, default=0.0), 2),
        "max_left_foot_z_mm": round(max(lf_z_mm, default=0.0), 2),
        "max_right_foot_z_mm": round(max(rf_z_mm, default=0.0), 2),
        "max_foot_lift_mm": round(max_foot_z, 2),
        "max_head_ptp_rad": round(max_head_ptp, 4),
        "max_leg_ptp_rad": round(max_leg_ptp, 4),
        "mean_joint_speed_rad_s": round(mean_spd, 4),
        "fidelity_floors": floors,
        "failures": failures,
        "fidelity_pass": len(failures) == 0,
    }
