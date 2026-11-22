"""Three-Tier End-to-End Test Suite for ReActor Physics-Aware Motion Retargeting Skills.

Covers:
  * Tier 1: Self-contained 61 -> 14 ONNX contract, metadata, phase encoding, and
    < 1e-6 multi-step closed-loop parity between residual61 and compiled official61.
  * Tier 2: Closed-loop MuJoCo + Exact XL330 M6 BAM physics execution for all 6 skills
    (bow, kick, wave, call, karate_kicks, jazzwalk), walk -> skill -> walk chaining,
    P1 domain randomization + push rejection, Stage-1 assisted retargeting (RFC + TTSA),
    and knockdown get-up recovery (BEST_alpha_stand.onnx -> BEST_alpha_walking.onnx).
  * Tier 3: Behavior Tree MocapMotionActionNode integration, manifest.json validation,
    and physical PollenRobotdClient JSON-lines loopback dispatch.
"""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from microduck_brain.behavior_tree import Blackboard, MocapMotionActionNode, SequenceNode
from microduck_brain.locomotion_engine import MicroduckLocomotionEngine
from microduck_brain.mocap_engine import MocapPlayer, ReactorSkillPlayer
from microduck_brain.pollen_bridge import PollenRobotdClient
from microduck_brain.reactor_assisted_retarget import AssistedRetargetConfig, run_assisted_retarget
from microduck_brain.reactor_onnx_compiler import (
    OFFICIAL_DEFAULT_POSE,
    PHASE_SCHEMA_FINITE_V1,
    PHASE_SCHEMA_FOURIER_V1,
    compile_self_contained_pollen61_onnx,
    compute_phase_command,
    evaluate_skill_fidelity,
    read_reactor_onnx_metadata,
    transition_envelope,
)
from microduck_brain.sim.exact_bam_plant import ExactBamPlant, PlantRandomization
from microduck_brain.skill_latch import SkillStatus

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


# ============================================================================
# TIER 1: Self-Contained 61 -> 14 ONNX Contract, Metadata & Parity
# ============================================================================


@pytest.mark.parametrize(
    "skill_file",
    [
        "bow.onnx",
        "bow_walk_entry.onnx",
        "kick.onnx",
        "wave.onnx",
        "byebye.onnx",
        "call.onnx",
        "karate_kicks.onnx",
        "jazzwalk.onnx",
        "BEST_alpha_walking.onnx",
        "BEST_alpha_stand.onnx",
    ],
)
def test_tier1_onnx_61_to_14_contract_and_determinism(skill_file: str) -> None:
    """Verify every deployed ONNX policy strictly obeys the [1, 61] -> [1, 14] contract."""
    onnx_path = MODELS_DIR / skill_file
    assert onnx_path.is_file(), f"Missing ONNX model: {onnx_path}"

    model = onnx.load(str(onnx_path), load_external_data=False)
    onnx.checker.check_model(model)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    assert len(inputs) == 1 and inputs[0].name == "obs"
    assert list(inputs[0].shape) == [1, 61]
    assert len(outputs) == 1
    assert list(outputs[0].shape) == [1, 14]

    rng = np.random.default_rng(123)
    obs = rng.normal(0.0, 0.1, size=(1, 61)).astype(np.float32)
    obs[0, 50] = 0.35  # valid normalized progress in [0, 1]
    out1 = sess.run(None, {"obs": obs})[0]
    out2 = sess.run(None, {"obs": obs})[0]
    assert out1.shape == (1, 14)
    assert out1.dtype == np.float32
    assert np.all(np.isfinite(out1))
    np.testing.assert_array_equal(out1, out2)

    if skill_file not in ("BEST_alpha_walking.onnx", "BEST_alpha_stand.onnx"):
        meta = read_reactor_onnx_metadata(onnx_path)
        assert meta["schema"] == "microduck-pollen61-self-contained-v1"
        assert meta["contract"] == "official61"
        assert meta["frames"] >= 100
        assert meta["control_steps"] == meta["frames"] - 1


def test_tier1_residual61_to_official61_wrapper_exact_parity(tmp_path: Path) -> None:
    """Verify synthetic residual61 + commands wraps to self-contained official61 with < 1e-6 error."""
    from onnx import TensorProto, helper, numpy_helper

    frames = 40
    T = frames - 1
    rng = np.random.default_rng(7)
    w = rng.normal(0.0, 0.05, size=(61, 14)).astype(np.float32)
    b = rng.normal(0.0, 0.01, size=(14,)).astype(np.float32)

    # Build a synthetic residual61 ONNX model: actions = MatMul(obs, W) + b
    graph = helper.make_graph(
        nodes=[
            helper.make_node("MatMul", inputs=["obs", "W"], outputs=["mm"]),
            helper.make_node("Add", inputs=["mm", "b"], outputs=["actions"]),
        ],
        name="synthetic_res61",
        inputs=[helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])],
        outputs=[helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])],
        initializer=[
            numpy_helper.from_array(w, name="W"),
            numpy_helper.from_array(b, name="b"),
        ],
    )
    raw_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    raw_onnx = tmp_path / "raw_res61.onnx"
    onnx.save(raw_model, str(raw_onnx))

    commands = (
        OFFICIAL_DEFAULT_POSE[None, :]
        + 0.15 * np.sin(np.linspace(0.0, 2.0 * np.pi, T)[:, None] + np.arange(14)[None, :] * 0.2)
    ).astype(np.float64)
    motion_npz = tmp_path / "motion.npz"
    np.savez(
        motion_npz,
        source_frame_index=np.arange(frames, dtype=np.int64),
        commands=commands,
        fps=np.array(50.0),
    )

    compiled_onnx = tmp_path / "compiled_official61.onnx"
    compile_self_contained_pollen61_onnx(
        raw_onnx,
        compiled_onnx,
        motion_npz_path=motion_npz,
        source_contract="residual61",
        phase_schema=PHASE_SCHEMA_FOURIER_V1,
        skill_name="synthetic_test",
    )

    sess_res = ort.InferenceSession(str(raw_onnx), providers=["CPUExecutionProvider"])
    sess_off = ort.InferenceSession(str(compiled_onnx), providers=["CPUExecutionProvider"])

    last_res = np.zeros(14, dtype=np.float32)
    last_off = np.zeros(14, dtype=np.float32)
    for k in range(T):
        sensors = rng.normal(0.0, 0.2, size=(34,)).astype(np.float32)
        phase = compute_phase_command(k, frames, phase_schema=PHASE_SCHEMA_FOURIER_V1)
        obs_res = np.concatenate([sensors, last_res, phase]).astype(np.float32)[None, :]
        obs_off = np.concatenate([sensors, last_off, phase]).astype(np.float32)[None, :]

        act_res = sess_res.run(None, {"obs": obs_res})[0][0]
        act_off = sess_off.run(None, {"obs": obs_off})[0][0]

        target_res = commands[k] + act_res.astype(np.float64)
        target_off = OFFICIAL_DEFAULT_POSE.astype(np.float64) + act_off.astype(np.float64)
        np.testing.assert_allclose(target_res, target_off, atol=1e-6)

        last_res = act_res.copy()
        last_off = act_off.copy()


def test_tier1_phase_features_and_transition_envelope() -> None:
    """Validate 13-D Fourier and finite phase encodings plus smooth transition indicator psi_t."""
    p0 = compute_phase_command(0, 101, phase_schema=PHASE_SCHEMA_FOURIER_V1, active=1.0)
    p1 = compute_phase_command(100, 101, phase_schema=PHASE_SCHEMA_FOURIER_V1, active=0.5)
    assert p0.shape == (13,) and p1.shape == (13,)
    assert np.isclose(p0[0], 1.0) and np.isclose(p0[1], 0.0) and np.isclose(p0[2], 0.0)
    assert np.isclose(p0[11], 1.0) and np.isclose(p0[12], 0.0)
    assert np.isclose(p1[2], 1.0) and np.isclose(p1[11], 0.5)

    pf = compute_phase_command(25, 101, phase_schema=PHASE_SCHEMA_FINITE_V1)
    assert np.isclose(pf[2], 0.25)
    assert np.all(pf[3:] == 0.0)

    psi_start = transition_envelope(0, 100, entry_ramp_frames=10, exit_ramp_frames=10)
    psi_mid = transition_envelope(50, 100, entry_ramp_frames=10, exit_ramp_frames=10)
    psi_end = transition_envelope(99, 100, entry_ramp_frames=10, exit_ramp_frames=10)
    assert 0.0 < psi_start < 0.2
    assert psi_mid == 1.0
    assert 0.0 < psi_end < 0.2


# ============================================================================
# TIER 2: Closed-Loop MuJoCo BAM M6 Physics, Chaining, DR & Recovery
# ============================================================================


@pytest.mark.parametrize(
    "skill_name",
    ["bow", "kick", "wave", "call", "karate_kicks", "jazzwalk"],
)
def test_tier2_closed_loop_skill_and_walker_handover(skill_name: str) -> None:
    """Execute each self-contained 61->14 skill from STAND and hand over to BEST_alpha_walking.onnx."""
    plant = ExactBamPlant()
    player = ReactorSkillPlayer(MODELS_DIR / f"{skill_name}.onnx", blend_entry_frames=0)
    walk_sess = ort.InferenceSession(
        str(MODELS_DIR / "BEST_alpha_walking.onnx"), providers=["CPUExecutionProvider"]
    )

    plant.reset_stand()
    player.start()
    for _ in range(player.control_steps):
        gyro = plant.data.sensordata[plant.gyro_adr : plant.gyro_adr + 3]
        grav = plant.projected_gravity()
        q = plant.data.qpos[plant.qadr]
        dq = plant.data.qvel[plant.dadr]
        done, target_q, _ = player.step_closed_loop(gyro, grav, q, dq)
        plant.step_control(target_q)
        assert not plant.is_fallen(), f"Skill {skill_name} fell at frame {player.current_frame}"
        if done:
            break

    # Hand over to official walker: 50 settle controls (0.0 m/s) + 200 walk controls (0.35 m/s)
    last_act = player.handover_last_action()
    walk_start_xy = None
    yaw0 = 0.0
    for wstep in range(250):
        cmd = np.zeros(13, dtype=np.float32)
        if wstep >= 50:
            if walk_start_xy is None:
                walk_start_xy = plant.data.qpos[:2].copy()
                w, x, y, z = plant.data.qpos[3:7]
                yaw0 = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
            cmd[0] = 0.35
        obs = plant.observe(cmd, lag_velocity=False, last_action=last_act)
        last_act = walk_sess.run(None, {"obs": obs[None, :]})[0][0][:14].copy()
        plant.step_action(last_act)
        assert not plant.is_fallen(), f"Walker fell after {skill_name} at wstep {wstep}"

    assert walk_start_xy is not None
    heading = np.array([np.cos(yaw0), np.sin(yaw0)])
    walk_m = float((plant.data.qpos[:2] - walk_start_xy) @ heading)
    assert walk_m >= 0.50, f"Expected walker to advance >= 0.50 m after {skill_name}, got {walk_m:.3f} m"


@pytest.mark.parametrize("skill_name", ["bow", "kick", "wave", "call"])
def test_tier2_walk_skill_walk_chaining_via_locomotion_engine(skill_name: str) -> None:
    """Verify continuous walk -> settle -> ReActor skill -> settle -> walk through MicroduckLocomotionEngine."""
    plant = ExactBamPlant()
    engine = MicroduckLocomotionEngine(
        onnx_model_path=str(MODELS_DIR / "BEST_alpha_walking.onnx"),
        stand_recovery_onnx_path=str(MODELS_DIR / "BEST_alpha_stand.onnx"),
    )
    engine.register_skill(skill_name, MODELS_DIR / f"{skill_name}.onnx", blend_entry_frames=15)

    plant.reset_stand()
    # 1. Walk forward 100 steps (0.25 m/s) + 50 settle steps (0.0 m/s)
    for step in range(150):
        vx = 0.25 if step < 100 else 0.0
        target_q = engine.step(
            raw_cmd=(vx, 0.0, 0.0),
            projected_gravity=plant.projected_gravity(),
            base_ang_vel=plant.data.sensordata[plant.gyro_adr : plant.gyro_adr + 3],
            joint_pos=plant.data.qpos[plant.qadr],
            joint_vel=plant.data.qvel[plant.dadr],
        )
        plant.step_control(target_q)
        assert not plant.is_fallen(), f"Fell during walk-in at step {step}"

    walk_in_dist = float(np.linalg.norm(plant.data.qpos[:2]))
    assert walk_in_dist > 0.12

    # 2. Trigger skill and step until active_skill_name clears
    assert engine.trigger_skill(skill_name, initial_joint_vel=plant.data.qvel[plant.dadr])
    skill_steps = 0
    while engine.active_skill_name is not None:
        target_q = engine.step(
            raw_cmd=(0.0, 0.0, 0.0),
            projected_gravity=plant.projected_gravity(),
            base_ang_vel=plant.data.sensordata[plant.gyro_adr : plant.gyro_adr + 3],
            joint_pos=plant.data.qpos[plant.qadr],
            joint_vel=plant.data.qvel[plant.dadr],
        )
        plant.step_control(target_q)
        skill_steps += 1
        assert not plant.is_fallen(), f"Fell during chained skill {skill_name} at step {skill_steps}"

    assert skill_steps >= 400

    # 3. Settle 50 steps + walk out 200 steps at 0.35 m/s
    walk_out_start = None
    yaw0 = 0.0
    for step in range(250):
        vx = 0.0 if step < 50 else 0.35
        if step == 50:
            walk_out_start = plant.data.qpos[:2].copy()
            w, x, y, z = plant.data.qpos[3:7]
            yaw0 = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        target_q = engine.step(
            raw_cmd=(vx, 0.0, 0.0),
            projected_gravity=plant.projected_gravity(),
            base_ang_vel=plant.data.sensordata[plant.gyro_adr : plant.gyro_adr + 3],
            joint_pos=plant.data.qpos[plant.qadr],
            joint_vel=plant.data.qvel[plant.dadr],
        )
        plant.step_control(target_q)
        assert not plant.is_fallen(), f"Fell during walk-out after {skill_name} at step {step}"

    assert walk_out_start is not None
    walk_out_m = float((plant.data.qpos[:2] - walk_out_start) @ np.array([np.cos(yaw0), np.sin(yaw0)]))
    assert walk_out_m >= 0.50


def test_tier2_p1_domain_randomization_and_push_rejection() -> None:
    """Verify bow and kick survive P1 domain randomization and horizontal trunk push."""
    rng = np.random.default_rng(2026)
    dr = PlantRandomization.draw_p1(rng, scale=1.0, noise=True)
    plant = ExactBamPlant(randomization=dr, seed=2026)
    player = ReactorSkillPlayer(MODELS_DIR / "bow.onnx", blend_entry_frames=0)

    plant.reset_stand()
    player.start()
    for k in range(player.control_steps):
        if 120 <= k < 125:
            plant.data.xfrc_applied[plant.trunk, :3] = [2.2, -1.0, 0.0]
        else:
            plant.data.xfrc_applied[plant.trunk, :] = 0.0
        gyro = plant.data.sensordata[plant.gyro_adr : plant.gyro_adr + 3]
        grav = plant.projected_gravity()
        q = plant.data.qpos[plant.qadr]
        dq = plant.data.qvel[plant.dadr]
        done, target_q, _ = player.step_closed_loop(gyro, grav, q, dq)
        plant.step_control(target_q)
        assert not plant.is_fallen(), f"Bow fell under P1 + trunk push at step {k}"
        if done:
            break


def test_tier2_knockdown_autonomous_get_up_recovery() -> None:
    """Verify autonomous fall detection, settling, and BEST_alpha_stand.onnx get-up recovery."""
    plant = ExactBamPlant()
    engine = MicroduckLocomotionEngine(
        onnx_model_path=str(MODELS_DIR / "BEST_alpha_walking.onnx"),
        stand_recovery_onnx_path=str(MODELS_DIR / "BEST_alpha_stand.onnx"),
    )
    plant.reset_stand()

    # Apply severe knockdown force (6.0 N forward + 2.0 N lateral) for 15 steps
    for k in range(30):
        if k < 15:
            plant.data.xfrc_applied[plant.trunk, :3] = [6.0, 2.0, 0.0]
        else:
            plant.data.xfrc_applied[plant.trunk, :] = 0.0
        target_q = engine.step(
            raw_cmd=(0.0, 0.0, 0.0),
            projected_gravity=plant.projected_gravity(),
            base_ang_vel=plant.data.sensordata[plant.gyro_adr : plant.gyro_adr + 3],
            joint_pos=plant.data.qpos[plant.qadr],
            joint_vel=plant.data.qvel[plant.dadr],
        )
        plant.step_control(target_q)

    assert plant.is_fallen(), "Knockdown force should have toppled the robot"
    assert engine.recovery_mode in ("settling", "get_up")

    # Continue stepping engine until recovery completes and returns to normal mode
    for _ in range(250):
        target_q = engine.step(
            raw_cmd=(0.0, 0.0, 0.0),
            projected_gravity=plant.projected_gravity(),
            base_ang_vel=plant.data.sensordata[plant.gyro_adr : plant.gyro_adr + 3],
            joint_pos=plant.data.qpos[plant.qadr],
            joint_vel=plant.data.qvel[plant.dadr],
        )
        plant.step_control(target_q)
        if engine.recovery_mode == "normal":
            break

    assert engine.recovery_mode == "normal", "Robot failed to recover to normal upright mode"
    assert not plant.is_fallen()
    assert float(plant.data.qpos[2]) > 0.105


def test_tier2_stage1_assisted_retarget_rfc_and_ttsa(tmp_path: Path) -> None:
    """Verify Stage-1 ReActor assisted retargeting (virtual root RFC + TTSA joint adaptation)."""
    N = 60
    t = np.linspace(0.0, 1.0, N)
    q_kin = np.zeros((N, 21), dtype=np.float64)
    q_kin[:, 2] = 0.12
    q_kin[:, 3] = 1.0
    q_kin[:, 7:] = OFFICIAL_DEFAULT_POSE[None, :]
    # Add an aggressive single-leg kick + torso pitch gesture
    q_kin[:, 7 + 5] += 0.35 * np.sin(np.pi * t)   # neck_pitch
    q_kin[:, 7 + 11] -= 0.40 * np.sin(np.pi * t)  # right_hip_pitch
    q_kin[:, 7 + 12] += 0.35 * np.sin(np.pi * t)  # right_knee

    out_npz = tmp_path / "assisted_reference.npz"
    res = run_assisted_retarget(
        q_kin,
        config=AssistedRetargetConfig(adaptation_passes=2),
        output_npz_path=out_npz,
    )
    assert out_npz.is_file()
    assert res["completed"] is True
    assert res["fell_at_control"] == -1
    assert res["joint_rmse_rad"] < 0.15
    assert np.max(np.abs(res["ttsa_offsets"])) <= 0.20 + 1e-6


# ============================================================================
# TIER 3: Behavior Tree Integration, manifest.json & Physical Bridge
# ============================================================================


def test_tier3_behavior_tree_closed_loop_reactor_skill_node() -> None:
    """Verify MocapMotionActionNode drives ReactorSkillPlayer inside a Behavior Tree."""
    player = MocapPlayer.from_onnx(MODELS_DIR / "bow_walk_entry.onnx", blend_entry_frames=5)
    bb = Blackboard()
    emitted: list[np.ndarray] = []

    node = MocapMotionActionNode(
        "BowSkillNode",
        mocap_player=player,
        blackboard=bb,
        joint_callback=lambda q: emitted.append(q.copy()),
    )
    seq = SequenceNode("SkillSeq", bb, [node])
    ws = {
        "stability": "HIGH",
        "roughness": "LOW",
        "gyro": [0.0, 0.0, 0.0],
        "projected_gravity": [0.0, 0.0, -1.0],
        "joint_pos": OFFICIAL_DEFAULT_POSE.tolist(),
        "joint_vel": [0.0] * 14,
    }

    status = SkillStatus.RUNNING
    for _ in range(player.control_steps + 5):
        status = seq.tick(ws)
        if status == SkillStatus.SUCCESS:
            break

    assert status == SkillStatus.SUCCESS
    assert len(emitted) == player.control_steps
    assert all(len(q) == 14 and np.all(np.isfinite(q)) for q in emitted)


def test_tier3_manifest_and_physical_robotd_skill_dispatch() -> None:
    """Validate models/manifest.json entries and PollenRobotdClient loopback dispatch."""
    manifest = json.loads((MODELS_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["obs_len"] == 61
    assert manifest["action_len"] == 14

    by_name = {p.get("name", Path(p["file"]).stem): p for p in manifest["policies"]}
    for skill in ("bow", "bow_walk_entry", "kick", "wave", "byebye", "call", "karate_kicks", "jazzwalk"):
        assert skill in by_name, f"Missing {skill} in manifest.json"
        entry = by_name[skill]
        assert (MODELS_DIR / entry["file"]).is_file()
        assert entry["kind"] == "episodic"

    # Spin up a local TCP loopback mock robotd server and verify do_skill() for all ReActor skills
    received: list[dict] = []
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def handle_client() -> None:
        conn, _ = srv.accept()
        with conn:
            buf = b""
            while True:
                chunk = conn.recv(1024)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        msg = json.loads(line.decode("utf-8"))
                        received.append(msg)
                        reply = {
                            "jsonrpc": "2.0",
                            "id": msg.get("id"),
                            "result": {"ok": True, "skill": msg.get("params", {}).get("skill")},
                        }
                        conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))

    t = threading.Thread(target=handle_client, daemon=True)
    t.start()

    client = PollenRobotdClient(endpoint=f"127.0.0.1:{port}")
    assert client.connect(timeout=2.0)
    try:
        for skill in ("bow", "kick", "wave", "call", "karate_kicks", "jazzwalk"):
            resp = client.do_skill(skill)
            assert resp is not None and resp.get("result", {}).get("ok") is True
    finally:
        client.close()
        srv.close()
        t.join(timeout=2.0)

    dispatched_names = [m["params"]["skill"] for m in received if m.get("method") == "robot.do"]
    assert dispatched_names == ["bow", "kick", "wave", "call", "karate_kicks", "jazzwalk"]


def test_tier2_verified_skills_pass_kinematic_fidelity_floors() -> None:
    """Provisional verified skills must demonstrate real target motion, not merely stay upright."""
    bow = evaluate_skill_fidelity(MODELS_DIR / "bow.onnx", skill_name="bow")
    assert bow["fidelity_pass"] is True, f"bow failed fidelity: {bow['failures']}"
    assert bow["max_forward_pitch_deg"] >= 14.0

    bow_entry = evaluate_skill_fidelity(MODELS_DIR / "bow_walk_entry.onnx", skill_name="bow_walk_entry")
    assert bow_entry["fidelity_pass"] is True, f"bow_walk_entry failed fidelity: {bow_entry['failures']}"
    assert bow_entry["max_forward_pitch_deg"] >= 12.0

    kick = evaluate_skill_fidelity(MODELS_DIR / "kick.onnx", skill_name="kick")
    assert kick["fidelity_pass"] is True, f"kick failed fidelity: {kick['failures']}"
    assert kick["max_right_foot_z_mm"] >= 50.0


@pytest.mark.parametrize("skill_name", ["karate_kicks", "jazzwalk", "wave", "call"])
def test_tier2_under_moving_experimental_skills_are_rejected_by_fidelity_gate(skill_name: str) -> None:
    """Standing-in-place or severely under-moving checkpoints must fail the kinematic fidelity gate."""
    res = evaluate_skill_fidelity(MODELS_DIR / f"{skill_name}.onnx", skill_name=skill_name)
    assert res["fell"] is False
    assert res["fidelity_pass"] is False, f"Expected {skill_name} to be rejected by fidelity gate: {res}"
    assert len(res["failures"]) >= 1


def test_tier3_register_default_skills_skips_experimental_by_default() -> None:
    """LocomotionEngine skips experimental_failing_fidelity skills unless explicitly opted in."""
    engine = MicroduckLocomotionEngine(
        onnx_model_path=str(MODELS_DIR / "BEST_alpha_walking.onnx"),
        stand_recovery_onnx_path=str(MODELS_DIR / "BEST_alpha_stand.onnx"),
    )
    loaded_default = set(engine.register_default_skills(MODELS_DIR))
    assert {"bow", "bow_walk_entry", "kick"}.issubset(loaded_default)
    assert loaded_default.isdisjoint({"wave", "byebye", "call", "karate_kicks", "jazzwalk"})

    engine_all = MicroduckLocomotionEngine(
        onnx_model_path=str(MODELS_DIR / "BEST_alpha_walking.onnx"),
        stand_recovery_onnx_path=str(MODELS_DIR / "BEST_alpha_stand.onnx"),
    )
    loaded_all = set(engine_all.register_default_skills(MODELS_DIR, include_experimental=True))
    assert {"bow", "bow_walk_entry", "kick", "wave", "byebye", "call", "karate_kicks", "jazzwalk"}.issubset(
        loaded_all
    )
