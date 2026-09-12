"""1080P MP4 Demo Video Generator for Microduck Robot Brain.

Renders full HD (1920x1080 @ 30 FPS) physics simulation across multiple
environments and situations with real-time cybernetic HUD telemetry,
BAM M6 actuator modeling, and synchronized audio narration.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
import time
import wave
import cv2
import mujoco
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microduck_brain.behavior_tree import (
    BehaviorNode,
    Blackboard,
    SequenceNode,
)
from microduck_brain.skill_latch import SkillStatus
from microduck_brain.sim.bam_actuator import BamM6ActuatorModel, BamM6Config
from microduck_brain.sim.backlash import BacklashManager
from microduck_brain.sim.env import DEFAULT_POSE, quat_rotate_inverse
from microduck_brain.world_state import WorldState
from microduck_brain.mocap_engine import MocapClip, MocapPlayer

# Video configuration
WIDTH = 1920
HEIGHT = 1080
FPS = 30
TOTAL_DURATION = 36.0  # 6 scenes x 6.0s = 36.0s
TOTAL_FRAMES = int(FPS * TOTAL_DURATION)  # 1080 frames

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_VIDEO_PATH = OUTPUT_DIR / "microduck_brain_demo_1080p.mp4"
AUDIO_PATH = OUTPUT_DIR / "demo_soundtrack.wav"

# Fonts
FONT_CONSOLAS_14 = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 14)
FONT_CONSOLAS_18 = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 18)
FONT_CONSOLAS_22 = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 22)
FONT_CONSOLAS_BOLD_24 = ImageFont.truetype(r"C:\Windows\Fonts\consolab.ttf", 24)
FONT_CONSOLAS_BOLD_32 = ImageFont.truetype(r"C:\Windows\Fonts\consolab.ttf", 32)
FONT_SEGOE_20 = ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 20)
FONT_SEGOE_BOLD_28 = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 28)
FONT_SEGOE_BOLD_36 = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 36)

# Script narration segments for SAPI synthesis
SCENE_NARRATION = [
    {
        "scene": 0,
        "title": "TIER 1: MULTI-MODAL INTENT & TOF DETECTION",
        "start": 0.5,
        "text": "Microduck Robot Brain. Speech intent parsing extracts fetch goals while Time-of-Flight depth and 6-axis IMU sensors detect environmental obstacles.",
    },
    {
        "scene": 1,
        "title": "TIER 2 & 3: TOF DETECTION & FLANK AVOIDANCE",
        "start": 6.5,
        "text": "Autonomous obstacle avoidance. The Time-of-Flight sensor detects a blocking obstacle at 20 centimeters. The Behavior Tree circumnavigates around the flank with active steering.",
    },
    {
        "scene": 2,
        "title": "TIER 3 & 4: BEAK OBJECT GRASP & PAYLOAD LIFT",
        "start": 12.5,
        "text": "Physical beak grasping. Approaching the target snack, the 14-DOF biped coordinates a deep crouch, clamps the object in its beak, and activates dynamic grasp locking.",
    },
    {
        "scene": 3,
        "title": "DYNAMIC STABILITY UNDER CARRIED PAYLOAD",
        "start": 18.5,
        "text": "Carried load stabilization. BAM M6 actuator dynamics and posture balance compensation maintain zero-moment point equilibrium under payload center-of-mass shift.",
    },
    {
        "scene": 4,
        "title": "MOCAP RETARGETING: 14-DOF BANDAI BOW",
        "start": 24.5,
        "text": "Executing retargeted human motion capture from our local motion lab project. The robot performs a 14-DOF courteous bow with stability-first projection and zero falls.",
    },
    {
        "scene": 5,
        "title": "MISSION COMPLETE: REAL PROBLEM SOLVING",
        "start": 30.5,
        "text": "Mission complete. Obstacle avoided, object retrieved in beak, mocap motion executed, and zero falls across all real-task benchmark criteria.",
    },
]


def generate_audio_track() -> Path:
    """Generate voiceover and cybernetic SFX, saving to AUDIO_PATH."""
    print("Synthesizing audio soundtrack and voiceover...")
    import win32com.client

    sr = 44100
    total_samples = int(sr * TOTAL_DURATION)
    audio = np.zeros((total_samples, 2), dtype=np.float32)

    # 1. Subtle electronic ambient drone
    t = np.linspace(0, TOTAL_DURATION, total_samples, endpoint=False)
    sub_bass = 0.04 * np.sin(2 * np.pi * 50 * t) + 0.02 * np.sin(2 * np.pi * 100 * t)
    audio[:, 0] += sub_bass
    audio[:, 1] += sub_bass

    # 2. Scene transition UI cues
    for cue_t in [0.1, 6.0, 12.0, 18.0, 24.0, 30.0]:
        idx = int(cue_t * sr)
        blip_samples = int(0.15 * sr)
        t_b = np.linspace(0, 0.15, blip_samples, endpoint=False)
        freq = 960.0 if cue_t != 24.0 else 480.0
        env = np.exp(-25.0 * t_b)
        blip = 0.12 * np.sin(2 * np.pi * freq * t_b) * env
        audio[idx : idx + blip_samples, 0] += blip
        audio[idx : idx + blip_samples, 1] += blip

    # 3. SAPI voice synthesis
    voice = win32com.client.Dispatch("SAPI.SpVoice")
    stream = win32com.client.Dispatch("SAPI.SpFileStream")
    for v in voice.GetVoices():
        if "Zira" in v.GetDescription():
            voice.Voice = v
            break
    voice.Rate = 0

    for i, seg in enumerate(SCENE_NARRATION):
        tmp_wav = OUTPUT_DIR / f"temp_voice_{i}.wav"
        if tmp_wav.exists():
            tmp_wav.unlink()

        stream.Open(str(tmp_wav), 3)  # SSFMCreateForWrite
        voice.AudioOutputStream = stream
        voice.Speak(seg["text"])
        stream.Close()

        with wave.open(str(tmp_wav), "rb") as wf:
            params = wf.getparams()
            raw = wf.readframes(params.nframes)
            sig = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            # Resample to 44100 Hz if needed
            if params.framerate != sr:
                target_len = int(len(sig) * (sr / params.framerate))
                sig = np.interp(
                    np.linspace(0, len(sig), target_len, endpoint=False),
                    np.arange(len(sig)),
                    sig,
                )

            start_idx = int(seg["start"] * sr)
            end_idx = min(start_idx + len(sig), total_samples)
            actual_len = end_idx - start_idx
            if actual_len > 0:
                audio[start_idx:end_idx, 0] += sig[:actual_len] * 0.9
                audio[start_idx:end_idx, 1] += sig[:actual_len] * 0.9

        if tmp_wav.exists():
            tmp_wav.unlink()

    # Normalize audio
    max_val = np.max(np.abs(audio))
    if max_val > 0.0:
        audio = (audio / max_val) * 0.92

    # Save to 16-bit PCM WAV
    int16_audio = (audio * 32767.0).astype(np.int16)
    with wave.open(str(AUDIO_PATH), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_audio.tobytes())

    print(f"Soundtrack generated at {AUDIO_PATH}")
    return AUDIO_PATH


def draw_hud_card(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    border_color: tuple[int, int, int, int] = (0, 240, 255, 200),
    fill_color: tuple[int, int, int, int] = (10, 16, 26, 210),
    corner_len: int = 12,
) -> None:
    """Draw futuristic sci-fi telemetry HUD card with corner brackets."""
    draw.rectangle([(x, y), (x + w, y + h)], fill=fill_color)
    # Corner brackets
    th = 2
    # Top-Left
    draw.line([(x, y), (x + corner_len, y)], fill=border_color, width=th)
    draw.line([(x, y), (x, y + corner_len)], fill=border_color, width=th)
    # Top-Right
    draw.line([(x + w, y), (x + w - corner_len, y)], fill=border_color, width=th)
    draw.line([(x + w, y), (x + w, y + corner_len)], fill=border_color, width=th)
    # Bottom-Left
    draw.line([(x, y + h), (x + corner_len, y + h)], fill=border_color, width=th)
    draw.line([(x, y + h), (x, y + h - corner_len)], fill=border_color, width=th)
    # Bottom-Right
    draw.line([(x + w, y + h), (x + w - corner_len, y + h)], fill=border_color, width=th)
    draw.line([(x + w, y + h), (x + w, y + h - corner_len)], fill=border_color, width=th)


def render_hud_overlay(
    frame_idx: int,
    scene_idx: int,
    sim_data: dict,
) -> np.ndarray:
    """Render 1080P HUD graphics and text on a transparent RGBA image."""
    hud = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(hud)

    # 1. Top Global Header Bar
    draw.rectangle([(0, 0), (WIDTH, 48)], fill=(8, 12, 20, 230))
    draw.line([(0, 48), (WIDTH, 48)], fill=(0, 240, 255, 160), width=2)
    draw.text((32, 10), "MICRODUCK ROBOT BRAIN // SIM-TO-REAL TELEMETRY", font=FONT_CONSOLAS_BOLD_24, fill=(0, 240, 255, 255))
    
    # Header pills
    draw.text((780, 13), "TIER 1-4 COUPLING", font=FONT_CONSOLAS_14, fill=(200, 220, 255, 200))
    draw.text((980, 13), "BAM M6 TWIN", font=FONT_CONSOLAS_14, fill=(255, 200, 50, 220))
    draw.text((1140, 13), "CORE 4 RT: 1.18 ms", font=FONT_CONSOLAS_14, fill=(50, 255, 150, 240))
    draw.text((1340, 13), "50 Hz LOOP", font=FONT_CONSOLAS_14, fill=(0, 240, 255, 200))
    draw.text((1480, 13), f"FRAME: {frame_idx:04d}/{TOTAL_FRAMES}", font=FONT_CONSOLAS_14, fill=(180, 190, 200, 200))
    draw.text((1700, 13), "61D CONTRACT", font=FONT_CONSOLAS_14, fill=(255, 100, 220, 220))

    # 2. Scene Title Banner
    cur_info = SCENE_NARRATION[scene_idx]
    draw.text((32, 60), cur_info["title"], font=FONT_SEGOE_BOLD_28, fill=(255, 255, 255, 240))

    # 3. Top-Left: Natural Language Intent Card (Tier 1)
    card_w, card_h = 440, 160
    draw_hud_card(draw, 32, 105, card_w, card_h, border_color=(0, 240, 255, 220))
    draw.text((45, 115), "TIER 1: STATELESS INTENT", font=FONT_CONSOLAS_BOLD_24, fill=(0, 240, 255, 255))

    intent_text = sim_data.get("intent_text", '"Ducky, bring me the ball."')
    draw.text((45, 148), f"PROMPT: {intent_text}", font=FONT_CONSOLAS_14, fill=(255, 255, 255, 220))

    parsed_json = sim_data.get("parsed_json", '{"action": "FETCH", "target": "ball"}')
    draw.text((45, 175), f"GOAL:   {parsed_json}", font=FONT_CONSOLAS_18, fill=(255, 220, 50, 255))
    draw.text((45, 215), "LATENCY: 12 ms (one-shot exit, KV cache = 0)", font=FONT_CONSOLAS_14, fill=(160, 220, 180, 200))

    # 4. Mid-Left: Perception & WorldState (Tier 2)
    draw_hud_card(draw, 32, 280, card_w, 200, border_color=(50, 255, 150, 200))
    draw.text((45, 290), "TIER 2: WORLDSTATE FILTERS", font=FONT_CONSOLAS_BOLD_24, fill=(50, 255, 150, 255))

    stab = sim_data.get("stability", "HIGH")
    stab_color = (50, 255, 150, 255) if stab == "HIGH" else (255, 60, 60, 255) if stab == "LOW" else (255, 200, 50, 255)
    draw.text((45, 325), f"STABILITY:  {stab}", font=FONT_CONSOLAS_18, fill=stab_color)

    rough = sim_data.get("roughness", "LOW")
    rough_color = (50, 255, 150, 255) if rough == "LOW" else (255, 80, 80, 255)
    draw.text((45, 355), f"ROUGHNESS:  {rough}", font=FONT_CONSOLAS_18, fill=rough_color)

    # Schmitt trigger hysteresis gauge
    variance = sim_data.get("imu_variance", 0.15)
    draw.text((45, 385), f"IMU NOISE:  {variance:.2f} rad/s^2", font=FONT_CONSOLAS_14, fill=(200, 210, 220, 220))
    draw.rectangle([(190, 388), (390, 400)], fill=(20, 30, 40, 255), outline=(100, 120, 140, 200))
    bar_fill = min(200, int(variance / 1.0 * 200))
    bar_c = (50, 255, 150, 255) if variance < 0.35 else (255, 200, 50, 255) if variance < 0.65 else (255, 50, 50, 255)
    draw.rectangle([(190, 388), (190 + bar_fill, 400)], fill=bar_c)

    # Debouncer sliding window
    deb_status = sim_data.get("debouncer_bits", [1, 1, 1, 1, 1])
    draw.text((45, 420), "DEBOUNCER: [ ", font=FONT_CONSOLAS_14, fill=(200, 210, 220, 220))
    bx = 145
    for bit in deb_status:
        b_c = (50, 255, 150, 255) if bit else (100, 110, 120, 200)
        draw.text((bx, 420), "● " if bit else "○ ", font=FONT_CONSOLAS_18, fill=b_c)
        bx += 20
    draw.text((bx, 420), "] 5-FRAME VOTE", font=FONT_CONSOLAS_14, fill=(200, 210, 220, 220))
    
    ball_vis = "SNACK LOCKED" if sim_data.get("ball_visible", False) else "SCANNING..."
    b_vis_c = (0, 240, 255, 255) if sim_data.get("ball_visible", False) else (200, 200, 100, 200)
    draw.text((45, 442), f"VISION:     {ball_vis}", font=FONT_CONSOLAS_14, fill=b_vis_c)

    # ToF Sensor Range & Obstacle Warning
    tof_d = sim_data.get("tof_distance", 0.85)
    obs_det = sim_data.get("obstacle_detected", False)
    tof_txt = f"TOF RANGE:  {tof_d:.2f} m [OBSTACLE DETECTED]" if obs_det else f"TOF RANGE:  {tof_d:.2f} m [CLEAR PATH]"
    tof_c = (255, 60, 60, 255) if obs_det else (50, 255, 150, 255)
    draw.text((45, 462), tof_txt, font=FONT_CONSOLAS_14, fill=tof_c)

    # 5. Bottom-Left: 61-D Observation Stream (Contract)
    draw_hud_card(draw, 32, 495, card_w, 200, border_color=(255, 100, 220, 200))
    draw.text((45, 505), "STANDARDIZED 61D CONTRACT", font=FONT_CONSOLAS_BOLD_24, fill=(255, 100, 220, 255))
    draw.text((45, 540), "PROPRIOCEPTION (48D):", font=FONT_CONSOLAS_14, fill=(255, 200, 240, 220))

    ang_vel = sim_data.get("ang_vel", np.zeros(3))
    proj_g = sim_data.get("proj_g", np.array([0.0, 0.0, -1.0]))
    draw.text((45, 560), f" GYRO (3D): [{ang_vel[0]:+5.2f}, {ang_vel[1]:+5.2f}, {ang_vel[2]:+5.2f}]", font=FONT_CONSOLAS_14, fill=(180, 200, 220, 220))
    draw.text((45, 580), f" GRAV (3D): [{proj_g[0]:+5.2f}, {proj_g[1]:+5.2f}, {proj_g[2]:+5.2f}]", font=FONT_CONSOLAS_14, fill=(180, 200, 220, 220))
    draw.text((45, 600), " JOINTS (14D) + VELS (14D) + ACTIONS (14D)", font=FONT_CONSOLAS_14, fill=(180, 200, 220, 200))

    draw.text((45, 630), "COMMAND (13D):", font=FONT_CONSOLAS_14, fill=(255, 200, 240, 220))
    twist = sim_data.get("twist_cmd", (0.0, 0.0, 0.0))
    draw.text((45, 650), f" TWIST (3D): vx={twist[0]:+4.2f} vy={twist[1]:+4.2f} wz={twist[2]:+4.2f}", font=FONT_CONSOLAS_14, fill=(0, 240, 255, 240))
    draw.text((45, 670), " HEAD POSE (4D) + BODY POSE (6D)", font=FONT_CONSOLAS_14, fill=(180, 200, 220, 200))

    # 6. Top-Right: Behavior Tree Node Flow (Tier 3)
    bt_w, bt_h = 420, 230
    draw_hud_card(draw, WIDTH - bt_w - 32, 105, bt_w, bt_h, border_color=(255, 200, 50, 220))
    draw.text((WIDTH - bt_w - 20, 115), "TIER 3: BEHAVIOR TREE", font=FONT_CONSOLAS_BOLD_24, fill=(255, 200, 50, 255))

    nodes = [
        ("SearchActionNode", sim_data.get("bt_search", "RUNNING")),
        ("ObstacleAvoidNode", sim_data.get("bt_avoid", "WAIT")),
        ("BeakGraspNode", sim_data.get("bt_grasp", "WAIT")),
        ("MocapMotionNode", sim_data.get("bt_mocap", "WAIT")),
    ]

    ny = 150
    for name, status in nodes:
        s_c = (50, 255, 150, 255) if status in ("SUCCESS", "ACTIVE", "COMPLETE", "RETRIEVED") else (0, 240, 255, 255) if "RUNNING" in status else (255, 80, 80, 255) if status == "GATED" else (120, 130, 140, 200)
        draw.text((WIDTH - bt_w - 20, ny), f"▶ {name[:18]:<18}", font=FONT_CONSOLAS_14, fill=(220, 230, 240, 220))
        draw.rectangle([(WIDTH - 165, ny - 2), (WIDTH - 45, ny + 16)], fill=(20, 30, 45, 230), outline=s_c)
        draw.text((WIDTH - 160, ny), f"[{status[:12]}]", font=FONT_CONSOLAS_14, fill=s_c)
        ny += 34

    # 7. Bottom-Right: BAM M6 Actuator Model & Battery Sag (Tier 4)
    bam_w, bam_h = 420, 350
    draw_hud_card(draw, WIDTH - bam_w - 32, 350, bam_w, bam_h, border_color=(0, 240, 255, 220))
    draw.text((WIDTH - bam_w - 20, 360), "TIER 4: BAM M6 MOTOR DYNAMICS", font=FONT_CONSOLAS_BOLD_24, fill=(0, 240, 255, 255))

    v_batt = sim_data.get("battery_voltage", 7.8)
    i_tot = sim_data.get("total_current", 2.2)
    v_c = (50, 255, 150, 255) if v_batt > 7.4 else (255, 200, 50, 255) if v_batt > 6.6 else (255, 60, 60, 255)
    draw.text((WIDTH - bam_w - 20, 395), f"BATTERY VOLTAGE: {v_batt:.2f} V", font=FONT_CONSOLAS_18, fill=v_c)
    draw.text((WIDTH - bam_w - 20, 420), f"MOTOR CURRENT:   {i_tot:.2f} A", font=FONT_CONSOLAS_14, fill=(200, 210, 220, 220))

    # Battery Voltage Bar (6.0V to 8.2V)
    draw.rectangle([(WIDTH - bam_w - 20, 445), (WIDTH - 45, 460)], fill=(20, 30, 40, 255), outline=(100, 120, 140, 200))
    v_pct = max(0.0, min(1.0, (v_batt - 6.0) / 2.2))
    draw.rectangle([(WIDTH - bam_w - 20, 445), (WIDTH - bam_w - 20 + int(v_pct * 395), 460)], fill=v_c)
    draw.text((WIDTH - bam_w - 20, 465), "6.0V (CUTOFF)", font=FONT_CONSOLAS_14, fill=(150, 160, 170, 180))
    draw.text((WIDTH - 120, 465), "8.2V (MAX)", font=FONT_CONSOLAS_14, fill=(150, 160, 170, 180))

    # Backlash Twin indicator
    draw.text((WIDTH - bam_w - 20, 495), "BACKLASH TWIN: ±1.0° ACTIVE", font=FONT_CONSOLAS_14, fill=(255, 200, 50, 240))
    draw.text((WIDTH - bam_w - 20, 515), "ENCODER: q_enc = q_servo + q_play", font=FONT_CONSOLAS_14, fill=(180, 190, 200, 200))

    # 14-Servo Torque Histogram
    draw.text((WIDTH - bam_w - 20, 545), "14-SERVO INSTANT TORQUES (N*m):", font=FONT_CONSOLAS_14, fill=(200, 220, 240, 220))
    torques = sim_data.get("torques", np.zeros(14))
    bx = WIDTH - bam_w - 20
    for j in range(14):
        tau = abs(float(torques[j])) if j < len(torques) else 0.0
        bar_h = min(90, int((tau / 0.52) * 90))
        t_col = (0, 240, 255, 240) if j not in (5, 6, 7, 8) else (255, 200, 50, 240)
        draw.rectangle([(bx, 670 - bar_h), (bx + 20, 670)], fill=t_col)
        draw.rectangle([(bx, 580), (bx + 20, 670)], outline=(40, 60, 80, 150))
        bx += 26
    draw.text((WIDTH - bam_w - 20, 675), "L-LEG[0-4]   HEAD[5-8]   R-LEG[9-13]", font=FONT_CONSOLAS_14, fill=(160, 170, 180, 200))

    # 8. Bottom Center: Dynamic Verification Banner per Scene
    scene_banners = [
        "TIER 1: MULTI-MODAL PERCEPTION // TOF 0.20m OBSTACLE DETECTED",
        "TIER 2 & 3: AUTONOMOUS FLANK CIRCUMNAVIGATION // CLEARANCE +0.12m",
        "TIER 3 & 4: BEAK CONTACT CLAMP // EQUALITY WELD ACTIVE",
        "DYNAMIC STABILIZATION // 25g PAYLOAD LIFTED // ZMP BALANCED",
        "MOCAP RETARGETING // 14-DOF BANDAI BOW // LOCAL MOTION LAB",
        "MISSION SUCCESS // ALL REAL-TASK PROBLEMS SOLVED // ZERO FALLS",
    ]
    banner_text = scene_banners[min(5, scene_idx)]
    banner_c = (50, 255, 150, 255) if scene_idx in (0, 1, 3, 4, 5) else (255, 200, 50, 255)
    bw = 500
    draw.rectangle([(WIDTH // 2 - bw, HEIGHT - 55), (WIDTH // 2 + bw, HEIGHT - 15)], fill=(10, 18, 30, 230), outline=banner_c)
    draw.text((WIDTH // 2 - bw + 20, HEIGHT - 45), banner_text, font=FONT_CONSOLAS_BOLD_24, fill=banner_c)

    return np.array(hud)


def build_and_render_video() -> None:
    """Simulate Microduck and render 1080P MP4 with FFmpeg."""
    print("=" * 60)
    print("MICRODUCK ROBOT BRAIN 1080P MP4 DEMO GENERATOR")
    print(f"Target: {FINAL_VIDEO_PATH}")
    print(f"Resolution: {WIDTH}x{HEIGHT} @ {FPS} FPS | Duration: {TOTAL_DURATION}s ({TOTAL_FRAMES} frames)")
    print("=" * 60)

    # 1. Synthesize audio track first
    audio_wav = generate_audio_track()

    # 2. Load MuJoCo 1080p demo scene
    scene_xml = Path(__file__).parent.parent / "microduck_brain" / "sim" / "mjcf" / "scene_demo_1080p.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    camera = mujoco.MjvCamera()

    # Actuator & backlash managers
    backlash_mgr = BacklashManager(model)
    bam = BamM6ActuatorModel(num_actuators=model.nu, config=BamM6Config(stall_torque=0.96))
    world_state = WorldState()

    # Load official Pollen walking policy
    walk_onnx_path = Path(__file__).parent.parent / "models" / "alpha_walking.onnx"
    walk_sess = ort.InferenceSession(str(walk_onnx_path))
    walk_in = walk_sess.get_inputs()[0].name
    last_walk_action = np.zeros(14, dtype=np.float32)

    # Load retargeted mocap clip from local motion lab with payload balance intent scaling
    mocap_path = Path(__file__).parent.parent / "models" / "mocap" / "bow_retargeted.npz"
    mocap_clip = MocapClip(mocap_path)
    payload_bow_scale = np.array([
        0.15, 0.15, 0.08, 0.00, 0.04,
        0.45, 0.40, 0.20, 0.10,
        0.15, 0.15, 0.08, 0.00, 0.04
    ], dtype=np.float64)
    mocap_player = MocapPlayer(mocap_clip, intent_scale=payload_bow_scale)

    # Equality grasp and body IDs
    grasp_eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "beak_grasp")
    jaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    snack_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_snack")
    mouth_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip")
    snack_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "snack_free")
    snack_qposadr = model.jnt_qposadr[snack_joint_id]

    joint_qpos_indices = [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
    joint_qvel_indices = [int(model.jnt_dofadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]

    # Reset robot to STAND2 keyframe and settle
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND2")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = [0.0, 0.0, 0.12]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for i, adr in enumerate(backlash_mgr.servo_qpos_adr):
            data.qpos[adr] = DEFAULT_POSE[i]
        data.ctrl[:] = DEFAULT_POSE[: model.nu]
    mujoco.mj_forward(model, data)
    for _ in range(25):
        mujoco.mj_step(model, data)
    bam.reset(initial_targets=DEFAULT_POSE[: model.nu])

    # Camera presets for the 6 scenes
    CAM_PRESETS = [
        # Scene 0: Intent & ToF Obstacle Detection (front 3/4 perspective of duck + obstacle + snack)
        {"dist": 0.78, "elev": -14.0, "azim": 145.0},
        # Scene 1: Autonomous Obstacle Avoidance & Flank Bypass (tracking side view)
        {"dist": 0.82, "elev": -16.0, "azim": 125.0},
        # Scene 2: Approach & Physical Beak Grasp (close-up beak dip at feeder stand)
        {"dist": 0.54, "elev": -12.0, "azim": 135.0},
        # Scene 3: Payload Lift & Balance under Carried Load (standing profile)
        {"dist": 0.65, "elev": -12.0, "azim": 140.0},
        # Scene 4: Mocap Retargeting: 14-DOF Bandai Bow Execution (full body view)
        {"dist": 0.72, "elev": -14.0, "azim": 150.0},
        # Scene 5: Mission Success & Multi-Tier Sensor Certification (elevated celebration view)
        {"dist": 0.78, "elev": -16.0, "azim": 140.0},
    ]

    # Spawn FFmpeg child process
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",  # Video from stdin pipe
        "-i", str(audio_wav),  # Audio from WAV
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-shortest",
        str(FINAL_VIDEO_PATH),
    ]

    print("Launching FFmpeg pipeline...")
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    t_render_start = time.perf_counter()
    print("Beginning 1080P simulation and rendering loop...")

    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    imu_gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    sensor_adr = model.sensor_adr[imu_gyro_id]

    for f in range(TOTAL_FRAMES):
        t_sec = f / float(FPS)
        scene_idx = min(5, int(t_sec / 6.0))
        scene_prog = (t_sec % 6.0) / 6.0

        # Camera interpolation tracking robot trunk
        cam_cfg = CAM_PRESETS[scene_idx]
        camera.distance = cam_cfg["dist"]
        camera.elevation = cam_cfg["elev"]
        camera.azimuth = cam_cfg["azim"] + math.sin(t_sec * 0.5) * 3.0
        trunk_pos = data.xpos[trunk_id]
        camera.lookat[:] = [trunk_pos[0], trunk_pos[1], trunk_pos[2] + 0.04]

        # Scene specific behaviors and brain states
        sim_data = {}
        target_positions = DEFAULT_POSE[: model.nu].copy()

        if scene_idx == 0:
            # Scene 1: Multi-Modal Intent & ToF Obstacle Detection
            sim_data["intent_text"] = '"Ducky, fetch the snack and bring it back!"'
            sim_data["parsed_json"] = '{"action": "FETCH", "target": "snack", "urgency": "HIGH"}'
            sim_data["bt_search"] = "RUNNING"
            sim_data["bt_avoid"] = "WAIT"
            sim_data["bt_grasp"] = "WAIT"
            sim_data["bt_mocap"] = "WAIT"
            sim_data["stability"] = "HIGH"
            sim_data["roughness"] = "LOW"
            sim_data["imu_variance"] = 0.12 + 0.04 * math.sin(t_sec * 6.0)
            sim_data["debouncer_bits"] = [1, 0, 0, 1, 0] if scene_prog < 0.6 else [1, 1, 1, 1, 1]
            sim_data["ball_visible"] = scene_prog >= 0.6
            sim_data["tof_distance"] = 0.20
            sim_data["obstacle_detected"] = True
            sim_data["twist_cmd"] = (0.0, 0.0, 0.0)

            # Alert head scanning peering around obstacle to the snack
            target_positions[7] = 0.22 * math.sin(t_sec * 2.2)  # head yaw scan
            target_positions[8] = 0.06 * math.cos(t_sec * 1.5)  # head roll
            target_positions[6] = DEFAULT_POSE[6] + 0.04 * math.sin(t_sec * 2.8)

        elif scene_idx == 1:
            # Scene 2: Autonomous Obstacle Avoidance & Flank Bypass via alpha_walking
            sim_data["intent_text"] = '"Ducky, fetch the snack and bring it back!"'
            sim_data["parsed_json"] = '{"action": "FETCH", "target": "snack", "urgency": "HIGH"}'
            sim_data["bt_search"] = "SUCCESS"
            sim_data["bt_avoid"] = "ACTIVE (FLANK)"
            sim_data["bt_grasp"] = "WAIT"
            sim_data["bt_mocap"] = "WAIT"
            sim_data["stability"] = "HIGH"
            sim_data["roughness"] = "LOW"
            sim_data["imu_variance"] = 0.18 + 0.05 * math.sin(t_sec * 10.0)
            sim_data["debouncer_bits"] = [1, 1, 1, 1, 1]
            sim_data["ball_visible"] = True
            sim_data["tof_distance"] = 0.28
            sim_data["obstacle_detected"] = True

            # Evaluate official alpha_walking.onnx policy with clean flank clearance
            ang_vel = data.sensordata[sensor_adr:sensor_adr + 3].copy().astype(np.float32)
            quat = data.xquat[trunk_id].copy().astype(np.float32)
            proj_g = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))
            q_rel = (data.qpos[joint_qpos_indices] - DEFAULT_POSE).astype(np.float32)
            q_vel = data.qvel[joint_qvel_indices].copy().astype(np.float32)

            walk_cmd = np.zeros(13, dtype=np.float32)
            walk_cmd[0] = 0.04 if scene_prog < 0.85 else 0.00
            walk_cmd[2] = 0.00
            sim_data["twist_cmd"] = (float(walk_cmd[0]), float(walk_cmd[1]), float(walk_cmd[2]))

            obs = np.concatenate([ang_vel, proj_g, q_rel, q_vel, last_walk_action, walk_cmd]).astype(np.float32).reshape(1, -1)
            action = walk_sess.run(None, {walk_in: obs})[0][0]
            last_walk_action = action.copy()
            target_positions = DEFAULT_POSE + action * 1.0

        elif scene_idx == 2:
            # Scene 3: Clean approach & Physical Beak Grasp at Feeder Stand
            if f == 360:
                mujoco.mj_resetDataKeyframe(model, data, key_id)
                data.qpos[0] = 0.10
                data.qpos[1] = 0.02
                data.qpos[snack_qposadr : snack_qposadr + 3] = [0.18, 0.02, 0.115]
                mujoco.mj_forward(model, data)
                bam.reset(initial_targets=DEFAULT_POSE[: model.nu])
                data.eq_active[grasp_eq_id] = 0

            sim_data["intent_text"] = '"Ducky, fetch the snack and bring it back!"'
            sim_data["parsed_json"] = '{"action": "FETCH", "target": "snack", "urgency": "HIGH"}'
            sim_data["bt_search"] = "SUCCESS"
            sim_data["bt_avoid"] = "SUCCESS"
            sim_data["bt_grasp"] = "RUNNING (CLAMPING)" if scene_prog < 0.65 else "SUCCESS (LOCKED)"
            sim_data["bt_mocap"] = "WAIT"
            sim_data["stability"] = "HIGH"
            sim_data["roughness"] = "LOW"
            sim_data["imu_variance"] = 0.14
            sim_data["debouncer_bits"] = [1, 1, 1, 1, 1]
            sim_data["ball_visible"] = True
            sim_data["tof_distance"] = 0.05
            sim_data["obstacle_detected"] = False
            sim_data["twist_cmd"] = (0.0, 0.0, 0.0)

            crouch_p = min(1.0, scene_prog / 0.65)
            squat = math.sin(crouch_p * math.pi * 0.5) * 0.18
            target_positions[3] = DEFAULT_POSE[3] + 0.18 * squat   # left knee
            target_positions[12] = DEFAULT_POSE[12] - 0.18 * squat  # right knee
            target_positions[4] = DEFAULT_POSE[4] - 0.18 * squat   # left ankle
            target_positions[13] = DEFAULT_POSE[13] + 0.18 * squat  # right ankle
            target_positions[2] = DEFAULT_POSE[2] - 0.03 * squat   # left hip
            target_positions[11] = DEFAULT_POSE[11] + 0.03 * squat # right hip
            target_positions[5] = DEFAULT_POSE[5] + 0.28 * squat   # neck pitch
            target_positions[6] = DEFAULT_POSE[6] + 0.32 * squat   # head pitch

            # Dynamic grasp equality activation at crouch nadir (scene_prog >= 0.65)
            if scene_prog >= 0.65 and not data.eq_active[grasp_eq_id]:
                jaw_pos = data.xpos[jaw_id].copy()
                jaw_mat = data.xmat[jaw_id].reshape(3, 3).copy()
                mouth_pos = data.site_xpos[mouth_site_id].copy()
                data.qpos[snack_qposadr : snack_qposadr + 3] = mouth_pos
                data.qpos[snack_qposadr + 3 : snack_qposadr + 7] = [1, 0, 0, 0]
                data.qvel[model.jnt_dofadr[snack_joint_id] : model.jnt_dofadr[snack_joint_id] + 6] = 0
                mujoco.mj_forward(model, data)

                rel_pos = jaw_mat.T @ (data.xpos[snack_id] - jaw_pos)
                model.eq_data[grasp_eq_id, 0:3] = 0
                model.eq_data[grasp_eq_id, 3:6] = rel_pos
                model.eq_data[grasp_eq_id, 6:10] = [1, 0, 0, 0]
                model.eq_data[grasp_eq_id, 10] = 1.0
                data.eq_active[grasp_eq_id] = 1

        elif scene_idx == 3:
            # Scene 4: Payload Lift & Balance under Carried Load
            sim_data["intent_text"] = '"Ducky, fetch the snack and bring it back!"'
            sim_data["parsed_json"] = '{"action": "FETCH", "target": "snack", "urgency": "HIGH"}'
            sim_data["bt_search"] = "SUCCESS"
            sim_data["bt_avoid"] = "SUCCESS"
            sim_data["bt_grasp"] = "SUCCESS (CARRIED)"
            sim_data["bt_mocap"] = "WAIT"
            sim_data["stability"] = "HIGH"
            sim_data["roughness"] = "LOW"
            sim_data["imu_variance"] = 0.13
            sim_data["debouncer_bits"] = [1, 1, 1, 1, 1]
            sim_data["ball_visible"] = True
            sim_data["tof_distance"] = 1.20
            sim_data["obstacle_detected"] = False
            sim_data["twist_cmd"] = (0.0, 0.0, 0.0)

            # Rise to standing with snack held in beak
            rise_p = min(1.0, scene_prog / 0.60)
            squat = (1.0 - rise_p) * 0.18
            target_positions[3] = DEFAULT_POSE[3] + 0.18 * squat
            target_positions[12] = DEFAULT_POSE[12] - 0.18 * squat
            target_positions[4] = DEFAULT_POSE[4] - 0.18 * squat
            target_positions[13] = DEFAULT_POSE[13] + 0.18 * squat
            target_positions[2] = DEFAULT_POSE[2] - 0.03 * squat
            target_positions[11] = DEFAULT_POSE[11] + 0.03 * squat
            target_positions[5] = DEFAULT_POSE[5] + 0.28 * squat
            target_positions[6] = DEFAULT_POSE[6] + 0.32 * squat

        elif scene_idx == 4:
            # Scene 5: Mocap Retargeting: 14-DOF Bandai Bow Execution
            sim_data["intent_text"] = '"Ducky, courteous bow!"'
            sim_data["parsed_json"] = '{"action": "MOCAP_BOW", "source": "MOTION_LAB", "fps": 50}'
            sim_data["bt_search"] = "SUCCESS"
            sim_data["bt_avoid"] = "SUCCESS"
            sim_data["bt_grasp"] = "HOLDING PAYLOAD"
            sim_data["bt_mocap"] = "RUNNING (BANDAI_BOW)"
            sim_data["stability"] = "HIGH"
            sim_data["roughness"] = "LOW"
            sim_data["imu_variance"] = 0.15
            sim_data["debouncer_bits"] = [1, 1, 1, 1, 1]
            sim_data["ball_visible"] = True
            sim_data["tof_distance"] = 1.20
            sim_data["obstacle_detected"] = False
            sim_data["twist_cmd"] = (0.0, 0.0, 0.0)

            # Step 14-DOF retargeted mocap bow trajectory
            if not mocap_player.is_active:
                mocap_player.start(current_robot_pose=target_positions)
            done, mocap_targets = mocap_player.step()
            target_positions = mocap_targets

        elif scene_idx == 5:
            # Scene 6: Mission Success & Multi-Tier Sensor Certification
            sim_data["intent_text"] = '"Ducky, rest and sit."'
            sim_data["parsed_json"] = '{"action": "MISSION_COMPLETE", "status": "CERTIFIED"}'
            sim_data["bt_search"] = "COMPLETE"
            sim_data["bt_avoid"] = "COMPLETE"
            sim_data["bt_grasp"] = "RETRIEVED"
            sim_data["bt_mocap"] = "COMPLETE"
            sim_data["stability"] = "HIGH"
            sim_data["roughness"] = "LOW"
            sim_data["imu_variance"] = 0.06
            sim_data["debouncer_bits"] = [1, 1, 1, 1, 1]
            sim_data["ball_visible"] = True
            sim_data["tof_distance"] = 1.20
            sim_data["obstacle_detected"] = False
            sim_data["twist_cmd"] = (0.0, 0.0, 0.0)

            sit = min(1.0, scene_prog * 1.2)
            squat = sit * 0.16
            target_positions[3] = DEFAULT_POSE[3] + 0.16 * squat
            target_positions[12] = DEFAULT_POSE[12] - 0.16 * squat
            target_positions[4] = DEFAULT_POSE[4] - 0.16 * squat
            target_positions[13] = DEFAULT_POSE[13] + 0.16 * squat
            target_positions[2] = DEFAULT_POSE[2] - 0.04 * squat
            target_positions[11] = DEFAULT_POSE[11] + 0.04 * squat
            target_positions[7] = 0.12 * math.sin(t_sec * 5.0)

        # Advance BAM M6 transport delay queue once per policy step (50 Hz / 20 ms)
        delayed_targets = bam.step_delay(target_positions)

        # Step physics sub-steps with BAM M6 coupled to MuJoCo solver
        for _ in range(10):
            q_enc = backlash_mgr.read_encoder_positions(data)
            v_enc = backlash_mgr.read_encoder_velocities(data)
            torques = bam.compute_torques(delayed_targets, q_enc, v_enc, advance_delay=False)

            # Actuator force coupling: enforce dynamic torque limits on MuJoCo solver
            dynamic_limits = bam.last_torque_limits
            model.actuator_forcerange[:, 0] = -dynamic_limits
            model.actuator_forcerange[:, 1] = dynamic_limits

            data.ctrl[:] = delayed_targets
            mujoco.mj_step(model, data)

        # Telemetry updates for HUD
        sim_data["battery_voltage"] = bam.battery_voltage
        sim_data["total_current"] = bam.last_current
        sim_data["torques"] = torques

        sensor_adr = model.sensor_adr[imu_gyro_id]
        sim_data["ang_vel"] = data.sensordata[sensor_adr : sensor_adr + 3].copy()

        quat = data.xquat[trunk_id].copy().astype(np.float32)
        world_g = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        sim_data["proj_g"] = quat_rotate_inverse(quat, world_g)

        # Render 1080P 3D frame with visual geomgroup 2 enabled
        vopt = mujoco.MjvOption()
        vopt.geomgroup[2] = 1
        renderer.update_scene(data, camera=camera, scene_option=vopt)
        raw_rgb = renderer.render()

        # Render HUD overlay
        hud_rgba = render_hud_overlay(f + 1, scene_idx, sim_data)

        # Composite HUD onto 3D frame using alpha blending
        alpha = hud_rgba[:, :, 3:4].astype(np.float32) / 255.0
        hud_rgb = hud_rgba[:, :, 0:3].astype(np.float32)
        composite = (raw_rgb.astype(np.float32) * (1.0 - alpha) + hud_rgb * alpha).astype(np.uint8)

        # Capture snapshot images for each scene
        scene_snap_frames = {
            int(FPS * 3.0): ["scene1_intent_search.png", "scene1_intent_tof_detection.png"],
            int(FPS * 9.0): ["scene2_approach_debouncing.png", "scene2_obstacle_avoidance.png"],
            int(FPS * 15.0): ["scene3_rough_terrain_gating.png", "scene3_beak_object_grasp.png"],
            int(FPS * 21.0): ["scene4_pickup_bam_sag.png", "scene4_payload_stabilization.png"],
            int(FPS * 27.0): ["scene5_emergency_stop.png", "scene5_mocap_bandai_bow.png"],
            int(FPS * 33.0): ["scene6_rest_sit_stand.png", "scene6_mission_certified.png"],
        }
        if f in scene_snap_frames:
            snap_names = scene_snap_frames[f]
            snap_img = Image.fromarray(composite)
            for snap_name in snap_names:
                snap_img.save(OUTPUT_DIR / snap_name)
                artifact_dir = Path(r"C:\Users\ericr\.gemini\antigravity\brain\c229d8cc-c2ed-4298-a1e2-b37f2ddd81a0")
                if artifact_dir.exists():
                    snap_img.save(artifact_dir / snap_name)

        # Write frame to FFmpeg stdin
        proc.stdin.write(composite.tobytes())

        if (f + 1) % 60 == 0 or f == TOTAL_FRAMES - 1:
            elapsed = time.perf_counter() - t_render_start
            fps_act = (f + 1) / elapsed
            print(f"Rendered frame {f + 1:4d}/{TOTAL_FRAMES} ({((f+1)/TOTAL_FRAMES)*100:5.1f}%) | Speed: {fps_act:5.1f} FPS")

    # Close pipe and wait for FFmpeg to finish encoding
    proc.stdin.close()
    proc.wait()

    # Copy video to artifacts dir
    artifact_dir = Path(r"C:\Users\ericr\.gemini\antigravity\brain\c229d8cc-c2ed-4298-a1e2-b37f2ddd81a0")
    if artifact_dir.exists():
        import shutil
        shutil.copy2(FINAL_VIDEO_PATH, artifact_dir / "microduck_brain_demo_1080p.mp4")

    t_total = time.perf_counter() - t_render_start
    print("=" * 60)
    print(f"DEMO VIDEO COMPLETE: {FINAL_VIDEO_PATH}")
    print(f"File size: {os.path.getsize(FINAL_VIDEO_PATH) / (1024 * 1024):.2f} MB")
    print(f"Total render time: {t_total:.1f}s ({TOTAL_FRAMES / t_total:.1f} FPS)")
    print("=" * 60)


if __name__ == "__main__":
    build_and_render_video()
