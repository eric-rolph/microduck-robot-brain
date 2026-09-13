"""Voice Interface & Mission Audio Mixer for Microduck.

Synthesizes human speech commands via Windows Speech Synthesis and coordinates
authentic Microduck procedural vocalizations into a synchronized multi-track soundtrack.
"""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile

from microduck_brain.audio.personality import Personality
from microduck_brain.audio import recipes
from microduck_brain.audio import microduck_synth as s


def synthesize_human_command(
    text: str,
    output_path: str | Path,
    voice_name: str = "Microsoft Zira Desktop",
) -> Path:
    """Synthesize human speech command to a WAV file using Windows Speech Synthesizer."""
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    escaped_text = text.replace("'", "''")
    ps_script = (
        f"Add-Type -AssemblyName System.Speech; "
        f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"try {{ $s.SelectVoice('{voice_name}') }} catch {{ }}; "
        f"$s.Rate = 0; "
        f"$s.SetOutputToWaveFile('{str(out)}'); "
        f"$s.Speak('{escaped_text}'); "
        f"$s.Dispose()"
    )

    cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def overlay_audio(
    master: np.ndarray,
    clip: np.ndarray,
    start_time_s: float,
    gain: float = 1.0,
    sr: int = s.SR,
) -> None:
    """In-place additive overlay of an audio clip onto the master track at start_time_s."""
    start_idx = int(round(start_time_s * sr))
    end_idx = start_idx + len(clip)
    if start_idx >= len(master):
        return
    clip_slice = clip[: min(len(clip), len(master) - start_idx)]
    master[start_idx : start_idx + len(clip_slice)] += (clip_slice * gain).astype(np.float32)


def generate_mission_audio_track(
    output_path: str | Path,
    total_duration_s: float = 48.0,
    duck_seed: int = 42,
    human_command_text: str = (
        "Microduck, scan the room, navigate past the red obstacle, "
        "and retrieve the marker for deposit!"
    ),
) -> Path:
    """Generate the complete 48.0s synchronized multi-track audio soundtrack.

    Combines:
    - User synthesized voice command
    - Authentic Microduck procedural vocalizations (greet, inquire, peck, chirp, coo, wheee)
    - Ambient servo motor background hum
    - Mechanical dry-erase marker acrylic container drop transient
    """
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    sr = s.SR
    n_samples = int(round(total_duration_s * sr))
    master = np.zeros(n_samples, dtype=np.float32)

    # 1. Synthesize user voice command to temporary file
    temp_cmd_wav = out.parent / "temp_user_command.wav"
    synthesize_human_command(human_command_text, temp_cmd_wav)

    cmd_sr, cmd_data = wavfile.read(str(temp_cmd_wav))
    if cmd_data.ndim > 1:
        cmd_data = cmd_data[:, 0]
    cmd_float = cmd_data.astype(np.float32) / 32768.0

    # Resample to 48kHz if needed
    if cmd_sr != sr:
        target_len = int(round(len(cmd_float) * sr / cmd_sr))
        cmd_resampled = np.interp(
            np.linspace(0, len(cmd_float), target_len, endpoint=False),
            np.arange(len(cmd_float)),
            cmd_float,
        ).astype(np.float32)
    else:
        cmd_resampled = cmd_float

    # Overlay user speech at t = 0.5s
    overlay_audio(master, cmd_resampled, start_time_s=0.5, gain=0.95, sr=sr)

    # 2. Generate Microduck procedural vocalizations from seed personality
    personality = Personality.from_seed(duck_seed)

    # t = 4.2s: Microduck acknowledges command with friendly 'greet' quack ("wak-wak")
    v_greet = recipes.greet(personality, variant=0)
    overlay_audio(master, v_greet, start_time_s=4.2, gain=0.85, sr=sr)

    # t = 8.8s: Microduck visual lock on marker, emits inquisitive 'inquire' chirp
    v_inquire = recipes.inquire(personality, variant=1)
    overlay_audio(master, v_inquire, start_time_s=8.8, gain=0.85, sr=sr)

    # t = 11.2s: Ground crouch jaw clamp: percussive peck + satisfied chirp
    v_peck = recipes.peck(personality, variant=0)
    v_chirp = recipes.chirp(personality, variant=2)
    overlay_audio(master, v_peck, start_time_s=11.2, gain=0.80, sr=sr)
    overlay_audio(master, v_chirp, start_time_s=11.8, gain=0.85, sr=sr)

    # t = 20.0s: Navigating chicane: gentle low-frequency 'coo'
    v_coo = recipes.coo(personality, variant=0)
    overlay_audio(master, v_coo, start_time_s=20.0, gain=0.75, sr=sr)

    # t = 31.2s: Marker drops into container - plastic click transient
    click_t = s.t_axis(0.15)
    rng_click = np.random.default_rng(1234)
    plastic_click = (
        np.sin(2.0 * math.pi * 1450.0 * click_t) * np.exp(-click_t / 0.015)
        + 0.5 * np.sin(2.0 * math.pi * 3200.0 * click_t) * np.exp(-click_t / 0.008)
    ).astype(np.float32)
    plastic_click += 0.2 * rng_click.standard_normal(len(click_t)).astype(np.float32) * np.exp(-click_t / 0.01)
    plastic_click = s.normalise(plastic_click, -4.0)
    overlay_audio(master, plastic_click, start_time_s=31.2, gain=0.85, sr=sr)

    # t = 38.0s: Celebratory triumphant 'wheee' quack & melody
    v_wheee = recipes.wheee(personality, variant=0)
    overlay_audio(master, v_wheee, start_time_s=38.0, gain=0.90, sr=sr)

    # 3. Add ambient servo motor hum and subtle ToF sonar pings
    t_full = np.arange(n_samples, dtype=np.float32) / sr
    servo_hum = 0.015 * np.sin(2.0 * math.pi * 120.0 * t_full) + 0.008 * np.sin(2.0 * math.pi * 240.0 * t_full)

    # Sonar pings every 2.0 seconds during scanning/walking
    sonar_ping = np.sin(2.0 * math.pi * 2200.0 * s.t_axis(0.04)) * np.exp(-s.t_axis(0.04) / 0.008)
    for ping_t in range(1, 46, 3):
        overlay_audio(master, sonar_ping, start_time_s=float(ping_t), gain=0.03, sr=sr)

    master += servo_hum.astype(np.float32)

    # Final master normalisation to -1.0 dBFS
    master = s.normalise(master, -1.0)
    wavfile.write(str(out), sr, s.to_i16(master))

    if temp_cmd_wav.exists():
        try:
            temp_cmd_wav.unlink()
        except OSError:
            pass

    return out
