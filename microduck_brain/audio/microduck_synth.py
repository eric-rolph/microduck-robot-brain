"""DSP Primitives for Microduck Voice Synthesis at 48 kHz.

Faithfully ported from Pollen Robotics' microduck-main/sounds/src/synth.rs.
Generates organic, harmonic duck vocalizations using numpy vectorization.
"""

from __future__ import annotations

import math
import numpy as np

SR: int = 48000
TUNED_SR: float = 22050.0


def t_axis(duration_s: float) -> np.ndarray:
    """Generate time axis vector for a given duration in seconds."""
    n = max(1, int(round(duration_s * SR)))
    return (np.arange(n, dtype=np.float32) / SR).astype(np.float32)


def lerp(t: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    """Piecewise-linear curve through (time_s, value) points, clamping to endpoints."""
    times = [p[0] for p in points]
    values = [p[1] for p in points]
    return np.interp(t, times, values).astype(np.float32)


def expdecay(t: np.ndarray, attack_s: float, decay_s: float) -> np.ndarray:
    """Fast attack, exponential decay envelope with peak ~1.0."""
    attack = max(1e-4, attack_s)
    decay = max(1e-4, decay_s)
    a = np.clip(t / attack, 0.0, 1.0)
    d = np.exp(-np.maximum(0.0, t - attack) / decay)
    return (a * d).astype(np.float32)


def bell(t: np.ndarray, attack_s: float, release_s: float) -> np.ndarray:
    """Soft attack, plateau, soft release envelope."""
    total = float(t[-1]) if len(t) > 0 else 0.0
    rel_start = max(0.0, total - release_s)
    attack = max(1e-4, attack_s)
    release = max(1e-4, release_s)

    out = np.ones_like(t)
    mask_att = t < attack
    out[mask_att] = t[mask_att] / attack

    mask_rel = t > rel_start
    out[mask_rel] = np.maximum(0.0, 1.0 - (t[mask_rel] - rel_start) / release)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def phase_from_freq(freq: np.ndarray) -> np.ndarray:
    """Integrate instantaneous frequency curve (Hz) to phase (radians)."""
    acc = np.cumsum(freq.astype(np.float64))
    phase = (2.0 * math.pi * acc / SR).astype(np.float32)
    return phase


def harmonic_osc(phase: np.ndarray, weights: list[float] | np.ndarray) -> np.ndarray:
    """Sum of sin(n * phase) * weight for each harmonic (n = 1..N)."""
    out = np.zeros_like(phase, dtype=np.float32)
    for i, w in enumerate(weights):
        if w == 0.0:
            continue
        n = i + 1
        out += float(w) * np.sin(n * phase).astype(np.float32)
    return out


def vibrato(t: np.ndarray, rate_hz: float, depth_semitones: float, phase: float = 0.0) -> np.ndarray:
    """Pitch multiplier from a slow LFO for vibrato."""
    if rate_hz <= 0.0 or depth_semitones <= 0.0:
        return np.ones_like(t, dtype=np.float32)
    lfo = np.sin(2.0 * math.pi * rate_hz * t + phase)
    return (2.0 ** (depth_semitones * lfo / 12.0)).astype(np.float32)


def jitter(t: np.ndarray, depth_semitones: float, rng: np.random.Generator) -> np.ndarray:
    """Random pitch wobble using smoothed white noise for organic character."""
    if depth_semitones <= 0.0 or len(t) == 0:
        return np.ones_like(t, dtype=np.float32)
    raw = rng.standard_normal(len(t)).astype(np.float32)
    window = max(1, int(round(64.0 * SR / TUNED_SR)))
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(raw, kernel, mode="same")
    return (2.0 ** (depth_semitones * smoothed / 12.0)).astype(np.float32)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink noise generated via a 1-pole leaky integrator."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    a = 0.985 ** (TUNED_SR / SR)
    white = rng.standard_normal(n).astype(np.float32)
    pink = np.empty(n, dtype=np.float32)
    leak = 0.0
    for i in range(n):
        leak = a * leak + white[i]
        pink[i] = leak
    peak = float(np.max(np.abs(pink))) + 1e-9
    return (pink / peak).astype(np.float32)


def click(n: int, rng: np.random.Generator, length: int) -> np.ndarray:
    """Short percussive transient click for peck attack."""
    out = np.zeros(n, dtype=np.float32)
    l = min(length, n)
    if l <= 0:
        return out
    fade = (1.0 - np.arange(l, dtype=np.float32) / l) ** 2
    out[:l] = rng.uniform(-1.0, 1.0, size=l).astype(np.float32) * fade
    return out


def normalise(x: np.ndarray, peak_dbfs: float = -3.0) -> np.ndarray:
    """Normalise signal to a target peak level in dBFS."""
    peak = float(np.max(np.abs(x))) if len(x) > 0 else 0.0
    if peak < 1e-9:
        return x
    target = 10.0 ** (peak_dbfs / 20.0)
    return (x * (target / peak)).astype(np.float32)


def to_i16(x: np.ndarray) -> np.ndarray:
    """Convert floating-point audio signal [-1.0, +1.0] to signed 16-bit PCM."""
    return np.clip(x * 32767.0, -32768.0, 32767.0).astype(np.int16)
