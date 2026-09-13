"""Microduck Vocalization Recipes.

Faithfully ported from Pollen Robotics' microduck-main/sounds/src/voices.rs.
Generates authentic duck vocalizations matching hardware firmware.
"""

from __future__ import annotations

import math
import numpy as np
from microduck_brain.audio.personality import Personality
from microduck_brain.audio import microduck_synth as s


def attack(p: Personality, dur: float, snappy: float) -> float:
    """Modulated attack time in seconds."""
    soft = 0.04 * dur
    sharp = 0.003 * dur
    return max(0.001, soft + (sharp - soft) * p.attack_sharpness * snappy)


def voice(
    p: Personality,
    t: np.ndarray,
    freq: np.ndarray,
    rng: np.random.Generator,
    am_scale: float = 1.0,
    breath_scale: float = 1.0,
) -> np.ndarray:
    """Core vocal synthesis: harmonic osc + vibrato + jitter + AM buzz + breath."""
    vib = s.vibrato(
        t,
        p.vibrato_rate_hz,
        p.vibrato_depth,
        float(rng.uniform(0.0, 2.0 * math.pi)),
    )
    jit = s.jitter(t, p.jitter_depth, rng)
    f = freq * vib * jit
    phase = s.phase_from_freq(f)
    body = s.harmonic_osc(phase, p.harmonics())

    am_d = p.am_depth * am_scale * p.quackiness
    if am_d > 0.01:
        am = 1.0 - am_d * (0.5 + 0.5 * np.sin(2.0 * math.pi * p.am_rate_hz * t))
        body *= am.astype(np.float32)

    breath = p.breath * breath_scale
    if breath > 0.0:
        body += breath * s.pink_noise(len(t), rng)

    return body


def greet_syllable(
    p: Personality,
    rng: np.random.Generator,
    dur_scale: float = 1.0,
    f0_scale: float = 1.0,
) -> np.ndarray:
    """Single quack syllable."""
    dur = float((0.32 + 0.25 * rng.random()) * dur_scale / p.speed)
    t = s.t_axis(dur)
    f0 = p.pitch_center_hz * float(0.9 + 0.15 * rng.random()) * f0_scale
    bias = p.glide_bias
    bend = 0.10 + 0.15 * p.pitch_spread
    start = f0 * (1.0 - bias * bend * 0.5)
    mid = f0 * (1.0 + bias * bend)
    end = f0 * (1.0 - bias * bend * 0.3) * float(0.92 + 0.08 * rng.random())
    freq = s.lerp(t, [(0.0, start), (0.18 * dur, mid), (dur, end)])
    env = s.expdecay(t, attack(p, dur, 0.5), dur * 0.7)
    return voice(p, t, freq, rng, 1.0, 1.0) * env


def greet(p: Personality, variant: int = 0) -> np.ndarray:
    """Friendly greeting quack / double quack ('wak' / 'wak-wak')."""
    rng = p.variant_rng("greet", variant)
    sig = greet_syllable(p, rng, 1.0, 1.0)
    if rng.random() < 0.4:
        gap_n = max(1, int(round((0.05 + 0.06 * rng.random()) / p.speed * s.SR)))
        gap = np.zeros(gap_n, dtype=np.float32)
        f0_scale = float(0.95 + 0.06 * rng.random())
        sig2 = greet_syllable(p, rng, 0.8, f0_scale)
        sig = np.concatenate([sig, gap, sig2])
    return s.normalise(sig, -3.0)


def inquire(p: Personality, variant: int = 0) -> np.ndarray:
    """Upward inquisitive chirp when detecting/locking onto an object."""
    rng = p.variant_rng("inquire", variant)
    dur = float((0.42 + 0.25 * rng.random()) / p.speed)
    t = s.t_axis(dur)
    f0 = p.pitch_center_hz * float(0.88 + 0.10 * rng.random())
    rise = 1.15 + 0.50 * p.pitch_spread + 0.20 * max(0.0, p.glide_bias) + 0.10 * float(rng.random())
    freq = s.lerp(t, [(0.0, f0 * 0.92), (0.30 * dur, f0 * 0.95), (dur, f0 * rise)])
    env = s.bell(t, dur * (0.06 + 0.10 * (1.0 - p.attack_sharpness)), dur * 0.32)
    sig = voice(p, t, freq, rng, 0.6, 1.0) * env
    return s.normalise(sig, -3.0)


def peck(p: Personality, variant: int = 0) -> np.ndarray:
    """Snappy percussive peck sound with click transient."""
    rng = p.variant_rng("peck", variant)
    dur = float((0.16 + 0.12 * rng.random()) / p.speed)
    t = s.t_axis(dur)
    f0 = p.pitch_center_hz * float(0.45 + 0.20 * rng.random())
    freq = s.lerp(t, [(0.0, f0 * 1.5), (0.04 * dur, f0), (dur, f0 * 0.80)])
    env = s.expdecay(t, attack(p, dur, 1.0), dur * 0.35)
    body = voice(p, t, freq, rng, 0.3, 0.5) * env

    click_len = max(1, int(round((0.003 + 0.006 * p.attack_sharpness) * s.SR)))
    click_gain = float(0.4 + 0.4 * p.attack_sharpness)
    c = s.click(len(t), rng, click_len)
    body += click_gain * c
    return s.normalise(body, -3.0)


def chirp_syllable(
    p: Personality,
    rng: np.random.Generator,
    f0: float,
    dur: float,
    contour: list[tuple[float, float]],
) -> np.ndarray:
    """Soft trilled chirp syllable."""
    t = s.t_axis(dur)
    warble = s.vibrato(
        t,
        p.warble_hz,
        p.warble_depth,
        float(rng.uniform(0.0, 2.0 * math.pi)),
    )
    shape = s.lerp(t, contour)
    freq = (f0 * shape * warble).astype(np.float32)
    env = s.expdecay(t, attack(p, dur, 0.7), dur * 0.55)

    p_soft = Personality(
        seed=p.seed,
        pitch_center_hz=p.pitch_center_hz,
        register=p.register,
        pitch_spread=p.pitch_spread,
        glide_bias=p.glide_bias,
        brightness=p.brightness * 0.5,
        tilt=p.tilt,
        nasal=p.nasal,
        harmonic_skew=p.harmonic_skew,
        formant_n=p.formant_n,
        formant_gain=p.formant_gain * 0.5,
        vibrato_rate_hz=p.vibrato_rate_hz,
        vibrato_depth=p.vibrato_depth,
        jitter_depth=p.jitter_depth,
        breath=p.breath,
        quackiness=p.quackiness * 0.2,
        am_rate_hz=p.am_rate_hz,
        am_depth=p.am_depth,
        warble_hz=p.warble_hz,
        warble_depth=p.warble_depth,
        attack_sharpness=p.attack_sharpness,
        speed=p.speed,
    )
    return voice(p_soft, t, freq, rng, 1.0, 0.4) * env


def chirp(p: Personality, variant: int = 0) -> np.ndarray:
    """Mouth/beak trigger chirp (cycling rise, fall, trill, double shapes)."""
    rng = p.variant_rng("chirp", variant)
    shape = variant % 4
    f0 = p.pitch_center_hz * float(0.95 + 0.75 * rng.random())

    if shape == 0:
        dur = float((0.10 + 0.10 * rng.random()) / p.speed)
        peak = float(1.12 + 0.10 * rng.random())
        sig = chirp_syllable(p, rng, f0, dur, [(0.0, 0.88), (0.5 * dur, peak), (dur, 1.05)])
    elif shape == 1:
        dur = float((0.10 + 0.10 * rng.random()) / p.speed)
        start = float(1.12 + 0.10 * rng.random())
        sig = chirp_syllable(p, rng, f0, dur, [(0.0, start), (0.3 * dur, 1.0), (dur, 0.78)])
    elif shape == 2:
        dur = float((0.22 + 0.14 * rng.random()) / p.speed)
        p_trill = Personality(
            seed=p.seed,
            pitch_center_hz=p.pitch_center_hz,
            register=p.register,
            pitch_spread=p.pitch_spread,
            glide_bias=p.glide_bias,
            brightness=p.brightness,
            tilt=p.tilt,
            nasal=p.nasal,
            harmonic_skew=p.harmonic_skew,
            formant_n=p.formant_n,
            formant_gain=p.formant_gain,
            vibrato_rate_hz=p.vibrato_rate_hz,
            vibrato_depth=p.vibrato_depth,
            jitter_depth=p.jitter_depth,
            breath=p.breath,
            quackiness=p.quackiness,
            am_rate_hz=p.am_rate_hz,
            am_depth=p.am_depth,
            warble_hz=p.warble_hz,
            warble_depth=max(0.7, p.warble_depth) * 1.6,
            attack_sharpness=p.attack_sharpness,
            speed=p.speed,
        )
        sig = chirp_syllable(p_trill, rng, f0, dur, [(0.0, 1.0), (0.5 * dur, 1.06), (dur, 0.94)])
    else:
        dur = float((0.08 + 0.05 * rng.random()) / p.speed)
        s1 = chirp_syllable(p, rng, f0, dur, [(0.0, 0.95), (0.4 * dur, 1.10), (dur, 0.88)])
        gap_n = max(1, int(round((0.03 + 0.03 * rng.random()) / p.speed * s.SR)))
        gap = np.zeros(gap_n, dtype=np.float32)
        dur_b = dur * 0.9
        f0_b = f0 * float(0.92 + 0.10 * rng.random())
        s2 = chirp_syllable(p, rng, f0_b, dur_b, [(0.0, 0.95), (0.4 * dur_b, 1.10), (dur_b, 0.88)])
        sig = np.concatenate([s1, gap, s2])

    return s.normalise(sig, -6.0)


def coo(p: Personality, variant: int = 0) -> np.ndarray:
    """Gentle low-frequency coo vocalization during careful navigation."""
    rng = p.variant_rng("coo", variant)
    dur = float((0.85 + 0.55 * rng.random()) / p.speed)
    t = s.t_axis(dur)
    f0 = p.pitch_center_hz * float(0.42 + 0.15 * (1.0 - p.attack_sharpness)) * float(0.94 + 0.12 * rng.random())
    drift_a = float(1.0 + 0.05 * rng.random() + 0.04 * p.glide_bias)
    freq = s.lerp(t, [(0.0, f0 * 0.94), (dur * 0.5, f0 * drift_a), (dur, f0 * 0.90)])
    env = s.bell(t, dur * (0.18 + 0.10 * (1.0 - p.attack_sharpness)), dur * 0.30)

    p_soft = Personality(
        seed=p.seed,
        pitch_center_hz=p.pitch_center_hz,
        register=p.register,
        pitch_spread=p.pitch_spread,
        glide_bias=p.glide_bias,
        brightness=p.brightness,
        tilt=p.tilt,
        nasal=p.nasal,
        harmonic_skew=p.harmonic_skew,
        formant_n=p.formant_n,
        formant_gain=p.formant_gain,
        vibrato_rate_hz=p.vibrato_rate_hz * 0.45,
        vibrato_depth=p.vibrato_depth * 0.7,
        jitter_depth=p.jitter_depth,
        breath=max(0.12, p.breath) + 0.10,
        quackiness=p.quackiness * 0.25,
        am_rate_hz=p.am_rate_hz * 0.30,
        am_depth=p.am_depth,
        warble_hz=p.warble_hz,
        warble_depth=p.warble_depth,
        attack_sharpness=p.attack_sharpness,
        speed=p.speed,
    )
    sig = voice(p_soft, t, freq, rng, 1.0, 1.0) * env
    return s.normalise(sig, -5.0)


def wheee(p: Personality, variant: int = 0) -> np.ndarray:
    """Triumphant celebratory melodic quack upon mission completion."""
    rng = p.variant_rng("wheee", variant)
    d_start = float((0.80 + 0.30 * rng.random()) / p.speed)
    d_loop = float((1.60 + 0.60 * rng.random()) / p.speed)
    d_end = float((0.55 + 0.25 * rng.random()) / p.speed)
    total = d_start + d_loop + d_end
    t = s.t_axis(total)
    t1, t2 = d_start, d_start + d_loop

    f0 = p.pitch_center_hz * float(0.95 + 0.10 * rng.random())
    top = float(1.6 + 0.5 * p.pitch_spread + 0.25 * rng.random())
    base = s.lerp(
        t,
        [
            (0.0, f0 * 0.85),
            (0.15 * d_start, f0),
            (t1, f0 * top),
            (t2, f0 * top),
            (t2 + 0.25 * d_end, f0 * top * 1.04),
            (total, f0 * 0.60),
        ],
    )
    wob_hz = float(4.5 + 3.0 * rng.random())
    swell = s.lerp(t, [(0.0, 0.15), (t1, 1.0), (total, 1.0)])
    wob = np.sin(2.0 * math.pi * wob_hz * t)
    freq = (base * (2.0 ** (0.5 * swell * wob / 12.0))).astype(np.float32)

    env = s.lerp(
        t,
        [
            (0.0, 0.0),
            (min(0.06, 0.5 * d_start), 1.0),
            (t2 + 0.40 * d_end, 1.0),
            (total, 0.0),
        ],
    )
    p_joy = Personality(
        seed=p.seed,
        pitch_center_hz=p.pitch_center_hz,
        register=p.register,
        pitch_spread=p.pitch_spread,
        glide_bias=p.glide_bias,
        brightness=p.brightness,
        tilt=p.tilt,
        nasal=p.nasal,
        harmonic_skew=p.harmonic_skew,
        formant_n=p.formant_n,
        formant_gain=p.formant_gain,
        vibrato_rate_hz=p.vibrato_rate_hz,
        vibrato_depth=p.vibrato_depth * 0.5,
        jitter_depth=p.jitter_depth,
        breath=p.breath,
        quackiness=p.quackiness * 0.5,
        am_rate_hz=p.am_rate_hz,
        am_depth=p.am_depth,
        warble_hz=p.warble_hz,
        warble_depth=p.warble_depth,
        attack_sharpness=p.attack_sharpness,
        speed=p.speed,
    )
    sig = voice(p_joy, t, freq, rng, 0.5, 0.5) * env
    return s.normalise(sig, -4.0)


def alarm(p: Personality, variant: int = 0) -> np.ndarray:
    """Warning alarm honk with crackle noise."""
    rng = p.variant_rng("alarm", variant)
    dur = float((0.20 + 0.12 * rng.random()) / p.speed)
    t = s.t_axis(dur)
    f0 = p.pitch_center_hz * (1.25 + 0.35 * p.pitch_spread) * float(0.94 + 0.12 * rng.random())
    peak_mul = float(1.15 + 0.25 * p.pitch_spread + 0.10 * rng.random())
    fall_mul = float(0.75 + 0.20 * (1.0 - p.pitch_spread))
    freq = s.lerp(t, [(0.0, f0), (0.05 * dur, f0 * peak_mul), (dur, f0 * fall_mul)])
    env = s.expdecay(t, attack(p, dur, 1.0), dur * float(0.40 + 0.20 * rng.random()))
    sig = voice(p, t, freq, rng, 0.5, 1.0) * env
    crackle = float(0.04 + 0.10 * p.brightness)
    sig += crackle * rng.standard_normal(len(t)).astype(np.float32) * env
    return s.normalise(sig, -3.0)
