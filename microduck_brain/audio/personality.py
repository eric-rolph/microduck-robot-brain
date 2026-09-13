"""Microduck Vocal Personality Generator.

Faithfully ported from Pollen Robotics' microduck-main/sounds/src/personality.rs.
Derives stable, identifiable per-robot vocal traits from a single integer seed.
"""

from __future__ import annotations

from dataclasses import dataclass
import zlib
import numpy as np


@dataclass
class Personality:
    seed: int

    # Pitch
    pitch_center_hz: float
    register: float
    pitch_spread: float
    glide_bias: float

    # Timbre
    brightness: float
    tilt: float
    nasal: float
    harmonic_skew: float
    formant_n: int
    formant_gain: float

    # Modulation
    vibrato_rate_hz: float
    vibrato_depth: float
    jitter_depth: float
    breath: float
    quackiness: float
    am_rate_hz: float
    am_depth: float
    warble_hz: float
    warble_depth: float

    # Timing
    attack_sharpness: float
    speed: float

    @classmethod
    def from_seed(cls, seed: int = 42) -> Personality:
        """Derive consistent personality traits from seed."""
        rng = np.random.default_rng(seed)

        reg_choices = [-1.0, 0.0, 0.0, 1.0]
        register = float(rng.choice(reg_choices) + rng.uniform(-0.4, 0.4))
        base = float(rng.uniform(160.0, 380.0))
        pitch = float(np.clip(base * (2.0 ** (register * 0.45)), 110.0, 620.0))

        return cls(
            seed=seed,
            pitch_center_hz=pitch,
            register=register,
            pitch_spread=float(rng.uniform(0.4, 1.2)),
            glide_bias=float(rng.uniform(-1.0, 1.0)),

            brightness=float(rng.uniform(0.05, 0.55)),
            tilt=float(rng.uniform(1.4, 2.8)),
            nasal=float(rng.uniform(0.1, 1.0)),
            harmonic_skew=float(rng.uniform(-1.0, 1.0)),
            formant_n=int(rng.integers(1, 6)),
            formant_gain=float(rng.uniform(0.0, 1.4)),

            vibrato_rate_hz=float(rng.uniform(3.5, 9.5)),
            vibrato_depth=float(rng.uniform(0.0, 0.7)),
            jitter_depth=float(rng.uniform(0.03, 0.35)),
            breath=float(rng.uniform(0.0, 0.30)),
            quackiness=float(rng.uniform(0.2, 1.0)),
            am_rate_hz=float(rng.uniform(18.0, 55.0)),
            am_depth=float(rng.uniform(0.15, 0.70)),
            warble_hz=float(rng.uniform(7.0, 18.0)),
            warble_depth=float(rng.uniform(0.0, 1.4)),

            attack_sharpness=float(rng.uniform(0.0, 1.0)),
            speed=float(rng.uniform(0.82, 1.22)),
        )

    def variant_rng(self, tag: str, variant: int) -> np.random.Generator:
        """Stable sub-RNG for specific vocal recipe variants using CRC32."""
        tag_crc = zlib.crc32(tag.encode("utf-8"))
        h = (
            ((self.seed * 1000003) & 0xFFFFFFFF)
            ^ tag_crc
            ^ ((variant * 2654435761) & 0xFFFFFFFF)
        ) & 0xFFFFFFFF
        return np.random.default_rng(h)

    def harmonics(self) -> list[float]:
        """Generate harmonic weights for the additive oscillator."""
        n_harm = 7
        weights = []
        for n in range(1, n_harm + 1):
            base = 1.0 / (float(n) ** self.tilt)
            high_lift = self.brightness * ((float(n) / float(n_harm)) ** 1.5)
            nasal = self.nasal * (0.6 if n in (2, 3) else 0.0)
            if self.harmonic_skew >= 0.0:
                skew = self.harmonic_skew * (0.4 if n % 2 == 0 else -0.2)
            else:
                skew = -self.harmonic_skew * (-0.3 if n % 2 == 0 else 0.4)
            formant = self.formant_gain if n == self.formant_n else 0.0
            w = max(0.0, base + high_lift + nasal + skew + formant * base * 1.5)
            weights.append(w)
        weights[0] = max(0.7, weights[0])
        return weights
