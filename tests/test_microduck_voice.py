"""Tests for Microduck procedural voice response system and audio synthesis."""

import numpy as np
import pytest
from pathlib import Path

from microduck_brain.audio.personality import Personality
from microduck_brain.audio import microduck_synth as s
from microduck_brain.audio import recipes
from microduck_brain.audio.voice_interface import synthesize_human_command, generate_mission_audio_track


def test_synth_primitives():
    """Verify DSP synthesis primitives produce bounded, continuous signals."""
    t = s.t_axis(0.1)
    assert len(t) == int(0.1 * s.SR)
    assert np.isclose(t[0], 0.0)

    # Linear interpolation
    pts = [(0.0, 100.0), (0.05, 200.0), (0.1, 150.0)]
    f = s.lerp(t, pts)
    assert len(f) == len(t)
    assert np.isclose(f[0], 100.0)
    assert np.isclose(f[-1], 150.0, atol=0.1)

    # Envelopes
    env_exp = s.expdecay(t, 0.01, 0.05)
    assert np.all(env_exp >= 0.0) and np.all(env_exp <= 1.05)

    env_bell = s.bell(t, 0.02, 0.02)
    assert np.all(env_bell >= 0.0) and np.all(env_bell <= 1.0)

    # Oscillators
    phase = s.phase_from_freq(f)
    body = s.harmonic_osc(phase, [1.0, 0.5, 0.25])
    assert len(body) == len(t)
    assert np.all(np.isfinite(body))

    # Noise & Normalisation
    rng = np.random.default_rng(42)
    noise = s.pink_noise(len(t), rng)
    assert len(noise) == len(t)
    normed = s.normalise(body, -3.0)
    peak = float(np.max(np.abs(normed)))
    expected_peak = 10.0 ** (-3.0 / 20.0)
    assert np.isclose(peak, expected_peak, atol=1e-3)


def test_personality_stability():
    """Verify personality derived from seed is deterministic and bounded."""
    p1 = Personality.from_seed(42)
    p2 = Personality.from_seed(42)
    assert p1.pitch_center_hz == p2.pitch_center_hz
    assert p1.speed == p2.speed
    assert p1.quackiness == p2.quackiness

    # Different seeds yield distinct ducks
    p3 = Personality.from_seed(99)
    assert p1.pitch_center_hz != p3.pitch_center_hz

    # Harmonics contain 7 entries with f0 dominant
    harmonics = p1.harmonics()
    assert len(harmonics) == 7
    assert harmonics[0] >= 0.7


def test_all_vocalization_recipes():
    """Verify all 7 vocalization recipes generate finite, normalized waveforms."""
    p = Personality.from_seed(100)
    for variant in [0, 1]:
        g = recipes.greet(p, variant)
        assert len(g) > 2000 and np.all(np.isfinite(g))

        iq = recipes.inquire(p, variant)
        assert len(iq) > 2000 and np.all(np.isfinite(iq))

        pk = recipes.peck(p, variant)
        assert len(pk) > 1000 and np.all(np.isfinite(pk))

        ch = recipes.chirp(p, variant)
        assert len(ch) > 1000 and np.all(np.isfinite(ch))

        co = recipes.coo(p, variant)
        assert len(co) > 4000 and np.all(np.isfinite(co))

        wh = recipes.wheee(p, variant)
        assert len(wh) > 10000 and np.all(np.isfinite(wh))

        al = recipes.alarm(p, variant)
        assert len(al) > 2000 and np.all(np.isfinite(al))


def test_human_command_synthesis(tmp_path: Path):
    """Test generating a spoken speech command WAV via Windows SAPI."""
    wav_out = tmp_path / "test_cmd.wav"
    res = synthesize_human_command("Microduck, stand up!", wav_out)
    assert res.exists()
    assert res.stat().st_size > 1000


def test_full_mission_audio_generation(tmp_path: Path):
    """Test full multi-track soundtrack generator."""
    wav_out = tmp_path / "full_demo_track.wav"
    res = generate_mission_audio_track(wav_out, total_duration_s=5.0, duck_seed=42)
    assert res.exists()
    assert res.stat().st_size > 50000
