"""
Unit tests for SchmittTrigger and TemporalDebouncer.
"""

import pytest
from microduck_brain.world_state import SchmittTrigger, MultiLevelSchmittTrigger, TemporalDebouncer


def test_schmitt_trigger_hysteresis():
    # Low th: 0.35, High th: 0.65
    trigger = SchmittTrigger(low_threshold=0.35, high_threshold=0.65, initial_state="LOW")

    # Values below high threshold do not flip state
    assert trigger.update(0.50) == "LOW"
    assert trigger.update(0.64) == "LOW"

    # Crossing high threshold flips to HIGH
    assert trigger.update(0.66) == "HIGH"

    # Fluctuations above low threshold stay HIGH
    assert trigger.update(0.60) == "HIGH"
    assert trigger.update(0.40) == "HIGH"

    # Falling below low threshold flips back to LOW
    assert trigger.update(0.34) == "LOW"
    assert trigger.update(0.30) == "LOW"


def test_schmitt_trigger_invalid_thresholds():
    with pytest.raises(ValueError):
        SchmittTrigger(low_threshold=0.8, high_threshold=0.4)


def test_multi_level_schmitt_trigger():
    mls = MultiLevelSchmittTrigger(
        low_to_med=0.35,
        med_to_low=0.25,
        med_to_high=0.65,
        high_to_med=0.50,
        initial_state="LOW",
    )
    assert mls.state == "LOW"
    assert mls.update(0.40) == "MEDIUM"
    assert mls.update(0.70) == "HIGH"
    assert mls.update(0.55) == "HIGH"
    assert mls.update(0.45) == "MEDIUM"
    assert mls.update(0.20) == "LOW"


def test_temporal_debouncer():
    # Window size 5, ratio 0.8 -> requires 4 of 5 True
    debouncer = TemporalDebouncer(window_size=5, required_ratio=0.8)

    # Initial inputs
    assert not debouncer.update(False)
    assert not debouncer.update(True)

    # Fill window with True
    debouncer.update(True)
    debouncer.update(True)
    debouncer.update(True)
    # Window: [F, T, T, T, T] -> 4/5 True = 0.8 -> True
    assert debouncer.update(True)

    # A single dropout should not flip if ratio maintained
    # Window: [T, T, T, T, F] -> 4/5 = 0.8 -> True
    assert debouncer.update(False)

    # A second dropout drops ratio to 3/5 = 0.6 -> False
    assert not debouncer.update(False)


def test_temporal_debouncer_invalid_params():
    with pytest.raises(ValueError):
        TemporalDebouncer(window_size=0)

    with pytest.raises(ValueError):
        TemporalDebouncer(required_ratio=1.5)
