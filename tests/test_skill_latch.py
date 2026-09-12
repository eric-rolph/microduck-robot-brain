"""
Unit tests for SkillLatch and event-driven replanning.
"""

from microduck_brain.skill_latch import SkillLatch, SkillStatus


def test_skill_latch_lifecycle():
    call_counts = {"approach": 0}

    def mock_approach():
        call_counts["approach"] += 1
        if call_counts["approach"] < 3:
            return SkillStatus.RUNNING
        return SkillStatus.SUCCESS

    latch = SkillLatch({"APPROACH": mock_approach})

    # First tick: latches APPROACH
    status = latch.execute_tick("APPROACH")
    assert status == SkillStatus.RUNNING
    assert latch.active_skill_name == "APPROACH"

    # Second tick: runs latched skill even if intent string is different
    status = latch.execute_tick("PICKUP")
    assert status == SkillStatus.RUNNING
    assert latch.active_skill_name == "APPROACH"

    # Third tick: skill succeeds and releases latch
    status = latch.execute_tick("PICKUP")
    assert status == SkillStatus.SUCCESS
    assert latch.active_skill_name is None


def test_skill_latch_unknown_skill():
    latch = SkillLatch({})
    assert latch.execute_tick("UNKNOWN") == SkillStatus.FAILURE


def test_skill_latch_abort():
    def running_skill():
        return SkillStatus.RUNNING

    latch = SkillLatch({"RUN": running_skill})
    latch.execute_tick("RUN")
    assert latch.active_skill_name == "RUN"

    latch.abort_active()
    assert latch.active_skill_name is None
