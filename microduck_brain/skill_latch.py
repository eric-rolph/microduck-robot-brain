"""
Action latching and event-driven replanning for Tier 2/3.
Prevents policy flip-flopping at state boundaries by executing skills
as asynchronous contracts with explicit terminal exit codes.
"""

from __future__ import annotations
from enum import Enum
from typing import Callable, Mapping, Optional


class SkillStatus(Enum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class SkillLatch:
    """
    Executes and latches active skills until completion or failure.
    The planner is invoked only on state boundary events or terminal signals.
    """

    def __init__(self, skills: Mapping[str, Callable[[], SkillStatus]]):
        self.skills = dict(skills)
        self.active_skill_name: Optional[str] = None

    def execute_tick(self, requested_intent: str) -> SkillStatus:
        """
        Executes one control cycle of the active skill.
        If no skill is currently active, latches the requested intent.
        """
        # If no skill is running, latch the requested skill
        if self.active_skill_name is None:
            if requested_intent not in self.skills:
                return SkillStatus.FAILURE
            self.active_skill_name = requested_intent

        # Run one control cycle of the currently latched skill
        status = self.skills[self.active_skill_name]()

        # Release the latch when the skill reaches a terminal state
        if status in (SkillStatus.SUCCESS, SkillStatus.FAILURE):
            self.active_skill_name = None

        return status

    def abort_active(self) -> None:
        """Forces release of the currently latched skill on interrupt."""
        self.active_skill_name = None
