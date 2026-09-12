"""
Unit tests for BehaviorTree sequences, selectors, parallel nodes, and expressions.
"""

from typing import Mapping, Any
from microduck_brain.behavior_tree import (
    BehaviorNode,
    SequenceNode,
    SelectorNode,
    AsymmetricParallelNode,
    SafeExpressionNode,
    AmbientBodyLanguageNode,
    FailureExpressionBranch,
    Blackboard,
)
from microduck_brain.skill_latch import SkillStatus


class MockAction(BehaviorNode):
    def __init__(self, name: str, status_sequence: list[SkillStatus], blackboard: Blackboard = None):
        super().__init__(name, blackboard)
        self.status_sequence = status_sequence
        self.idx = 0
        self.aborted = False

    def tick(self, world_state: Mapping[str, Any]) -> SkillStatus:
        if self.idx < len(self.status_sequence):
            res = self.status_sequence[self.idx]
            self.idx += 1
            return res
        return SkillStatus.SUCCESS

    def abort(self) -> None:
        self.aborted = True


def test_sequence_node():
    bb = Blackboard()
    child1 = MockAction("A", [SkillStatus.SUCCESS], bb)
    child2 = MockAction("B", [SkillStatus.RUNNING, SkillStatus.SUCCESS], bb)
    seq = SequenceNode("Seq", bb, [child1, child2])

    state = {"stability": "HIGH"}
    assert seq.tick(state) == SkillStatus.RUNNING
    assert seq.tick(state) == SkillStatus.SUCCESS


def test_selector_fallback():
    bb = Blackboard()
    child1 = MockAction("Primary", [SkillStatus.FAILURE], bb)
    child2 = MockAction("Fallback", [SkillStatus.SUCCESS], bb)
    sel = SelectorNode("Sel", bb, [child1, child2])

    state = {"stability": "HIGH"}
    assert sel.tick(state) == SkillStatus.SUCCESS


def test_asymmetric_parallel_node():
    bb = Blackboard()
    primary = MockAction("Nav", [SkillStatus.RUNNING, SkillStatus.SUCCESS], bb)
    ambient = MockAction("Scan", [SkillStatus.RUNNING, SkillStatus.RUNNING], bb)

    parallel = AsymmetricParallelNode("Par", bb, [primary, ambient], primary_idx=0)
    state = {"stability": "HIGH"}

    assert parallel.tick(state) == SkillStatus.RUNNING
    assert not ambient.aborted

    # When primary succeeds, ambient is aborted
    assert parallel.tick(state) == SkillStatus.SUCCESS
    assert ambient.aborted


def test_safe_expression_suppression():
    bb = Blackboard()
    executed = []
    node = SafeExpressionNode(
        "Tilt",
        bb,
        gesture_id="HEAD_TILT",
        duration_ticks=2,
        gesture_callback=lambda g: executed.append(g),
    )

    # When unstable, gesture is suppressed and returns SUCCESS
    unstable_state = {"stability": "LOW", "roughness": "HIGH"}
    assert node.tick(unstable_state) == SkillStatus.SUCCESS
    assert len(executed) == 0

    # When stable, executes gesture over duration
    stable_state = {"stability": "HIGH", "roughness": "LOW"}
    assert node.tick(stable_state) == SkillStatus.RUNNING
    assert len(executed) == 1
    assert executed[0] == "HEAD_TILT"
    assert node.tick(stable_state) == SkillStatus.SUCCESS


def test_failure_expression_branching():
    bb = Blackboard()
    gestures = []
    branch = FailureExpressionBranch(
        "FailBranch",
        bb,
        monitored_task="fetch_ball",
        gesture_callback=lambda g: gestures.append(g),
    )

    state = {"stability": "HIGH", "roughness": "LOW"}

    # Failure 1: Head Tilt
    branch.tick(state)
    assert bb.get_failure_count("fetch_ball") == 1
    assert gestures[-1] == "HEAD_TILT"

    # Failure 2: Head Tilt
    branch.tick(state)
    assert bb.get_failure_count("fetch_ball") == 2

    # Failure 3: Head Shake
    branch.tick(state)
    assert bb.get_failure_count("fetch_ball") == 3
    assert gestures[-1] == "HEAD_SHAKE"
