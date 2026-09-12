"""
Microduck Robot Brain
Decoupled four-tier brain architecture for the Microduck quadruped robot.
"""

from .intent_parser import ParsedIntent, parse_voice_command_stateless
from .world_state import SchmittTrigger, MultiLevelSchmittTrigger, TemporalDebouncer, WorldState
from .logit_masking import mask_invalid_actions
from .skill_latch import SkillStatus, SkillLatch
from .behavior_tree import (
    BehaviorNode,
    SequenceNode,
    SelectorNode,
    AsymmetricParallelNode,
    SafeExpressionNode,
    AmbientBodyLanguageNode,
    FailureExpressionBranch,
    Blackboard,
)
from .locomotion_engine import AttitudeCommandFilter, MicroduckLocomotionEngine
from .sim.env import MicroduckMuJoCoEnv
from .sim.bam_actuator import BamM6ActuatorModel, BamM6Config
from .sim.backlash import BacklashManager

__all__ = [
    "ParsedIntent",
    "parse_voice_command_stateless",
    "SchmittTrigger",
    "MultiLevelSchmittTrigger",
    "TemporalDebouncer",
    "WorldState",
    "mask_invalid_actions",
    "SkillStatus",
    "SkillLatch",
    "BehaviorNode",
    "SequenceNode",
    "SelectorNode",
    "AsymmetricParallelNode",
    "SafeExpressionNode",
    "AmbientBodyLanguageNode",
    "FailureExpressionBranch",
    "Blackboard",
    "AttitudeCommandFilter",
    "MicroduckLocomotionEngine",
    "MicroduckMuJoCoEnv",
    "BamM6ActuatorModel",
    "BamM6Config",
    "BacklashManager",
]
