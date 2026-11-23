"""Microduck Robot Brain Vision & Multi-Object Tracking Package.

Implements head-camera perception, Roboflow SORTTracker with BIoU,
persistent identity holding through bipedal walking head-bob, and visual servoing.
"""

from microduck_brain.vision.visual_servoing import VisualServoingController

__all__ = ["VisualServoingController"]

try:
    from microduck_brain.vision.head_camera import HeadCamera
    __all__.append("HeadCamera")
except ImportError:
    HeadCamera = None  # type: ignore[assignment,misc]

try:
    from microduck_brain.vision.tracker import MicroduckVisualTracker, VisualTrackState
    __all__.extend(["MicroduckVisualTracker", "VisualTrackState"])
except ImportError:
    MicroduckVisualTracker = None  # type: ignore[assignment,misc]
    VisualTrackState = None  # type: ignore[assignment,misc]
