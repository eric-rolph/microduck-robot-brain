"""Microduck Robot Brain Vision & Multi-Object Tracking Package.

Implements head-camera perception, Roboflow SORTTracker with BIoU,
persistent identity holding through bipedal walking head-bob, and visual servoing.
"""

from microduck_brain.vision.head_camera import HeadCamera
from microduck_brain.vision.tracker import MicroduckVisualTracker, VisualTrackState
from microduck_brain.vision.visual_servoing import VisualServoingController

__all__ = [
    "HeadCamera",
    "MicroduckVisualTracker",
    "VisualTrackState",
    "VisualServoingController",
]
