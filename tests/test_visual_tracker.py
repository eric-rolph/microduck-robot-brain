"""Tests for HeadCamera, SORTTracker, and Visual Servoing Controller."""

import os
import mujoco
import numpy as np
import pytest
import supervision as sv

from microduck_brain.vision.head_camera import HeadCamera
from microduck_brain.vision.tracker import MicroduckVisualTracker
from microduck_brain.vision.visual_servoing import VisualServoingController


@pytest.fixture
def scene_model_data():
    scene_path = os.path.abspath("microduck_brain/sim/mjcf/scene_demo_1080p.xml")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_head_camera_rendering_and_detection(scene_model_data):
    """Verify HeadCamera can render RGB frames and detect scene geoms."""
    model, data = scene_model_data
    cam = HeadCamera(model, camera_name="head_camera", width=320, height=320)

    rgb = cam.render_rgb(data)
    assert rgb.shape == (320, 320, 3)
    assert rgb.dtype == np.uint8

    detections = cam.detect_objects(data, min_pixels=5)
    # Check detections container shape
    assert isinstance(detections, sv.Detections)
    assert len(detections) >= 0


def test_visual_tracker_lifecycle():
    """Verify MicroduckVisualTracker maintains IDs and computes bearing/range."""
    tracker = MicroduckVisualTracker(frame_rate=50.0, image_width=480, image_height=480)

    # Frame 1: Detection of marker near center
    dets1 = sv.Detections(
        xyxy=np.array([[220.0, 200.0, 260.0, 280.0]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
    )
    rgb_dummy = np.zeros((480, 480, 3), dtype=np.uint8)

    state1 = tracker.update(dets1, rgb_frame=rgb_dummy)
    assert state1.annotated_frame is not None

    # Frame 2: Slight movement (duck walking head bob)
    dets2 = sv.Detections(
        xyxy=np.array([[222.0, 204.0, 262.0, 284.0]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
    )
    state2 = tracker.update(dets2, rgb_frame=rgb_dummy)

    # Tracker should lock onto target track and report small bearing error
    assert state2.target_locked is True
    assert abs(state2.bearing_rad) < 0.2
    assert state2.range_m > 0.0


def test_visual_servoing_controller():
    """Verify visual servoing produces smooth steering toward target."""
    controller = VisualServoingController(kp_yaw=0.8, kp_dist=0.3, target_range_m=0.20)
    tracker = MicroduckVisualTracker(frame_rate=50.0)

    dets = sv.Detections(
        xyxy=np.array([[300.0, 180.0, 340.0, 260.0]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
    )
    tracker.update(dets)
    state = tracker.update(dets)

    vx, wz, reached = controller.compute_commands(state)
    assert abs(vx) <= 0.08
    assert abs(wz) <= 0.18
    # Target was to the right (x=300 > 240), wz should be negative to turn right
    assert wz < 0.0
