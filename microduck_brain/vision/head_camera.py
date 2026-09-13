"""Head Camera Rendering & Object Detection for Microduck.

Provides camera frame rendering from spec/head_camera (90 deg FOV, pitch tilted 25 deg)
and extraction of object bounding boxes for multi-object tracking.
"""

from __future__ import annotations

from typing import Any
import mujoco
import numpy as np
import supervision as sv


class HeadCamera:
    """Microduck Head-Camera vision interface."""

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_name: str = "head_camera",
        width: int = 480,
        height: int = 480,
    ) -> None:
        self.model = model
        self.camera_name = camera_name
        self.width = width
        self.height = height

        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.cam_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found in MuJoCo model.")

        self.rgb_renderer = mujoco.Renderer(model, height=height, width=width)
        self.seg_renderer = mujoco.Renderer(model, height=height, width=width)
        self.seg_renderer.enable_segmentation_rendering()

        # Class body mappings: 0=marker, 1=red_box, 2=container
        self.class_names = ["marker", "obstacle_box", "container"]
        self.target_bodies = {
            0: "marker",
            1: "red_obstacle_box",
            2: "transparent_container",
        }

        self.class_geom_ids: dict[int, set[int]] = {}
        for class_id, body_name in self.target_bodies.items():
            b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            geoms = set()
            if b_id >= 0:
                for g_id in range(model.ngeom):
                    if model.geom_bodyid[g_id] == b_id:
                        geoms.add(g_id)
            self.class_geom_ids[class_id] = geoms

        # Camera intrinsic approximation: fovy is typically 90 deg
        fovy_deg = 90.0
        fovy_rad = np.radians(fovy_deg)
        self.fy = (self.height / 2.0) / np.tan(fovy_rad / 2.0)
        self.fx = (self.width / 2.0) / np.tan(fovy_rad / 2.0)

    def render_rgb(self, data: mujoco.MjData) -> np.ndarray:
        """Render RGB frame from head camera (H, W, 3) uint8."""
        self.rgb_renderer.update_scene(data, camera=self.camera_name)
        return self.rgb_renderer.render()

    def detect_objects(self, data: mujoco.MjData, min_pixels: int = 25) -> sv.Detections:
        """Detect visible target objects using pixel segmentation ground truth."""
        self.seg_renderer.update_scene(data, camera=self.camera_name)
        seg = self.seg_renderer.render()
        geom_field = seg[..., 0]
        type_field = seg[..., 1]
        is_geom = type_field == int(mujoco.mjtObj.mjOBJ_GEOM)

        boxes = []
        confidences = []
        class_ids = []

        for cid, g_ids in self.class_geom_ids.items():
            if not g_ids:
                continue
            mask = np.isin(geom_field, list(g_ids)) & is_geom
            pixel_count = int(np.sum(mask))
            if pixel_count >= min_pixels:
                y_idx, x_idx = np.where(mask)
                x1, y1 = float(np.min(x_idx)), float(np.min(y_idx))
                x2, y2 = float(np.max(x_idx)), float(np.max(y_idx))
                w = x2 - x1
                h = y2 - y1
                if w >= 4 and h >= 4:
                    boxes.append([x1, y1, x2, y2])
                    # Higher confidence if more pixels visible
                    conf = min(1.0, 0.5 + pixel_count / 500.0)
                    confidences.append(conf)
                    class_ids.append(cid)

        if not boxes:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            confidence=np.array(confidences, dtype=np.float32),
            class_id=np.array(class_ids, dtype=np.int32),
        )
