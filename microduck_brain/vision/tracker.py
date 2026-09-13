"""Multi-Object Visual Tracker for Microduck using Roboflow trackers & BIoU.

Based on Alex Bodner's microduck-tracking architecture:
- SORTTracker with Buffered IoU (buffer_ratio=2.0) to hold identity through walking head-bob.
- Velocity EMA in image coordinates to single out moving vs static targets.
- Bearing and range computation for visual servoing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np
import supervision as sv
from trackers import SORTTracker
from trackers.utils.iou import BIoU


@dataclass
class TrackInfo:
    track_id: int
    class_id: int
    xyxy: np.ndarray
    center_xy: tuple[float, float]
    velocity_ema: tuple[float, float] = (0.0, 0.0)
    age_frames: int = 1
    hits: int = 1


@dataclass
class VisualTrackState:
    active_tracks: dict[int, TrackInfo] = field(default_factory=dict)
    target_track_id: int | None = None
    target_locked: bool = False
    bearing_rad: float = 0.0
    range_m: float = 1.0
    target_class_id: int = 0
    annotated_frame: np.ndarray | None = None


class MicroduckVisualTracker:
    """Visual tracking system managing persistent tracks and visual servoing targets."""

    def __init__(
        self,
        frame_rate: float = 50.0,
        buffer_ratio: float = 2.0,
        image_width: int = 480,
        image_height: int = 480,
    ) -> None:
        self.frame_rate = frame_rate
        self.width = image_width
        self.height = image_height

        # Robust SORT tracker with Buffered IoU matching Alex Bodner's configuration
        self.tracker = SORTTracker(frame_rate=frame_rate, iou=BIoU(buffer_ratio=buffer_ratio))

        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

        self.tracks: dict[int, TrackInfo] = {}
        self.target_track_id: int | None = None
        self.target_class_id: int = 0  # 0=marker by default

        fovy_deg = 90.0
        fovy_rad = math.radians(fovy_deg)
        self.fx = (self.width / 2.0) / math.tan(fovy_rad / 2.0)
        self.fy = (self.height / 2.0) / math.tan(fovy_rad / 2.0)

        # Approximate real object heights (meters)
        self.real_dimensions = {
            0: 0.12,  # marker
            1: 0.07,  # red obstacle box
            2: 0.06,  # transparent container
        }

    def set_target_class(self, class_id: int) -> None:
        """Set which class to lock onto."""
        if self.target_class_id != class_id:
            self.target_class_id = class_id
            self.target_track_id = None

    def update(
        self,
        detections: sv.Detections,
        rgb_frame: np.ndarray | None = None,
    ) -> VisualTrackState:
        """Update tracker with new detections and return current visual tracking state."""
        if len(detections) > 0:
            tracked = self.tracker.update(detections)
        else:
            tracked = sv.Detections.empty()

        current_track_ids = set()

        if len(tracked) > 0 and tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                tid = int(tid)
                current_track_ids.add(tid)
                box = tracked.xyxy[i]
                cid = int(tracked.class_id[i]) if tracked.class_id is not None else 0
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0

                if tid in self.tracks:
                    prev = self.tracks[tid]
                    dt = 1.0 / self.frame_rate
                    vx = (cx - prev.center_xy[0]) / dt
                    vy = (cy - prev.center_xy[1]) / dt
                    # EMA smoothing alpha=0.3
                    ema_vx = 0.7 * prev.velocity_ema[0] + 0.3 * vx
                    ema_vy = 0.7 * prev.velocity_ema[1] + 0.3 * vy

                    prev.xyxy = box
                    prev.center_xy = (cx, cy)
                    prev.velocity_ema = (ema_vx, ema_vy)
                    prev.age_frames += 1
                    prev.hits += 1
                else:
                    self.tracks[tid] = TrackInfo(
                        track_id=tid,
                        class_id=cid,
                        xyxy=box,
                        center_xy=(cx, cy),
                        velocity_ema=(0.0, 0.0),
                        age_frames=1,
                        hits=1,
                    )

        # Prune stale tracks that disappeared
        stale_ids = [tid for tid in self.tracks if tid not in current_track_ids]
        for tid in stale_ids:
            # Drop after 10 frames of disappearance
            if self.tracks[tid].age_frames > 15:
                del self.tracks[tid]

        # Target selection / locking logic
        target_locked = False
        bearing_rad = 0.0
        range_m = 1.2

        # Check if current locked target is still active
        if self.target_track_id is not None and self.target_track_id not in self.tracks:
            self.target_track_id = None

        # Lock onto best track matching target class
        if self.target_track_id is None:
            confirmed = [
                t for t in self.tracks.values()
                if t.class_id == self.target_class_id and t.hits >= 1 and t.track_id >= 0
            ]
            candidates = confirmed if confirmed else [
                t for t in self.tracks.values()
                if t.class_id == self.target_class_id and t.hits >= 1
            ]
            if candidates:
                # Pick the candidate closest to image center
                candidates.sort(key=lambda c: abs(c.center_xy[0] - self.width / 2.0))
                self.target_track_id = candidates[0].track_id

        if self.target_track_id is not None and self.target_track_id in self.tracks:
            target_track = self.tracks[self.target_track_id]
            target_locked = True

            # Compute horizontal bearing error (radians)
            # Positive bearing = target is to the right
            cx = target_track.center_xy[0]
            dx = cx - (self.width / 2.0)
            bearing_rad = math.atan2(dx, self.fx)

            # Compute range estimate from bounding box height
            box = target_track.xyxy
            box_h = max(4.0, float(box[3] - box[1]))
            real_dim = self.real_dimensions.get(self.target_class_id, 0.10)
            range_m = float(np.clip((self.fy * real_dim) / box_h, 0.08, 3.0))

        # Annotate RGB frame if provided
        annotated = None
        if rgb_frame is not None:
            annotated = rgb_frame.copy()
            if len(tracked) > 0 and tracked.tracker_id is not None:
                labels = []
                for tid in tracked.tracker_id:
                    tid_int = int(tid)
                    disp_id = tid_int + 1 if tid_int >= 0 else 1
                    is_locked = (tid_int == self.target_track_id)
                    labels.append(f"ID:{disp_id} {'[LOCKED]' if is_locked else ''}")
                annotated = self.box_annotator.annotate(annotated, tracked)
                annotated = self.label_annotator.annotate(annotated, tracked, labels=labels)

        return VisualTrackState(
            active_tracks=self.tracks,
            target_track_id=self.target_track_id,
            target_locked=target_locked,
            bearing_rad=bearing_rad,
            range_m=range_m,
            target_class_id=self.target_class_id,
            annotated_frame=annotated,
        )
