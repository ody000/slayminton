"""Game state management module.
Tracks rally status, score, and hit counts."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


ShuttleTuple = Tuple[float, float, float, float, float]

"""
GameState include:
- rally_active: bool - whether a rally is currently active
- score: dict - current score for each player
- hit_count: int - number of hits in the current rally
"""
class GameState:
    def __init__(self, inactive_timeout_s: float = 1.0, min_displacement_px: float = 2.0):
        # motion-based rally rule config.
        self.inactive_timeout_s = float(inactive_timeout_s)
        self.min_displacement_px = float(min_displacement_px)

        self.rally_active = False
        self.last_center: Optional[Tuple[float, float]] = None
        self.last_motion_timestamp: Optional[float] = None
        self.current_rally_start_timestamp: Optional[float] = None
        self.rally_data = []

        self.score = {"player1": 0, "player2": 0}
        self.hit_count = 0

        # Trajectory prediction-based hit detection.
        self.position_history: List[Tuple[float, float, float]] = []  # (timestamp, x, y)
        self.history_max_len = 8  # keep last positions for trajectory fitting
        # Prediction error threshold for hit detection; leave moderate
        self.prediction_error_threshold = 20.0  # pixels; if prediction error > this, it's a hit
        self.last_hit_timestamp: Optional[float] = None
        self.hit_cooldown_s = 0.2  # minimum time between consecutive hits (200ms)

        # Motion-debounce to avoid single-frame jitter extending rallies.
        self.motion_streak: int = 0
        self.motion_required_streak: int = 3  # require this many frames of motion to accept it

        # Stability detector: if shuttle center is exactly unchanged for this
        # many frames, treat as background / inactive and end rally.
        self.stable_frames: int = 0
        self.stable_frame_threshold: int = 5

        # Large displacement filter: if displacement between two frames exceeds
        # this fraction of the frame size (height or width), consider the
        # detection invalid for tracking (ignored). Expressed as 1/6 of frame.
        self.max_displacement_fraction: float = 1.0 / 6.0

        # Visualization hints for current frame
        self.last_detection_discarded: bool = False  # True if latest detection was rejected by large-displacement filter
        self.consecutive_stationary_frames: int = 0  # count frames where shuttle doesn't move
        self.stationary_visualization_threshold: int = 3  # suppress visualization after this many stationary frames

    def start_rally(self):
        self.rally_active = True
        self.hit_count = 0
        ts = self.current_rally_start_timestamp
        if ts is None:
            print("[GAME] rally_start")
        else:
            print(f"[GAME] rally_start t={ts:.3f}s")

    def end_rally(self, winner=None):
        if winner in self.score:
            self.score[winner] += 1
        self.rally_active = False
        self.hit_count = 0
        print("[GAME] rally_end")

    def _record_rally_segment(self, end_timestamp: float):
        # Store structured rally entry for downstream analysis.
        if self.current_rally_start_timestamp is None:
            return
        duration_s = max(float(end_timestamp) - float(self.current_rally_start_timestamp), 0.0)
        self.rally_data.append(
            {
                "rally_id": len(self.rally_data) + 1,
                "start_time": float(self.current_rally_start_timestamp),
                "end_time": float(end_timestamp),
                "duration_s": float(duration_s),
            }
        )
        print(
            f"[GAME] rally_segment id={len(self.rally_data)} "
            f"start={self.current_rally_start_timestamp:.3f}s end={float(end_timestamp):.3f}s "
            f"duration={duration_s:.3f}s"
        )
        self.current_rally_start_timestamp = None

    def record_hit(self):
        if self.rally_active:
            self.hit_count += 1
            if self.last_motion_timestamp is not None:
                print(f"[GAME] hit t={self.last_motion_timestamp:.3f}s count={self.hit_count}")
            else:
                print(f"[GAME] hit count={self.hit_count}")

    def should_visualize_shuttle(self) -> bool:
        """Check if the current frame's shuttle should be visualized.
        
        Returns False if:
        - Last detection was discarded by large-displacement filter
        - Shuttle has been stationary for > threshold frames
        """
        if self.last_detection_discarded:
            return False
        if self.consecutive_stationary_frames > self.stationary_visualization_threshold:
            return False
        return True

    @staticmethod
    def _center_from_shuttle(shuttle_det: Optional[ShuttleTuple]) -> Optional[Tuple[float, float]]:
        # Convert shuttle detection tuple to center coordinates. Returns None if no detection.
        if shuttle_det is None:
            return None
        # Expect shuttle_det as (timestamp, x, y, w, h)
        _, x, y, w, h = shuttle_det
        return (x + 0.5 * w, y + 0.5 * h)

    def _fit_trajectory(self) -> Optional[Tuple[float, float, float]]:
        """Fit linear trajectory model to position history.
        Returns (vx, vy, avg_speed) or None if insufficient history.
        """
        if len(self.position_history) < 2:
            return None
        hist = self.position_history
        # Simple linear fit: use last two positions to estimate velocity.
        t1, x1, y1 = hist[-2]
        t2, x2, y2 = hist[-1]
        dt = max(t2 - t1, 1e-6)
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        speed = math.hypot(vx, vy)
        return (vx, vy, speed)

    def _predict_position(self, next_timestamp: float) -> Optional[Tuple[float, float]]:
        """Predict shuttle position at next_timestamp using linear trajectory model.
        Returns (pred_x, pred_y) or None if cannot predict.
        """
        if len(self.position_history) < 2:
            return None
        trajectory = self._fit_trajectory()
        if trajectory is None:
            return None
        vx, vy, _ = trajectory
        _, last_x, last_y = self.position_history[-1]
        last_t = self.position_history[-1][0]
        dt = next_timestamp - last_t
        pred_x = last_x + vx * dt
        pred_y = last_y + vy * dt
        return (pred_x, pred_y)

    def _detect_hit(self, current_pos: Tuple[float, float], current_timestamp: float) -> bool:
        """Detect if current position indicates a hit (trajectory discontinuity).
        Returns True if prediction error exceeds threshold and not in cooldown.
        """
        if len(self.position_history) < 2:
            # Not enough history yet; no hit detected.
            return False

        # Check cooldown: avoid multiple hits in quick succession.
        if self.last_hit_timestamp is not None:
            if (current_timestamp - self.last_hit_timestamp) < self.hit_cooldown_s:
                return False

        pred = self._predict_position(current_timestamp)
        if pred is None:
            return False

        pred_x, pred_y = pred
        cur_x, cur_y = current_pos
        error = math.hypot(cur_x - pred_x, cur_y - pred_y)

        if error > self.prediction_error_threshold:
            self.last_hit_timestamp = current_timestamp
            return True
        return False

    def update(self, timestamp: float, shuttle_det: Optional[ShuttleTuple], frame_size: Optional[Tuple[int, int]] = None) -> bool:
        """Update state from one frame and return current rally_active.

        frame_size: optional (height, width) tuple used to apply large-displacement
        filtering (ignore detections that move impossibly far between frames).
        """
        center = self._center_from_shuttle(shuttle_det)
        self.last_detection_discarded = False  # reset flag each frame

        # If no detection, clear transient stability counter and return later.
        if center is None:
            self.stable_frames = 0
            self.consecutive_stationary_frames = 0
        else:
            # If we have a previous center, consider large-displacement filter first.
            if self.last_center is not None:
                dx = center[0] - self.last_center[0]
                dy = center[1] - self.last_center[1]
                displacement = math.hypot(dx, dy)

                if frame_size is not None:
                    fh, fw = frame_size
                    max_allowed = max(fh, fw) * self.max_displacement_fraction
                    if displacement > max_allowed:
                        # Ignore this candidate as implausible for tracking: do not
                        # update history or motion streak. Mark for visualization suppression.
                        self.motion_streak = 0
                        self.last_detection_discarded = True
                        self.consecutive_stationary_frames = 0
                        return self.rally_active

                # Stability: if center did not move at all for several frames,
                # treat as background and end the rally.
                if displacement < 1e-3:
                    self.stable_frames += 1
                    self.consecutive_stationary_frames += 1
                    if self.stable_frames > self.stable_frame_threshold:
                        if self.rally_active:
                            self._record_rally_segment(timestamp)
                            self.end_rally()
                        # reset tracking buffers
                        self.last_center = None
                        self.last_motion_timestamp = None
                        self.position_history.clear()
                        self.motion_streak = 0
                        self.consecutive_stationary_frames = 0
                        return self.rally_active
                else:
                    self.stable_frames = 0
                    self.consecutive_stationary_frames = 0  # reset stationary counter on motion

            # Accept detection for tracking: append to history and proceed.
            self.position_history.append((timestamp, center[0], center[1]))
            if len(self.position_history) > self.history_max_len:
                self.position_history.pop(0)

            if self.last_center is None:
                # First visible shuttle sample starts a candidate active period.
                # Accept first detection as motion and start rally immediately.
                self.motion_streak = self.motion_required_streak
                self.last_motion_timestamp = timestamp
                if not self.rally_active:
                    self.current_rally_start_timestamp = timestamp
                self.start_rally()
                self.consecutive_stationary_frames = 0
            else:
                # Compare current shuttle center to last known center to determine motion.
                dx = center[0] - self.last_center[0]
                dy = center[1] - self.last_center[1]
                displacement = math.hypot(dx, dy)

                # Motion-debounce: require multiple consecutive frames exceeding displacement.
                if displacement >= self.min_displacement_px:
                    self.motion_streak = min(self.motion_streak + 1, self.motion_required_streak)
                    if self.motion_streak >= self.motion_required_streak:
                        self.last_motion_timestamp = timestamp
                        if not self.rally_active:
                            self.current_rally_start_timestamp = timestamp
                            self.start_rally()

                        # Check for hit using trajectory prediction-based detector.
                        if self._detect_hit(center, timestamp):
                            self.record_hit()
                    self.consecutive_stationary_frames = 0  # motion detected, reset counter
                else:
                    # Reset streak on small movement (likely jitter)
                    self.motion_streak = 0
                    self.consecutive_stationary_frames += 1  # track stationary frames

            self.last_center = center

        if self.rally_active:
            # If no recent movement, rally is over.
            # If no shuttle detected, this also falls out since center will be None and last_motion_timestamp won't update.
            if self.last_motion_timestamp is None:
                self._record_rally_segment(timestamp)
                self.end_rally()
            elif (timestamp - self.last_motion_timestamp) >= self.inactive_timeout_s:
                # Rally end is pinned to last observed motion timestamp.
                # Record the rally end at the current timestamp (when we observed inactivity).
                self._record_rally_segment(timestamp)
                self.end_rally()

        return self.rally_active

    def update_game_state(self, player_detected, shuttle_detected, timestamp: float = 0.0, shuttle_det=None):
        """Compatibility wrapper using your original method name.

        If a full shuttle detection tuple is provided, motion-based logic is used.
        Otherwise this falls back to simple visibility-based rally state updates.
        """
        if shuttle_det is not None:
            # For compatibility, if player supplies raw frame_size via kwargs,
            # accept it. Otherwise call update without frame size.
            fs = None
            # allow caller to pass (frame_h, frame_w) as attribute on shuttle_det tuple extras
            if isinstance(shuttle_det, tuple) and len(shuttle_det) >= 6:
                # last two elements may be frame_h, frame_w
                try:
                    fh = int(shuttle_det[5])
                    fw = int(shuttle_det[6])
                    fs = (fh, fw)
                except Exception:
                    fs = None
            return self.update(timestamp=timestamp, shuttle_det=shuttle_det, frame_size=fs)

        if player_detected and shuttle_detected:
            if not self.rally_active:
                self.current_rally_start_timestamp = timestamp
                self.start_rally()
            self.record_hit()
        elif self.rally_active and (not player_detected or not shuttle_detected):
            self._record_rally_segment(timestamp)
            self.end_rally()
        return self.rally_active

    def finalize_rally_data(self, final_timestamp: float):
        # If stream ends during an active rally, close it at final timestamp.
        if self.rally_active:
            self._record_rally_segment(final_timestamp)
            self.end_rally()
        print(f"[GAME] finalize rallies={len(self.rally_data)} at t={float(final_timestamp):.3f}s")

    def get_rally_data(self):
        return list(self.rally_data)


def build_rally_status_per_frame(rally_data: list, total_frames: int, fps: float) -> tuple[list[bool], list]:
    """Build a per-frame rally active status and consolidated rally data.
    
    Consolidates rally active/inactive periods shorter than 0.5s by merging them
    into the surrounding state. This prevents short false positives/negatives from
    fragmenting the rally detection.
    
    Args:
        rally_data: List of rally segments with start_time, end_time, duration_s
        total_frames: Total number of frames in the video
        fps: Frames per second
    
    Returns:
        rally_status: List of bool (True=rally active, False=no rally) per frame
        consolidated_rally_data: Updated rally data with merged short segments
    """
    frame_duration = 1.0 / max(fps, 1e-6)
    min_period_duration = 0.5  # seconds
    min_period_frames = max(1, int(min_period_duration / frame_duration))
    
    # Initialize rally status as all False
    rally_status = [False] * total_frames
    
    # Mark frames in each rally segment
    for rally in rally_data:
        start_frame = max(0, int(rally["start_time"] * fps))
        end_frame = min(total_frames - 1, int(rally["end_time"] * fps))
        for i in range(start_frame, end_frame + 1):
            if i < len(rally_status):
                rally_status[i] = True
    
    # Post-process: merge short non-rally periods into surrounding rally periods
    # Bias towards rally_active=True: short gaps of False get absorbed into True
    consolidated = rally_status.copy()
    i = 0
    while i < len(consolidated):
        current_state = consolidated[i]
        start_i = i
        # Count consecutive frames with same state
        while i < len(consolidated) and consolidated[i] == current_state:
            i += 1
        period_len = i - start_i
        
        # If period is shorter than threshold and is False (no rally),
        # merge it into surrounding True (rally active)
        if period_len < min_period_frames and current_state == False and start_i > 0:
            # Look ahead to see if next state is True
            next_state = consolidated[i] if i < len(consolidated) else None
            prev_state = consolidated[start_i - 1]
            
            # Prefer to merge into True if available
            if next_state == True or prev_state == True:
                merge_into = True
                for j in range(start_i, i):
                    consolidated[j] = merge_into
    
    # Rebuild rally_data from consolidated status
    consolidated_rally_data = []
    rally_id = 1
    in_rally = False
    rally_start = None
    
    for frame_idx, is_rally in enumerate(consolidated):
        if is_rally and not in_rally:
            # Start of a new rally
            rally_start = frame_idx * frame_duration
            in_rally = True
        elif not is_rally and in_rally:
            # End of a rally
            rally_end = frame_idx * frame_duration
            duration = max(rally_end - rally_start, 0.0)
            consolidated_rally_data.append({
                "rally_id": rally_id,
                "start_time": float(rally_start),
                "end_time": float(rally_end),
                "duration_s": float(duration),
            })
            rally_id += 1
            in_rally = False
    
    # Close any open rally at the end
    if in_rally and rally_start is not None:
        rally_end = (total_frames - 1) * frame_duration
        duration = max(rally_end - rally_start, 0.0)
        consolidated_rally_data.append({
            "rally_id": rally_id,
            "start_time": float(rally_start),
            "end_time": float(rally_end),
            "duration_s": float(duration),
        })
    
    return consolidated, consolidated_rally_data
