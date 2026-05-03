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
    def __init__(self, inactive_timeout_s: float = 0.8, min_displacement_px: float = 3.0):
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
        self.motion_required_streak: int = 2  # require this many frames of motion to accept it

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

    def update(self, timestamp: float, shuttle_det: Optional[ShuttleTuple]) -> bool:
        """Update state from one frame and return current rally_active."""
        center = self._center_from_shuttle(shuttle_det)

        if center is not None:
            # Update position history for trajectory tracking.
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
                else:
                    # Reset streak on small movement (likely jitter)
                    self.motion_streak = 0

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
            return self.update(timestamp=timestamp, shuttle_det=shuttle_det)

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
