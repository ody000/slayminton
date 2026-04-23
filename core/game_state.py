"""Game state management module.
Tracks rally status, score, and hit counts."""

from __future__ import annotations

import math
from typing import Optional, Tuple


ShuttleTuple = Tuple[float, float, float, float, float]

"""
GameState include:
- rally_active: bool - whether a rally is currently active
- score: dict - current score for each player
- hit_count: int - number of hits in the current rally
"""
class GameState:
    def __init__(self, inactive_timeout_s: float = 0.5, min_displacement_px: float = 2.0):
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

    def start_rally(self):
        self.rally_active = True
        self.hit_count = 0

    def end_rally(self, winner=None):
        if winner in self.score:
            self.score[winner] += 1
        self.rally_active = False
        self.hit_count = 0

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
        self.current_rally_start_timestamp = None

    def record_hit(self):
        if self.rally_active:
            self.hit_count += 1

    @staticmethod
    def _center_from_shuttle(shuttle_det: Optional[ShuttleTuple]) -> Optional[Tuple[float, float]]:
        # Convert shuttle detection tuple to center coordinates. Returns None if no detection.
        if shuttle_det is None:
            return None
        _, x, y, h, w = shuttle_det
        return (x + 0.5 * w, y + 0.5 * h)

    def update(self, timestamp: float, shuttle_det: Optional[ShuttleTuple]) -> bool:
        """Update state from one frame and return current rally_active."""
        center = self._center_from_shuttle(shuttle_det)

        if center is not None:
            if self.last_center is None:
                # First visible shuttle sample starts a candidate active period.
                self.last_motion_timestamp = timestamp
                if not self.rally_active:
                    self.current_rally_start_timestamp = timestamp
                self.start_rally()
            else:
                # Compare current shuttle center to last known center to determine motion.
                dx = center[0] - self.last_center[0]
                dy = center[1] - self.last_center[1]
                displacement = math.hypot(dx, dy)

                if displacement >= self.min_displacement_px:
                    self.last_motion_timestamp = timestamp
                    if not self.rally_active:
                        self.current_rally_start_timestamp = timestamp
                        self.start_rally()
                    self.record_hit()

            self.last_center = center

        if self.rally_active:
            # If no recent movement, rally is over.
            # If no shuttle detected, this also falls out since center will be None and last_motion_timestamp won't update.
            if self.last_motion_timestamp is None:
                self._record_rally_segment(timestamp)
                self.end_rally()
            elif (timestamp - self.last_motion_timestamp) >= self.inactive_timeout_s:
                # Rally end is pinned to last observed motion timestamp.
                self._record_rally_segment(self.last_motion_timestamp)
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

    def get_rally_data(self):
        return list(self.rally_data)
