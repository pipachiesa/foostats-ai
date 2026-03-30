import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70
        self._last_assigned = -1  # estado para histéresis
        self._frames_with_current = 0  # max lock duration counter to prevent sticky possession

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = float('inf')
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        # Max possession lock: release hysteresis after 30 frames (~1.2s at 25fps)
        # to prevent one team holding possession indefinitely
        self._frames_with_current += 1
        if self._frames_with_current > 30:
            self._last_assigned = -1
            self._frames_with_current = 0

        # Histéresis: el poseedor actual mantiene la pelota a menos que
        # otro jugador esté al menos un 12% más cerca (was 20%, too sticky).
        # Reduced threshold fixes possession heavily skewing to one team.
        if self._last_assigned != -1 and self._last_assigned in players:
            last_bbox = players[self._last_assigned]['bbox']
            last_dist = min(
                measure_distance((last_bbox[0], last_bbox[-1]), ball_position),
                measure_distance((last_bbox[2], last_bbox[-1]), ball_position)
            )
            if last_dist < self.max_player_ball_distance:
                if assigned_player != self._last_assigned:
                    if minimum_distance > last_dist * 0.88:
                        assigned_player = self._last_assigned

        if assigned_player != -1:
            if assigned_player != self._last_assigned:
                self._frames_with_current = 0  # reset counter on possession change
            self._last_assigned = assigned_player

        return assigned_player
