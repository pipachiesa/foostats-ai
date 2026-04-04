import math


class PassDetector:
    PIXEL_TO_METER = 0.1  # rough fallback when field coords unavailable

    def __init__(self, min_possession_frames=3, ball_travel_threshold=2.0, max_pass_frames=15, min_pass_gap=2):
        self.min_possession_frames = min_possession_frames
        self.ball_travel_threshold = ball_travel_threshold
        self.max_pass_frames = max_pass_frames
        self.min_pass_gap = min_pass_gap

    def detect(self, tracks, team_ball_control) -> list[dict]:
        """
        Detect passes and interceptions from tracking data.

        Returns list of detected passes:
        {
            "type": "pass" | "interception",
            "frame_start": int,
            "frame_end": int,
            "from_player": int,
            "to_player": int,
            "from_team": int,
            "to_team": int,
            "ball_start_pos": [x, y],
            "ball_end_pos": [x, y],
            "distance_m": float
        }
        """
        possession_per_frame = self._build_possession_per_frame(tracks)
        segments = self._build_possession_segments(possession_per_frame)
        print(f"  [PassDetector] {len(segments)} possession segments found")
        ball_positions = self._build_ball_positions(tracks)
        return self._match_passes(segments, ball_positions)

    def _build_possession_per_frame(self, tracks):
        """Build list of (player_id, team) or None per frame."""
        num_frames = len(tracks['players'])
        result = [None] * num_frames

        for frame_num in range(num_frames):
            for player_id, track in tracks['players'][frame_num].items():
                if track.get('has_ball'):
                    result[frame_num] = (player_id, track.get('team'))
                    break

        return result

    def _build_possession_segments(self, possession_per_frame):
        """
        Group consecutive frames where the same player has the ball.

        Returns list of dicts:
        {
            "player_id": int,
            "team": int,
            "frame_start": int,
            "frame_end": int,   # inclusive
            "length": int
        }
        """
        segments = []
        current_player = None
        current_team = None
        start_frame = None
        last_possession_frame = None  # track actual last frame with possession

        for frame_num, poss in enumerate(possession_per_frame):
            if poss is not None:
                player_id, team = poss
                if player_id == current_player:
                    last_possession_frame = frame_num  # extend
                else:
                    # close previous segment using actual last possession frame
                    if current_player is not None and last_possession_frame is not None:
                        segments.append({
                            'player_id': current_player,
                            'team': current_team,
                            'frame_start': start_frame,
                            'frame_end': last_possession_frame,  # actual last frame
                            'length': last_possession_frame - start_frame + 1,
                        })
                    current_player = player_id
                    current_team = team
                    start_frame = frame_num
                    last_possession_frame = frame_num

        # close last segment
        if current_player is not None and last_possession_frame is not None:
            segments.append({
                'player_id': current_player,
                'team': current_team,
                'frame_start': start_frame,
                'frame_end': last_possession_frame,
                'length': last_possession_frame - start_frame + 1,
            })

        return segments

    def _build_ball_positions(self, tracks):
        """
        Build list of (x, y, is_field_coords, is_interpolated) per frame.
        Returns None for frames with no ball.
        """
        num_frames = len(tracks['ball'])
        result = [None] * num_frames

        for frame_num in range(num_frames):
            ball_frame = tracks['ball'][frame_num]
            if 1 not in ball_frame:
                continue

            ball = ball_frame[1]
            interpolated = ball.get('interpolated', False)

            # Try field coords first (set by ViewTransformer if available)
            pos_t = ball.get('position_transformed')
            if pos_t is not None:
                result[frame_num] = (pos_t[0], pos_t[1], True, interpolated)
                continue

            # Try adjusted pixel position (set by camera movement estimator)
            pos = ball.get('position_adjusted') or ball.get('position')
            if pos is not None:
                result[frame_num] = (
                    pos[0] * self.PIXEL_TO_METER,
                    pos[1] * self.PIXEL_TO_METER,
                    False,
                    interpolated,
                )
                continue

            # Primary fallback: compute center from bbox (always available)
            bbox = ball.get('bbox')
            if bbox is not None:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                result[frame_num] = (
                    cx * self.PIXEL_TO_METER,
                    cy * self.PIXEL_TO_METER,
                    False,
                    interpolated,
                )

        return result

    def _get_ball_pos_at_frame(self, ball_positions, frame, allow_interpolated=False):
        """Get ball (x, y) at a specific frame, or None."""
        if frame < 0 or frame >= len(ball_positions):
            return None
        bp = ball_positions[frame]
        if bp is None:
            return None
        x, y, _is_field, is_interpolated = bp
        if is_interpolated and not allow_interpolated:
            return None
        return [x, y]

    def _distance(self, pos_a, pos_b):
        if pos_a is None or pos_b is None:
            return 0.0
        return math.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)

    def _match_passes(self, segments, ball_positions):
        """
        Check consecutive possession segments for pass events.

        A pass requires:
        1. Segment A has >= min_possession_frames
        2. Ball travels > ball_travel_threshold meters between end of A and start of B
        3. Gap between A ending and B starting <= max_pass_frames
        4. Segment B has >= min_possession_frames
        5. Player A != Player B
        """
        passes = []

        # Temporary debug — remove after verification
        rejected = {'short_a': 0, 'short_b': 0, 'same_player': 0, 'gap': 0, 'instant': 0, 'dist': 0}

        for i in range(len(segments) - 1):
            seg_a = segments[i]
            seg_b = segments[i + 1]

            if seg_a['length'] < self.min_possession_frames:
                rejected['short_a'] += 1
                continue
            if seg_b['length'] < self.min_possession_frames:
                rejected['short_b'] += 1
                continue
            if seg_a['player_id'] == seg_b['player_id']:
                rejected['same_player'] += 1
                continue

            gap = seg_b['frame_start'] - seg_a['frame_end']
            if gap > self.max_pass_frames:
                rejected['gap'] += 1
                continue
            if gap < self.min_pass_gap:
                rejected['instant'] += 1
                continue

            # Get ball position at end of A's possession (non-interpolated preferred)
            ball_start = None
            for f in range(seg_a['frame_end'], max(seg_a['frame_start'] - 1, seg_a['frame_end'] - 5) - 1, -1):
                ball_start = self._get_ball_pos_at_frame(ball_positions, f, allow_interpolated=False)
                if ball_start is not None:
                    break
            # fallback: allow interpolated
            if ball_start is None:
                ball_start = self._get_ball_pos_at_frame(ball_positions, seg_a['frame_end'], allow_interpolated=True)

            # Get ball position at start of B's possession (non-interpolated preferred)
            ball_end = None
            for f in range(seg_b['frame_start'], min(seg_b['frame_end'] + 1, seg_b['frame_start'] + 5) + 1):
                ball_end = self._get_ball_pos_at_frame(ball_positions, f, allow_interpolated=False)
                if ball_end is not None:
                    break
            if ball_end is None:
                ball_end = self._get_ball_pos_at_frame(ball_positions, seg_b['frame_start'], allow_interpolated=True)

            dist = self._distance(ball_start, ball_end)
            if dist < self.ball_travel_threshold:
                rejected['dist'] += 1
                continue

            is_same_team = seg_a['team'] == seg_b['team']

            passes.append({
                'type': 'pass' if is_same_team else 'interception',
                'frame_start': seg_a['frame_end'],
                'frame_end': seg_b['frame_start'],
                'from_player': seg_a['player_id'],
                'to_player': seg_b['player_id'],
                'from_team': seg_a['team'],
                'to_team': seg_b['team'],
                'ball_start_pos': ball_start if ball_start else [0, 0],
                'ball_end_pos': ball_end if ball_end else [0, 0],
                'distance_m': round(dist, 2),
            })

        print(f"  [PassDetector] rejected: {rejected}")
        return passes
