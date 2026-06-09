import math


# ---------------------------------------------------------------------------
# Field geometry (meters, origin at center).
# Field 105 x 68: X in [-52.5, 52.5], Y in [-34, 34].
# Goals at x = +/-52.5, posts at y = +/-3.66, goal depth ~2.4 m.
# All geometry below operates on position_transformed field coordinates.
# ---------------------------------------------------------------------------
HALF_X = 52.5          # half field length (end line)
HALF_Y = 34.0          # half field width (touch line)
GOAL_POST_Y = 3.66     # goal post half-width
GOAL_DEPTH = 2.4       # net depth behind the goal line
# Generous aim band used when testing whether a trajectory points at the goal
# mouth (a bit wider than the posts to tolerate tracking noise). Mirrors the
# reference implementation's +/-6 band, scaled to our goal.
PATH_NET_Y_TOL = 6.0
# Corner / center proximity zones (tuned to our field, from the reference).
CORNER_X = 50.0
CORNER_Y = 32.0
CENTER_RADIUS = 1.0


def is_ball_out(pos):
    """True if the ball position [x, y] (meters) is outside the field of play."""
    if pos is None:
        return False
    x = round(pos[0], 2)
    y = round(pos[1], 2)
    if x < -HALF_X or x > HALF_X:
        return True
    if y < -HALF_Y or y > HALF_Y:
        return True
    return False


def is_ball_in_corner(x, y):
    """True if (x, y) is in a corner region (near both an end line and a touch line)."""
    return abs(x) > CORNER_X and abs(y) > CORNER_Y


def is_ball_in_center(x, y):
    """True if (x, y) is in the center-circle region (kickoff spot)."""
    return abs(x) < CENTER_RADIUS and abs(y) < CENTER_RADIUS


def in_path_of_net(p1, p2):
    """True if the segment p1->p2 is aimed at the goal mouth.

    Extrapolates the p1->p2 direction to the goal line (x = +/-HALF_X) and checks
    whether it would cross within the goal-mouth aim band.
    """
    if p1 is None or p2 is None:
        return False
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return False
    NET_X = HALF_X if dx > 0 else -HALF_X
    t = (NET_X - x1) / dx
    if t <= 0:
        return False
    y_at_net = y1 + t * dy
    return -PATH_NET_Y_TOL <= y_at_net <= PATH_NET_Y_TOL


def in_net(x, y):
    """True if (x, y) sits inside the net volume behind either goal line."""
    if x <= -HALF_X and x >= -(HALF_X + GOAL_DEPTH):
        if -GOAL_POST_Y <= y <= GOAL_POST_Y:
            return True
    elif x >= HALF_X and x <= (HALF_X + GOAL_DEPTH):
        if -GOAL_POST_Y <= y <= GOAL_POST_Y:
            return True
    return False


def ball_maintains_direction(p1_start, p1_end, p2_start, p2_end,
                             angle_tol_deg=10,
                             collinear_tol=0.05):
    """True if the ball kept rolling along the same line through a possession change.

    When this is True between two possessions of different teams, the ball simply
    rolled past a player rather than being genuinely intercepted.
    """
    def vec(a, b):
        return (b[0] - a[0], b[1] - a[1])

    def norm(v):
        return math.hypot(v[0], v[1])

    def normalize(v):
        n = norm(v)
        if n == 0:
            return None
        return (v[0] / n, v[1] / n)

    if p1_start is None or p1_end is None or p2_start is None or p2_end is None:
        return False

    v1 = vec(p1_start, p1_end)
    v2 = vec(p2_start, p2_end)
    n1 = normalize(v1)
    n2 = normalize(v2)
    if n1 is None or n2 is None:
        return False
    dot = n1[0] * n2[0] + n1[1] * n2[1]
    dot = max(-1.0, min(1.0, dot))
    angle = math.degrees(math.acos(dot))
    if angle > angle_tol_deg:
        return False
    cross = abs(
        (p2_start[0] - p1_start[0]) * v1[1] -
        (p2_start[1] - p1_start[1]) * v1[0]
    )
    line_len = norm(v1)
    if line_len == 0:
        return False
    if cross / line_len > collinear_tol:
        return False
    return True


class PassDetector:
    def __init__(self, min_possession_frames=8, ball_travel_threshold=3.5,
                 max_pass_frames=15, min_receiver_frames=8):
        # min_possession_frames: a player must hold the ball for >= this many frames
        #   (~0.32s at 25fps) before/after an event for it to count. Raised from 3 to 8
        #   to filter single-frame possession flips from noisy ball assignment.
        # ball_travel_threshold: ball must travel >= this many METERS (field coords)
        #   to count as a pass. Raised from 2.0 to 3.5 to filter micro-movements.
        # min_receiver_frames: receiving segment must span >= this many frames.
        self.min_possession_frames = min_possession_frames
        self.ball_travel_threshold = ball_travel_threshold
        self.max_pass_frames = max_pass_frames
        self.min_receiver_frames = min_receiver_frames

    def detect(self, tracks, team_ball_control) -> list[dict]:
        """
        Detect on-ball events from tracking data.

        Returns a flat list of events sharing one base schema, distinguished by
        "type": "pass", "interception", "shot", "goal", "ball_out", "set_piece".
        Each event:
        {
            "type": str,
            "subtype": str | None,
            "frame_start": int,
            "frame_end": int,
            "from_player": int | None,
            "to_player": int | None,
            "from_team": int | None,
            "to_team": int | None,
            "ball_start_pos": [x, y],
            "ball_end_pos": [x, y],
            "distance_m": float
        }
        """
        possession_per_frame = self._build_possession_per_frame(tracks)
        segments = self._build_possession_segments(possession_per_frame)
        print(f"  [PassDetector] {len(segments)} possession segments found")
        ball_positions = self._build_ball_positions(tracks)

        # Transitions where the ball left the field are stoppages, not passes.
        out_transitions = self._find_out_transitions(segments, ball_positions)

        passes = self._match_passes(segments, ball_positions, skip_transitions=set(out_transitions))
        field_events = self._detect_field_events(segments, ball_positions, out_transitions)

        events = passes + field_events
        events.sort(key=lambda e: (e['frame_start'], e['frame_end']))

        goals = sum(1 for e in events if e['type'] == 'goal')
        shots = sum(1 for e in events if e['type'] == 'shot')
        outs = sum(1 for e in events if e['type'] == 'ball_out')
        setp = sum(1 for e in events if e['type'] == 'set_piece')
        print(f"  [PassDetector] field events: {goals} goals, {shots} shots, "
              f"{outs} ball-outs, {setp} set pieces")
        return events

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
        Returns None for frames with no ball OR no field coordinates.

        Only the ViewTransformer's position_transformed (field meters) is used.
        Pixel-based fallbacks (bbox-center / position_adjusted scaled by an ad-hoc
        PIXEL_TO_METER) were removed: mixing pixel-scaled values into field-metre
        geometry produced out-of-range coordinates (e.g. x=120) and corrupted pass
        distances. When field coords are unavailable for a frame we leave it None;
        the nearby-frame search in _match_passes recovers a valid adjacent reading.
        """
        num_frames = len(tracks['ball'])
        result = [None] * num_frames

        for frame_num in range(num_frames):
            ball_frame = tracks['ball'][frame_num]
            if 1 not in ball_frame:
                continue

            ball = ball_frame[1]
            interpolated = ball.get('interpolated', False)

            pos_t = ball.get('position_transformed')
            if pos_t is not None:
                result[frame_num] = (pos_t[0], pos_t[1], True, interpolated)

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

    def _get_ball_field_pos(self, ball_positions, frame, allow_interpolated=True):
        """Get ball (x, y) at a frame, but only when they are real field coords.

        Field-event geometry (out-of-bounds, goals, set pieces) is only meaningful
        in meters. If the ball position is a pixel fallback, return None so those
        detectors degrade gracefully instead of firing on garbage coordinates.
        """
        if frame < 0 or frame >= len(ball_positions):
            return None
        bp = ball_positions[frame]
        if bp is None:
            return None
        x, y, is_field, is_interpolated = bp
        if not is_field:
            return None
        if is_interpolated and not allow_interpolated:
            return None
        return [x, y]

    def _distance(self, pos_a, pos_b):
        if pos_a is None or pos_b is None:
            return 0.0
        return math.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)

    def _make_event(self, type_, subtype, frame_start, frame_end,
                    from_player, to_player, from_team, to_team,
                    ball_start, ball_end):
        """Assemble an event dict in the shared base schema."""
        return {
            'type': type_,
            'subtype': subtype,
            'frame_start': frame_start,
            'frame_end': frame_end,
            'from_player': from_player,
            'to_player': to_player,
            'from_team': from_team,
            'to_team': to_team,
            'ball_start_pos': ball_start if ball_start else [0, 0],
            'ball_end_pos': ball_end if ball_end else [0, 0],
            'distance_m': round(self._distance(ball_start, ball_end), 2),
        }

    def _ball_pos_near_frame_end(self, ball_positions, seg, allow_interpolated=False):
        """Ball position around the end of a possession segment (search backwards)."""
        ball_start = None
        for f in range(seg['frame_end'],
                       max(seg['frame_start'] - 1, seg['frame_end'] - 5) - 1, -1):
            ball_start = self._get_ball_pos_at_frame(ball_positions, f, allow_interpolated=allow_interpolated)
            if ball_start is not None:
                break
        if ball_start is None:
            ball_start = self._get_ball_pos_at_frame(ball_positions, seg['frame_end'], allow_interpolated=True)
        return ball_start

    def _ball_pos_near_frame_start(self, ball_positions, seg, allow_interpolated=False):
        """Ball position around the start of a possession segment (search forwards)."""
        ball_end = None
        for f in range(seg['frame_start'],
                       min(seg['frame_end'] + 1, seg['frame_start'] + 5) + 1):
            ball_end = self._get_ball_pos_at_frame(ball_positions, f, allow_interpolated=allow_interpolated)
            if ball_end is not None:
                break
        if ball_end is None:
            ball_end = self._get_ball_pos_at_frame(ball_positions, seg['frame_start'], allow_interpolated=True)
        return ball_end

    def _match_passes(self, segments, ball_positions, skip_transitions=None):
        """
        Check consecutive possession segments for pass / interception events.

        A pass requires:
        1. Segment A has >= min_possession_frames
        2. Ball travels > ball_travel_threshold meters between end of A and start of B
        3. Gap between A ending and B starting <= max_pass_frames
        4. Segment B has >= min_possession_frames
        5. Player A != Player B

        When the teams differ (interception candidate) we additionally check the
        ball direction: if the ball simply kept rolling along the same line
        (ball_maintains_direction) it rolled past the player and is not a real
        interception, so we skip it.

        Transitions in ``skip_transitions`` are stoppages (ball went out) and are
        skipped here — they are emitted as ball_out / set_piece / shot / goal by
        _detect_field_events instead.
        """
        if skip_transitions is None:
            skip_transitions = set()
        passes = []

        # Temporary debug — remove after verification
        rejected = {'short_a': 0, 'short_b': 0, 'short_receiver': 0, 'same_player': 0,
                    'gap': 0, 'dist': 0, 'out': 0, 'rolled_past': 0}

        for i in range(len(segments) - 1):
            seg_a = segments[i]
            seg_b = segments[i + 1]

            if i in skip_transitions:
                rejected['out'] += 1
                continue
            if seg_a['length'] < self.min_possession_frames:
                rejected['short_a'] += 1
                continue
            if seg_b['length'] < self.min_possession_frames:
                rejected['short_b'] += 1
                continue
            # Receiver must hold the ball for a real duration, not a 1-frame blip.
            if seg_b['frame_end'] - seg_b['frame_start'] < self.min_receiver_frames:
                rejected['short_receiver'] += 1
                continue
            if seg_a['player_id'] == seg_b['player_id']:
                rejected['same_player'] += 1
                continue

            gap = seg_b['frame_start'] - seg_a['frame_end']
            if gap > self.max_pass_frames:
                rejected['gap'] += 1
                continue

            ball_start = self._ball_pos_near_frame_end(ball_positions, seg_a)
            ball_end = self._ball_pos_near_frame_start(ball_positions, seg_b)

            dist = self._distance(ball_start, ball_end)
            if dist < self.ball_travel_threshold:
                rejected['dist'] += 1
                continue

            is_same_team = seg_a['team'] == seg_b['team']

            # Geometry-based interception filter: if the ball kept the same
            # heading through the team change, it rolled past — not intercepted.
            if not is_same_team:
                a_start = self._get_ball_pos_at_frame(ball_positions, seg_a['frame_start'], allow_interpolated=True)
                a_end = self._get_ball_pos_at_frame(ball_positions, seg_a['frame_end'], allow_interpolated=True)
                b_start = self._get_ball_pos_at_frame(ball_positions, seg_b['frame_start'], allow_interpolated=True)
                b_end = self._get_ball_pos_at_frame(ball_positions, seg_b['frame_end'], allow_interpolated=True)
                if ball_maintains_direction(a_start, a_end, b_start, b_end):
                    rejected['rolled_past'] += 1
                    continue

            passes.append(self._make_event(
                'pass' if is_same_team else 'interception',
                None,
                seg_a['frame_end'], seg_b['frame_start'],
                seg_a['player_id'], seg_b['player_id'],
                seg_a['team'], seg_b['team'],
                ball_start, ball_end,
            ))

        print(f"  [PassDetector] rejected: {rejected}")
        return passes

    def _find_out_transitions(self, segments, ball_positions):
        """Find possession transitions where the ball left the field.

        Returns {i: (out_xy, out_frame)} mapping the index of seg_a (the transition
        between segments[i] and segments[i+1]) to where/when the ball went out.
        Only field-coordinate ball positions are considered.
        """
        out_transitions = {}
        for i in range(len(segments) - 1):
            seg_a = segments[i]
            seg_b = segments[i + 1]
            for f in range(seg_a['frame_end'], seg_b['frame_start'] + 1):
                pos = self._get_ball_field_pos(ball_positions, f, allow_interpolated=True)
                if pos is not None and is_ball_out(pos):
                    out_transitions[i] = (pos, f)
                    break
        return out_transitions

    def _detect_field_events(self, segments, ball_positions, out_transitions):
        """Emit ball_out, set_piece, shot and goal events from ball-out transitions.

        Uses a 3-possession sliding window: shooter (A) -> ball out -> restart (B).
        A restart in the center circle (kickoff) confirms a goal.
        """
        events = []

        # Match-start / restart kickoff: first possession that begins at center.
        if segments:
            first_start = self._ball_pos_near_frame_start(ball_positions, segments[0], allow_interpolated=True)
            if (first_start is not None
                    and self._get_ball_field_pos(ball_positions, segments[0]['frame_start'], allow_interpolated=True) is not None
                    and is_ball_in_center(first_start[0], first_start[1])):
                events.append(self._make_event(
                    'set_piece', 'kick_off',
                    segments[0]['frame_start'], segments[0]['frame_start'],
                    None, segments[0]['player_id'],
                    None, segments[0]['team'],
                    first_start, first_start,
                ))

        for i, (out_xy, out_frame) in sorted(out_transitions.items()):
            seg_a = segments[i]
            seg_b = segments[i + 1]

            shooter_end = self._ball_pos_near_frame_end(ball_positions, seg_a, allow_interpolated=True)
            restart_start = self._ball_pos_near_frame_start(ball_positions, seg_b, allow_interpolated=True)

            toward_net = in_path_of_net(shooter_end, out_xy)
            crossed_end_line = abs(out_xy[0]) > HALF_X
            crossed_touch = abs(out_xy[1]) > HALF_Y
            reaches_net = in_net(out_xy[0], out_xy[1])
            restart_center = (restart_start is not None
                              and is_ball_in_center(restart_start[0], restart_start[1]))

            if toward_net and crossed_end_line:
                # Shot on target. Kickoff afterwards confirms it as a goal.
                if reaches_net and restart_center:
                    events.append(self._make_event(
                        'goal', None,
                        seg_a['frame_end'], out_frame,
                        seg_a['player_id'], None,
                        seg_a['team'], None,
                        shooter_end, out_xy,
                    ))
                    events.append(self._make_event(
                        'set_piece', 'kick_off',
                        out_frame, seg_b['frame_start'],
                        None, seg_b['player_id'],
                        None, seg_b['team'],
                        out_xy, restart_start,
                    ))
                else:
                    events.append(self._make_event(
                        'shot', None,
                        seg_a['frame_end'], out_frame,
                        seg_a['player_id'], None,
                        seg_a['team'], None,
                        shooter_end, out_xy,
                    ))
                continue

            # Plain ball out -> classify the stoppage and the restart set piece.
            ball_out_sub, set_piece_sub = self._classify_set_piece(out_xy, crossed_end_line, crossed_touch)

            events.append(self._make_event(
                'ball_out', ball_out_sub,
                seg_a['frame_end'], out_frame,
                seg_a['player_id'], None,
                seg_a['team'], None,
                shooter_end, out_xy,
            ))
            events.append(self._make_event(
                'set_piece', set_piece_sub,
                out_frame, seg_b['frame_start'],
                None, seg_b['player_id'],
                None, seg_b['team'],
                out_xy, restart_start,
            ))

        return events

    def _classify_set_piece(self, out_xy, crossed_end_line, crossed_touch):
        """Return (ball_out_subtype, set_piece_subtype) for a ball that went out.

        Classified from where the ball crossed the line (== the restart location):
          - corner zone over end line -> corner / corner_kick
          - over end line, not a corner -> goal_kick / goal_kick
          - over touch line -> throw_in / throw_in
        """
        x, y = out_xy[0], out_xy[1]
        if crossed_end_line and is_ball_in_corner(x, y):
            return 'corner', 'corner_kick'
        if crossed_end_line:
            return 'goal_kick', 'goal_kick'
        if crossed_touch:
            return 'throw_in', 'throw_in'
        return None, None


if __name__ == "__main__":
    # ---- Lightweight geometry unit tests (no external deps) ----
    results = []

    def check(name, got, expected):
        ok = got == expected
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={got} expected={expected}")

    # is_ball_out
    check("is_ball_out center", is_ball_out([0, 0]), False)
    check("is_ball_out past end line", is_ball_out([53, 0]), True)
    check("is_ball_out on end line (52.5)", is_ball_out([52.5, 0]), False)
    check("is_ball_out past touch line", is_ball_out([0, 35]), True)
    check("is_ball_out neg end line", is_ball_out([-53, 0]), True)
    check("is_ball_out None", is_ball_out(None), False)

    # corner / center zones
    check("is_ball_in_corner true", is_ball_in_corner(51, 33), True)
    check("is_ball_in_corner center", is_ball_in_corner(0, 0), False)
    check("is_ball_in_corner edge-only-x", is_ball_in_corner(51, 0), False)
    check("is_ball_in_center true", is_ball_in_center(0.5, -0.5), True)
    check("is_ball_in_center false", is_ball_in_center(2, 0), False)

    # in_path_of_net
    check("in_path straight at goal", in_path_of_net([40, 0], [52.5, 0]), True)
    check("in_path vertical (dx=0)", in_path_of_net([40, 0], [40, 10]), False)
    check("in_path wide miss", in_path_of_net([40, 0], [52.5, 20]), False)
    check("in_path net behind start (t<=0)", in_path_of_net([53, 0], [54, 0]), False)
    check("in_path toward left goal", in_path_of_net([-40, 0], [-52.5, 1]), True)

    # in_net
    check("in_net just inside right", in_net(52.6, 0), True)
    check("in_net right but wide", in_net(52.6, 5), False)
    check("in_net midfield", in_net(0, 0), False)
    check("in_net left goal", in_net(-52.6, 0), True)
    check("in_net too deep", in_net(60, 0), False)

    # ball_maintains_direction
    check("rolled straight through", ball_maintains_direction([0, 0], [1, 0], [2, 0], [3, 0]), True)
    check("direction changed 90deg", ball_maintains_direction([0, 0], [1, 0], [1, 0], [1, 1]), False)
    check("parallel but offset (not collinear)", ball_maintains_direction([0, 0], [1, 0], [2, 1], [3, 1]), False)
    check("maintains None input", ball_maintains_direction(None, [1, 0], [2, 0], [3, 0]), False)

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} checks passed — {'ALL PASS' if passed == total else 'SOME FAILED'}")
