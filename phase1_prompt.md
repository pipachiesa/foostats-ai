# Phase 1 Implementation Prompt — foostats_ai

You are implementing two concrete improvements to a football analytics pipeline:
1. Fix goalkeeper team classification (currently broken by design)
2. Improve ball position interpolation (currently too naive)

Do NOT add features beyond what is described. Do NOT refactor unrelated code.

---

## Context

The pipeline detects: `players`, `referees`, `ball`, `goalkeeper` (YOLO classes).

Current broken behavior: **line 79-80 of `trackers/tracker.py` immediately converts `goalkeeper` → `player`** before tracking, losing the class entirely. This means goalkeepers enter KMeans color clustering alongside outfield players, which corrupts team assignment (goalkeepers wear a different jersey color than their team).

The `tracks` dict currently has keys: `players`, `referees`, `ball`.

---

## Fix 1 — Goalkeeper tracking and team assignment

### 1a. tracker.py — track goalkeepers separately

In `get_object_tracks`, **stop converting goalkeeper → player**. Instead, route them to a new `goalkeepers` track key.

Replace this block:
```python
# Convert GK to Player object
for object_ind, class_id in enumerate(detection_supervision.class_id):
    if cls_names[class_id] == "goalkeeper":
        detection_supervision.class_id[object_ind] = cls_names_inv["player"]
```

With logic that:
- Keeps the goalkeeper class as-is in the detections
- Adds a `"goalkeepers": []` key to the `tracks` dict (alongside `players`, `referees`, `ball`)
- Routes detections with class `goalkeeper` to `tracks["goalkeepers"][frame_num][track_id]`
- Continues routing `player` → `tracks["players"]` as before

The goalkeeper detections DO go through ByteTrack (they need stable IDs), so they should remain in `detection_with_tracks`. The change is in the routing logic after tracking, not before.

### 1b. team_assigner.py — assign goalkeeper team by field position

Add a new method `get_goalkeeper_team(frame, goalkeeper_bbox, frame_width)`:

```python
def get_goalkeeper_team(self, frame, goalkeeper_bbox, frame_width):
    """
    Assign team to goalkeeper based on which half of the field they occupy.
    Goalkeepers stay near their own goal, so their x-position is a reliable signal.
    Returns team_id 1 or 2.
    """
    x_center = (goalkeeper_bbox[0] + goalkeeper_bbox[2]) / 2
    if x_center < frame_width / 2:
        return 1
    else:
        return 2
```

Important: `assign_team_color` must ONLY use `tracks['players']` (outfield players), never `tracks['goalkeepers']`. Verify this is already the case — it is called with `tracks['players'][0]` in `main.py`, so no change needed there.

### 1c. main.py — assign teams to goalkeepers and include in annotations

After the existing team assignment loop for players, add a loop for goalkeepers:

```python
frame_width = video_frames[0].shape[1]
for frame_num, goalkeeper_track in enumerate(tracks['goalkeepers']):
    for gk_id, track in goalkeeper_track.items():
        team = team_assigner.get_goalkeeper_team(
            video_frames[frame_num],
            track['bbox'],
            frame_width
        )
        tracks['goalkeepers'][frame_num][gk_id]['team'] = team
        tracks['goalkeepers'][frame_num][gk_id]['team_color'] = team_assigner.team_colors[team]
```

### 1d. tracker.py — draw goalkeepers in annotations

In `draw_annotations`, add goalkeeper drawing using the same `draw_ellipse` logic as players but with a distinct visual marker. A reasonable choice: draw them with their team color but add a 'GK' label instead of track_id, or draw with a slightly different ellipse style.

Minimal implementation — draw with team color and track_id like players:
```python
goalkeeper_dict = tracks["goalkeepers"][frame_num]
for track_id, goalkeeper in goalkeeper_dict.items():
    color = goalkeeper.get("team_color", (0, 255, 255))
    frame = self.draw_ellipse(frame, goalkeeper["bbox"], color, track_id)
```

Add this in `draw_annotations` right after the player drawing block.

### 1e. tracker.py — add_position_to_tracks must handle goalkeepers

In `add_position_to_tracks`, the loop iterates over `tracks.items()`. Since goalkeepers are now a separate key, they will naturally be included. Verify the loop handles `"goalkeepers"` the same as `"players"` (foot position). No change needed if the loop is generic, but confirm.

---

## Fix 2 — Ball interpolation improvement

### Current problem

`interpolate_ball_positions` in `tracker.py` uses `pd.DataFrame.interpolate()` (linear) followed by `bfill()`. This fills ALL gaps regardless of length. A 60-frame gap (2.4 seconds at 25fps) gets linearly filled with positions that are almost certainly wrong — the ball moved, bounced, went out of frame, etc.

### What to implement

Replace the current implementation with one that:

1. **Only interpolates short gaps** (≤ 15 frames). Longer gaps are left as `NaN` / empty — do not fill them.
2. **Uses polynomial order-2 interpolation** (`method='polynomial', order=2`) instead of linear for smoother trajectories over short gaps.
3. **Tags interpolated frames** by adding an `interpolated` boolean flag to the bbox dict, so downstream code (possession, event detection) knows whether a ball position is real or estimated.

```python
def interpolate_ball_positions(self, ball_positions):
    ball_positions_list = [x.get(1, {}).get("bbox", []) for x in ball_positions]
    df = pd.DataFrame(ball_positions_list, columns=["x1", "y1", "x2", "y2"])

    # Track which rows were originally missing
    missing_mask = df["x1"].isna()

    # Only interpolate gaps of <= 15 consecutive missing frames
    # Build a gap-length mask: for each NaN, find how long its gap is
    gap_too_long = pd.Series(False, index=df.index)
    in_gap = False
    gap_start = 0
    for i in range(len(df)):
        if df["x1"].isna().iloc[i]:
            if not in_gap:
                gap_start = i
                in_gap = True
        else:
            if in_gap:
                gap_len = i - gap_start
                if gap_len > 15:
                    gap_too_long.iloc[gap_start:i] = True
                in_gap = False
    if in_gap:  # handle trailing gap
        gap_too_long.iloc[gap_start:] = True

    # Interpolate (polynomial order 2 handles short gaps well)
    df_interp = df.interpolate(method='polynomial', order=2)
    df_interp = df_interp.bfill()  # fill leading NaNs if video starts with no ball

    # Re-apply NaN to long gaps (don't fill those)
    df_interp[gap_too_long] = float('nan')

    # Reconstruct the list with interpolated flag
    result = []
    for i, row in df_interp.iterrows():
        if pd.isna(row["x1"]):
            result.append({})  # no ball detected, no interpolation possible
        else:
            bbox = row.tolist()
            was_interpolated = bool(missing_mask.iloc[i])
            result.append({1: {"bbox": bbox, "interpolated": was_interpolated}})

    return result
```

Note: `pd.DataFrame.interpolate(method='polynomial', order=2)` requires at least 3 non-NaN points nearby to work. If there aren't enough, it silently falls back to NaN — which is fine, the `bfill` handles leading NaNs only, and long gaps remain empty.

### Update downstream code that accesses ball bbox

In `main.py`, the ball possession loop accesses `tracks['ball'][frame_num][1]['bbox']` unconditionally. After this change, frames with no ball (long gaps) will have `tracks['ball'][frame_num] == {}`. Add a guard:

```python
ball_frame = tracks['ball'][frame_num]
if 1 not in ball_frame:
    team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
    continue
ball_bbox = ball_frame[1]['bbox']
```

---

## Verification checklist

After implementing:

1. Run `python -c "from trackers import Tracker; print('OK')"` — no import errors
2. Run `python -c "from teams_assigner import TeamAssigner; print('OK')"` — no import errors
3. Confirm `tracks` dict has 4 keys: `players`, `referees`, `ball`, `goalkeepers`
4. Confirm `assign_team_color` is still only called with `tracks['players'][0]` (no goalkeepers in clustering)
5. Confirm `draw_annotations` draws goalkeeper ellipses without KeyError
6. Confirm ball possession loop in `main.py` no longer crashes when `tracks['ball'][frame_num]` is empty

Do NOT delete the existing stub files — the stub format will change since `tracks` now has a new `goalkeepers` key. Add a note in the code that existing stubs need to be regenerated (`read_from_stub=False` on first run after this change).

---

## What NOT to change

- `view_transformer.py` — untouched
- `camara_movement_estimator.py` — untouched
- `speed_and_distance_estimator.py` — untouched
- `utils/` — untouched
- Do not change the ByteTrack configuration
- Do not add new dependencies
