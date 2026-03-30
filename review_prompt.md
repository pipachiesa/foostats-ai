# Code Review & Fix Prompt — foostats_ai

You are reviewing a football (soccer) computer vision analytics system built with Python, YOLO, and OpenCV. Your job is to find and fix **small, concrete bugs and imperfections** across the codebase. Do NOT add new features. Do NOT refactor architecture. Fix only what's clearly wrong or fragile.

## Project structure

```
main.py
trackers/tracker.py
teams_assigner/team_assigner.py
camara_movement_estimator/camara_movement_estimator.py
player_ball_assigner/player_ball_assigner.py
speed_and_distance_estimator/speed_and_distance_estimator.py
view_transformer/view_transformer.py
utils/bbox_utils.py
utils/video_utils.py
```

---

## Known issues to fix

### 1. `main.py` — IndexError risk on first frame

```python
# Line ~74
team_ball_control.append(team_ball_control[-1])  # crashes if list is empty on frame 0
```
Fix: add a fallback for when `team_ball_control` is still empty (e.g. default to team 1 or skip).

---

### 2. `trackers/tracker.py` — Debug print left in production

```python
# Line ~108
print(detection_with_tracks)  # should not be in production code
```
Remove it.

Also fix: **division by zero** in `draw_team_ball_control` when `team_1_num_frames + team_2_num_frames == 0`:
```python
team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)  # ZeroDivisionError
```

---

### 3. `camara_movement_estimator/camara_movement_estimator.py` — Double import + redundant assignment

```python
# Lines 1 and 8 — measure_distance is imported twice
from utils import measure_distance
...
from utils import measure_distance, measure_xy_distance
```
Remove the first import (line 1), keep only the complete one.

Also fix: **redundant line** in `add_adjust_positions_to_tracks`:
```python
tracks[object][frame_num][track_id]['position'] = position  # already set upstream, useless
```
Remove it.

Also fix: **"Camara"** is a typo. In the UI text drawn on frame:
```python
cv2.putText(frame, f"Camara Movement X: ...")
cv2.putText(frame, f"Camara Movement Y: ...")
```
Change to `"Camera Movement X"` and `"Camera Movement Y"`.

---

### 4. `teams_assigner/team_assigner.py` — Magic number hardcoded hack

```python
# Line ~72
if player_id == 111:
    team_id = 1
```
This is a hardcoded fix for a specific player in a specific video. Remove it or replace with a comment explaining it's a temporary override, and wrap it in a configurable override dict (e.g. `self.player_team_overrides = {}`) that can be set externally. Default it to empty so it doesn't affect other videos.

Also fix: `get_clustering_model` calls KMeans without `n_init`:
```python
kmeans = KMeans(n_clusters=2, init='k-means++').fit(image_2d)
```
Add `n_init=10` to suppress sklearn deprecation warning and ensure deterministic behavior.

---

### 5. `speed_and_distance_estimator/speed_and_distance_estimator.py` — Hardcoded FPS + unused return value

`frame_rate = 24` is hardcoded in `__init__`. The FPS is already read from the video in `main.py`. Fix `SpeedAndDistanceEstimator.__init__` to accept `frame_rate` as a parameter with default 24:
```python
def __init__(self, frame_rate=24):
    self.frame_rate = frame_rate
```
Then in `main.py`, pass the actual fps:
```python
speed_and_distance_estimator = SpeedAndDistanceEstimator(frame_rate=fps)
```

Also fix: `draw_speed_and_distance` returns `output_frames` but in `main.py` the return value is ignored:
```python
speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)  # return ignored
```
The method modifies frames in-place AND returns them — pick one. Since all other draw methods return the frames, make this consistent: ensure `main.py` captures the return value, or remove the return and document in-place mutation. Prefer keeping the return pattern consistent with the rest of the pipeline.

---

### 6. `player_ball_assigner/player_ball_assigner.py` — Magic number for infinity

```python
minimum_distance = 99999
```
Replace with:
```python
minimum_distance = float('inf')
```

---

### 7. All modules — inconsistent `sys.path` manipulation

Some modules use `sys.path.append("../")` (fragile, relative path):
- `camara_movement_estimator/camara_movement_estimator.py`
- `speed_and_distance_estimator/speed_and_distance_estimator.py`
- `player_ball_assigner/player_ball_assigner.py`

`tracker.py` already uses the correct pattern:
```python
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
```
Apply this same pattern to all modules that still use `sys.path.append("../")`.

---

### 8. Trailing whitespace and dead blank lines

Multiple files have excessive blank lines at the end (e.g. `camara_movement_estimator.py` ends with ~10 blank lines, `speed_and_distance_estimator.py` similarly). Clean up trailing whitespace and extra blank lines at end of all files.

---

## What NOT to do

- Do not change the pipeline architecture
- Do not add logging libraries or new dependencies
- Do not modify the stub/pickle caching logic
- Do not change the ViewTransformer pixel vertices (those are video-specific calibration values)
- Do not rename classes or public method signatures (other code may depend on them)

## Verification

After making changes:
1. Confirm `main.py` can be imported without error
2. Confirm no `print(detection_with_tracks)` remains anywhere in the codebase
3. Confirm no `sys.path.append("../")` remains in any module
4. Confirm `draw_speed_and_distance` return value is properly used in `main.py`
5. Grep for "Camara" and confirm it no longer appears in any UI string
