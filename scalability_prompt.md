# Scalability Refactor Prompt — foostats_ai

You are a backend engineer refactoring a football analytics pipeline from **all-in-memory** to **streaming/windowed processing**, so it can handle full 90-minute matches without running out of RAM.

Do NOT add new features. Do NOT change detection or tracking logic. Only change the data flow architecture.

---

## The problem

`utils/video_utils.py` → `read_video()` loads **all frames** into a Python list before processing begins.

A 90-minute match at 1080p/25fps ≈ 135,000 frames ≈ 800GB uncompressed in RAM. It crashes.

The full in-memory pipeline looks like:

```
read_video() → [135k frames in RAM]
    → get_object_tracks([135k frames]) → tracks dict [135k entries in RAM]
    → camera_movement([135k frames]) → [135k entries in RAM]
    → transform all tracks [135k entries in RAM]
    → interpolate ball [135k entries]
    → speed/distance [135k entries]
    → team assign loop [135k iterations]
    → ball assign loop [135k iterations]
    → draw_annotations([135k frames]) → [135k output frames in RAM]
    → save_video([135k frames]) → writes to disk
```

Everything needs to move from "process all then render" to "process a window, render it, discard it, move to next window."

---

## Target architecture

Process the video in **overlapping windows of 500 frames** with **30-frame overlap** between consecutive windows. The overlap handles:
- ByteTrack ID continuity (tracker stays stateful, no issue)
- Ball interpolation across window boundaries (gaps ≤ 15 frames won't span a 30-frame overlap)
- Speed/distance continuity (accumulated totals carry forward as state)

Output frames are written to `cv2.VideoWriter` immediately — never accumulated as a list.

```
for each window of 500 frames (with 30-frame overlap):
    detect + track window
    estimate camera motion for window
    transform positions for window
    interpolate ball for window
    compute speed/distance for window (with accumulated state)
    assign teams for window (reuse calibration from frame 0)
    assign ball possession for window
    draw + write frames 0..469 to VideoWriter (skip last 30 = overlap zone)

after last window: write remaining overlap frames
```

---

## Changes required

### 1. `utils/video_utils.py` — add frame stream generator

Keep `read_video()` and `save_video()` as-is for backward compat (used in dev/testing on short clips).

Add a new function:

```python
def frame_stream(video_path, batch_size=500, overlap=30):
    """
    Generator that yields (frames, start_frame_idx, is_last_batch) tuples.
    Each batch has up to batch_size frames.
    Consecutive batches overlap by `overlap` frames to handle boundary effects.

    Yields:
        frames: list of numpy arrays (the window)
        start_idx: absolute frame index of frames[0] in the original video
        is_last: bool, True if this is the final batch
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buffer = []
    global_idx = 0
    batch_start_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        buffer.append(frame)
        global_idx += 1

        if len(buffer) == batch_size:
            is_last = (global_idx >= total_frames)
            yield buffer.copy(), batch_start_idx, fps, frame_width, frame_height, is_last
            # Keep last `overlap` frames as start of next batch
            buffer = buffer[-overlap:]
            batch_start_idx = global_idx - overlap
            if is_last:
                break

    # Yield remaining frames if any
    if buffer:
        yield buffer, batch_start_idx, fps, frame_width, frame_height, True

    cap.release()


def get_video_info(video_path):
    """Return (fps, total_frames, width, height) without loading frames."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total_frames, width, height
```

---

### 2. `speed_and_distance_estimator.py` — accept and return accumulated distance state

Currently `add_speed_and_distance_to_tracks` accumulates `total_distance` internally and discards it when the method returns. Across windows, each player's accumulated distance is lost.

Change the signature to accept and return the state:

```python
def add_speed_and_distance_to_tracks(self, tracks, accumulated_distance=None):
    """
    accumulated_distance: dict of {object: {track_id: float}} from previous window.
    If None, starts fresh (first window or short-clip mode).
    Returns updated accumulated_distance dict for passing to next window.
    """
    if accumulated_distance is None:
        accumulated_distance = {}

    total_distance = accumulated_distance  # mutate in place

    # ... existing logic unchanged, using total_distance instead of local var ...

    return total_distance
```

Keep the old behavior (no argument = starts fresh) to avoid breaking existing short-clip usage.

---

### 3. `main.py` — restructure into streaming pipeline

Replace the entire `main()` function. Keep the module imports unchanged.

New structure:

```python
def main():
    VIDEO_PATH = 'input_videos/08fd33_4.mp4'
    MODEL_PATH = 'models/best.pt'
    OUTPUT_PATH = 'output_videos/output_video.avi'
    BATCH_SIZE = 500
    OVERLAP = 30

    from utils import frame_stream, get_video_info

    fps, total_frames, frame_width, frame_height = get_video_info(VIDEO_PATH)
    print(f"Video: {total_frames} frames at {fps:.1f} fps ({total_frames/fps/60:.1f} min)")

    # --- One-time setup: calibrate on first batch ---
    print("Calibrating on first batch...")
    first_batch = next(iter(frame_stream(VIDEO_PATH, batch_size=BATCH_SIZE, overlap=OVERLAP)))
    first_frames, _, _, _, _, _ = first_batch

    tracker = Tracker(MODEL_PATH)

    # Get first-batch tracks for team color calibration
    first_tracks = tracker.get_object_tracks(first_frames, read_from_stub=False)

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(first_frames[0], first_tracks['players'][0])
    print("Team colors calibrated.")

    # --- Initialize output video writer ---
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

    # --- Persistent state across windows ---
    accumulated_distance = {}
    team_ball_control = []          # rolling list, kept in memory (1 int per frame, ~1MB for 90min)
    player_assigner = PlayerBallAssigner()
    camera_movement_estimator = CamaraMovementEstimator(first_frames[0])
    view_transformer = ViewTransformer()
    speed_and_distance_estimator = SpeedAndDistanceEstimator(frame_rate=fps)

    # --- Process windows ---
    frames_written = 0
    for batch_idx, (window_frames, start_idx, fps_batch, fw, fh, is_last) in enumerate(
        frame_stream(VIDEO_PATH, batch_size=BATCH_SIZE, overlap=OVERLAP)
    ):
        print(f"Processing batch {batch_idx+1}: frames {start_idx}–{start_idx+len(window_frames)-1}")

        # 1. Detect + track
        window_tracks = tracker.get_object_tracks(window_frames, read_from_stub=False)

        # 2. Add positions
        tracker.add_position_to_tracks(window_tracks)

        # 3. Camera motion
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
            window_frames, read_from_stub=False
        )
        camera_movement_estimator.add_adjust_positions_to_tracks(
            window_tracks, camera_movement_per_frame
        )

        # 4. Perspective transform
        view_transformer.add_transformed_position_to_tracks(window_tracks)

        # 5. Ball interpolation
        window_tracks['ball'] = tracker.interpolate_ball_positions(window_tracks['ball'])

        # 6. Speed and distance (with carry-over state)
        accumulated_distance = speed_and_distance_estimator.add_speed_and_distance_to_tracks(
            window_tracks, accumulated_distance
        )

        # 7. Team assignment (uses calibrated kmeans, no refit)
        for frame_num, player_track in enumerate(window_tracks['players']):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(
                    window_frames[frame_num], track['bbox'], player_id
                )
                window_tracks['players'][frame_num][player_id]['team'] = team
                window_tracks['players'][frame_num][player_id]['team_color'] = \
                    team_assigner.team_colors[team]

        # 8. Ball possession
        window_ball_control = []
        for frame_num, player_track in enumerate(window_tracks['players']):
            ball_frame = window_tracks['ball'][frame_num]
            if 1 not in ball_frame:
                window_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
                continue
            ball_bbox = ball_frame[1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
            if assigned_player != -1:
                window_tracks['players'][frame_num][assigned_player]['has_ball'] = True
                window_ball_control.append(
                    window_tracks['players'][frame_num][assigned_player]['team']
                )
            else:
                window_ball_control.append(team_ball_control[-1] if team_ball_control else 1)

        team_ball_control.extend(window_ball_control)
        team_ball_control_arr = np.array(team_ball_control)

        # 9. Draw annotations
        # Only render the non-overlap zone (all frames if last batch)
        frames_to_render = len(window_frames) if is_last else len(window_frames) - OVERLAP

        annotated_frames = tracker.draw_annotations(
            window_frames[:frames_to_render],
            {k: v[:frames_to_render] for k, v in window_tracks.items()},
            team_ball_control_arr
        )
        annotated_frames = camera_movement_estimator.draw_camera_movement(
            annotated_frames, camera_movement_per_frame[:frames_to_render]
        )
        annotated_frames = speed_and_distance_estimator.draw_speed_and_distance(
            annotated_frames,
            {k: v[:frames_to_render] for k, v in window_tracks.items()}
        )

        # 10. Write frames to disk immediately, then discard
        for frame in annotated_frames:
            writer.write(frame)
        frames_written += len(annotated_frames)
        print(f"  → {frames_written} frames written total")

        # Explicitly free window memory
        del window_frames, window_tracks, annotated_frames

    writer.release()
    print(f"Done. Output: {OUTPUT_PATH} ({frames_written} frames)")


if __name__ == "__main__":
    main()
```

---

### 4. `utils/__init__.py` — export new functions

Add `frame_stream` and `get_video_info` to the exports:

```python
from .video_utils import read_video, save_video, frame_stream, get_video_info
```

---

## Important constraints

### ByteTrack state across windows
ByteTrack (`sv.ByteTrack()`) is initialized once in `Tracker.__init__` and keeps internal state. As long as the same `Tracker` instance is reused across windows (which it is in the new `main()`), track IDs remain consistent. **Do NOT re-initialize the tracker between windows.**

### Team assignment calibration
`assign_team_color()` (KMeans fit) runs **once** on the first batch. `get_player_team()` uses the cached `self.player_team_dict` — already handles this correctly. **Do NOT call `assign_team_color()` again in subsequent windows.**

### CamaraMovementEstimator first-frame reference
The estimator is initialized with `video_frames[0]` as the reference frame. This must remain the **absolute first frame of the video**, not the first frame of each window. Since we initialize it once before the loop with `first_frames[0]`, this is already correct. Do NOT re-initialize it inside the window loop.

### team_ball_control indexing in draw_team_ball_control
`draw_team_ball_control` uses `team_ball_control[:frame_num+1]` where `frame_num` is the frame index within the current annotation call. This is correct because `draw_annotations` receives `team_ball_control_arr` (the full rolling array) and the frame_num is the position within the current batch's render slice.

Wait — this is actually a subtle bug in the current design. The `frame_num` inside `draw_annotations` iterates from 0 for each batch, but `team_ball_control_arr` grows globally. The control display will be **correct** because `team_ball_control_arr[:frame_num+1]` for a local frame_num will index into the beginning of the full array, not the current window.

Fix this by passing the window's `start_idx` offset to `draw_annotations` so it can compute the correct slice:

Change `draw_team_ball_control` signature to:
```python
def draw_team_ball_control(self, frame, frame_num, team_ball_control, frame_offset=0):
    team_ball_control_till_frame = team_ball_control[:frame_offset + frame_num + 1]
    # rest unchanged
```

And in `draw_annotations`:
```python
def draw_annotations(self, video_frames, tracks, team_ball_control, frame_offset=0):
    ...
    frame = self.draw_team_ball_control(frame, frame_num, team_ball_control, frame_offset)
```

Pass `frame_offset=frames_written` (before the batch write) in `main.py`.

### Stub system
The pickle stub system (`read_from_stub=True`) is **incompatible with streaming** because it stores the entire video's tracks. In the new pipeline, always call with `read_from_stub=False`. Leave the stub logic in `Tracker.get_object_tracks` for backward compat (useful when testing on short clips), but in the new `main()` never use it.

---

## What NOT to change

- Detection logic in `tracker.py` (`detect_frames`, `get_object_tracks` internals)
- `draw_ellipse`, `draw_triangle` — untouched
- `CamaraMovementEstimator` internals
- `ViewTransformer` — untouched
- `PlayerBallAssigner` — untouched
- `TeamAssigner` color clustering — untouched
- `read_video()` and `save_video()` — keep for backward compat

---

## Verification

After implementing:

1. Run on the existing short test clip (`08fd33_4.mp4`) — output should be identical to the old pipeline.
2. Run `python -c "from utils import frame_stream, get_video_info; print('OK')"`.
3. Monitor RAM during processing: peak should be proportional to `BATCH_SIZE` (≈500 frames × 6MB ≈ 3GB), not total video length.
4. Confirm `writer.release()` is called even if an exception occurs mid-processing (wrap the window loop in try/finally).
5. Confirm output video has the correct frame count (no duplicate frames from overlap zone, no missing frames at window boundaries).
