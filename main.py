from utils import read_video, save_video, frame_stream, get_video_info
from trackers import Tracker
import supervision as sv
import cv2
import numpy as np
from teams_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camara_movement_estimator import CamaraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator
from event_detector import PassDetector
import json


def main():
    VIDEO_PATH = 'input_videos/08fd33_4.mp4'
    MODEL_PATH = 'models/best.pt'
    OUTPUT_PATH = 'output_videos/output_video.avi'
    BATCH_SIZE = 500
    OVERLAP = 30

    fps, total_frames, frame_width, frame_height = get_video_info(VIDEO_PATH)
    print(f"Video: {total_frames} frames at {fps:.1f} fps ({total_frames/fps/60:.1f} min)")

    # --- One-time setup: calibrate on FIRST FRAME ONLY ---
    # Solo necesitamos 1 frame para calibrar colores de equipo, no 500.
    print("Calibrating team colors on first frame...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    _, first_frame = cap.read()
    cap.release()

    tracker = Tracker(MODEL_PATH)

    # Detectar solo 1 frame para obtener jugadores del primer frame
    first_tracks = tracker.get_object_tracks([first_frame], read_from_stub=False)

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(first_frame, first_tracks['players'][0])
    # Reset tracker with same tuned params (must match Tracker.__init__)
    tracker.tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=60,
        minimum_matching_threshold=0.8,
        frame_rate=25
    )
    team_assigner.player_team_dict = {} # limpiar IDs del frame de calibración
    print("Team colors calibrated.")

    # Verify team clustering isn't collapsed to one cluster
    # Re-check using first frame players after calibration
    first_tracks_check = tracker.get_object_tracks([first_frame], read_from_stub=False)
    team_counts = {1: 0, 2: 0}
    for pid, track in first_tracks_check['players'][0].items():
        tid = team_assigner.get_player_team(first_frame, track['bbox'], pid)
        team_counts[tid] = team_counts.get(tid, 0) + 1
    print(f"Team assignment after calibration: {team_counts}")
    if team_counts[1] > 0 and team_counts[2] > 0:
        ratio = max(team_counts[1], team_counts[2]) / min(team_counts[1], team_counts[2])
        if ratio > 3:
            print(f"WARNING: Team imbalance ratio {ratio:.1f}:1 — KMeans may need re-clustering")
            # Force re-calibration with a fresh KMeans fit
            team_assigner.assign_team_color(first_frame, first_tracks_check['players'][0])
            team_assigner.player_team_dict = {}
            print("Re-calibrated team colors due to imbalance.")
    # Reset tracker again after the check so IDs start fresh for actual processing
    tracker.tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=60,
        minimum_matching_threshold=0.8,
        frame_rate=25
    )

    # --- Initialize output video writer ---
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

    # --- Persistent state across windows ---
    accumulated_distance = {}
    team_ball_control = []          # rolling list, kept in memory (1 int per frame, ~1MB for 90min)
    all_tracks = {'players': [], 'ball': [], 'goalkeepers': [], 'referees': []}
    player_assigner = PlayerBallAssigner()
    camera_movement_estimator = CamaraMovementEstimator(first_frame)
    view_transformer = ViewTransformer()
    speed_and_distance_estimator = SpeedAndDistanceEstimator(frame_rate=fps)

    # --- Process windows ---
    frames_written = 0
    try:
        for batch_idx, (window_frames, start_idx, fps_batch, fw, fh, is_last) in enumerate(
            frame_stream(VIDEO_PATH, batch_size=BATCH_SIZE, overlap=OVERLAP)
        ):
            print(f"Processing batch {batch_idx+1}: frames {start_idx}\u2013{start_idx+len(window_frames)-1}")

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

            # 7b. Goalkeeper team assignment (by field position, not color)
            window_frame_width = window_frames[0].shape[1]
            for frame_num, goalkeeper_track in enumerate(window_tracks['goalkeepers']):
                for gk_id, track in goalkeeper_track.items():
                    team = team_assigner.get_goalkeeper_team(
                        window_frames[frame_num], track['bbox'], window_frame_width
                    )
                    window_tracks['goalkeepers'][frame_num][gk_id]['team'] = team
                    window_tracks['goalkeepers'][frame_num][gk_id]['team_color'] = \
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
                    # Store explicit no-possession frames as 0 so the overlay can exclude them.
                    window_ball_control.append(0)

            team_ball_control.extend(window_ball_control)
            team_ball_control_arr = np.array(team_ball_control)

            # 8b. Accumulate tracks for pass detection (non-overlap only)
            keep = len(window_frames) if is_last else len(window_frames) - OVERLAP
            for key in all_tracks:
                all_tracks[key].extend(window_tracks[key][:keep])

            # 9. Draw annotations
            # Only render the non-overlap zone (all frames if last batch)
            frames_to_render = len(window_frames) if is_last else len(window_frames) - OVERLAP

            annotated_frames = tracker.draw_annotations(
                window_frames[:frames_to_render],
                {k: v[:frames_to_render] for k, v in window_tracks.items()},
                team_ball_control_arr,
                frame_offset=frames_written
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
            print(f"  \u2192 {frames_written} frames written total")

            # Explicitly free window memory
            del window_frames, window_tracks, annotated_frames
    finally:
        writer.release()

    # --- Pass detection ---
    print("Detecting passes...")
    pass_detector = PassDetector()
    passes = pass_detector.detect(all_tracks, team_ball_control)

    team_passes = sum(1 for p in passes if p['type'] == 'pass' and p['from_team'] == 1)
    team2_passes = sum(1 for p in passes if p['type'] == 'pass' and p['from_team'] == 2)
    interceptions = sum(1 for p in passes if p['type'] == 'interception')

    print(f"Passes detected: {len(passes)}")
    print(f"  Team 1 → Team 1: {team_passes}")
    print(f"  Team 2 → Team 2: {team2_passes}")
    print(f"  Interceptions: {interceptions}")

    passes_path = 'output_videos/passes.json'

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(passes_path, 'w') as f:
        json.dump(passes, f, indent=2, cls=_NumpyEncoder)
    print(f"Pass data saved to {passes_path}")

    # Expose final pipeline state for Colab-side debugging after main() completes.
    import main as _self
    _self._debug_all_tracks = all_tracks
    _self._debug_team_ball_control = team_ball_control

    print(f"Done. Output: {OUTPUT_PATH} ({frames_written} frames)")


if __name__ == "__main__":
    main()
