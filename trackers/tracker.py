from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import torch
import sys
from pathlib import Path
from boxmot import BotSort
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    POSSESSION_WINDOW_FRAMES = 150  # smooth the overlay over ~6 seconds instead of raw frame-to-frame swings

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # BoT-SORT with built-in camera motion compensation (CMC) to reduce
        # ID drift during broadcast camera pans.
        self.tracker = BotSort(
            reid_weights=None,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            half=False,
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            track_buffer=60,
            match_thresh=0.8,
            frame_rate=25,
        )
        print(f"  Tracker initialized: {type(self.tracker).__name__}")

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]["position"] = position
                        

    def interpolate_ball_positions(self, ball_positions):
        ball_positions_list = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df = pd.DataFrame(ball_positions_list, columns=["x1", "y1", "x2", "y2"])

        # Track which rows were originally missing
        missing_mask = df["x1"].isna()

        # Only interpolate gaps of <= 15 consecutive missing frames
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
        

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch  
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        # NOTE: existing stubs (tracks_stub.pkl) must be regenerated after this change
        # because the tracks dict now includes a "goalkeepers" key.
        # Run with read_from_stub=False on first run after this change.
        tracks={
            "players": [],
            "referees": [],
            "ball": [],
            "goalkeepers": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k, v in cls_names.items()}

            # Convert YOLO detections to Supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            ball_class_id = cls_names_inv["ball"]
            ball_mask = detection_supervision.class_id == ball_class_id

            # --- Ball: detection-only (no tracker) ---
            ball_bbox = None
            if np.any(ball_mask):
                ball_detections = detection_supervision[ball_mask]
                ball_detections = ball_detections[ball_detections.confidence > 0.5]
                if len(ball_detections) > 1:
                    best_idx = ball_detections.confidence.argmax()
                    ball_detections = ball_detections[best_idx:best_idx+1]
                if len(ball_detections) > 0:
                    ball_bbox = ball_detections.xyxy[0].tolist()

            # --- Non-ball: track with BoT-SORT ---
            non_ball = detection_supervision[~ball_mask]
            if len(non_ball) > 0:
                # boxmot expects [x1, y1, x2, y2, conf, class_id]
                dets_np = np.column_stack([
                    non_ball.xyxy,
                    non_ball.confidence,
                    non_ball.class_id.astype(float),
                ])
            else:
                dets_np = np.empty((0, 6))

            # Pass the actual frame for camera motion compensation
            tracked = self.tracker.update(dets_np, frames[frame_num])

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            tracks["goalkeepers"].append({})

            # Parse BoT-SORT output: [x1, y1, x2, y2, track_id, conf, class_id, ...]
            for trk in tracked:
                bbox = trk[0:4].tolist()
                track_id = int(trk[4])
                cls_id = int(trk[6]) if len(trk) > 6 else -1

                if cls_id == cls_names_inv.get("player", -1):
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv.get("referee", -1):
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv.get("goalkeeper", -1):
                    tracks["goalkeepers"][frame_num][track_id] = {"bbox": bbox}

            if ball_bbox is not None:
                tracks["ball"][frame_num][1] = {"bbox": ball_bbox}



        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

    
        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=225,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
            (int(x1_rect), int(y1_rect)),
            (int(x2_rect), int(y2_rect)),
            color,
            cv2.FILLED
            )

            x1_text = x1_rect + 12

            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2)
        
        return frame
    
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x -10, y - 20],
            [x + 10, y - 20]
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control, frame_offset=0):
        # Draw a semi transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Use a rolling window so short turnovers do not dominate the on-screen percentage.
        current_idx = frame_offset + frame_num + 1
        window_start = max(0, current_idx - self.POSSESSION_WINDOW_FRAMES)
        team_ball_control_till_frame = team_ball_control[window_start:current_idx]

        # Exclude frames with no possession so they do not count for either team.
        valid = team_ball_control_till_frame[team_ball_control_till_frame > 0]
        team_1_num_frames = (valid == 1).sum()
        team_2_num_frames = (valid == 2).sum()
        total_frames = len(valid)
        team_1 = team_1_num_frames / total_frames if total_frames > 0 else 0
        team_2 = team_2_num_frames / total_frames if total_frames > 0 else 0
        
        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        
        return frame
        
   
    def draw_annotations(self, video_frames, tracks, team_ball_control, frame_offset=0):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            goalkeeper_dict = tracks["goalkeepers"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw players

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (0,0,255))

            # Draw goalkeepers

            for track_id, goalkeeper in goalkeeper_dict.items():
                color = goalkeeper.get("team_color", (0, 255, 255))
                frame = self.draw_ellipse(frame, goalkeeper["bbox"], color, track_id)

            # Draw referees

            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255), track_id)

            # draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))

            # Draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control, frame_offset)

            output_video_frames.append(frame)

        return output_video_frames
