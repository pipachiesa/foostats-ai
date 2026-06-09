import sys
from pathlib import Path

import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer as SportsViewTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Default location of the pitch keypoint detection model (YOLOv8 keypoint, 32 vertices
# matching SoccerPitchConfiguration). See the FileNotFoundError below for how to obtain it.
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "pitch_keypoints.pt"

# Per-keypoint confidence required to use a detected vertex in the homography.
KEYPOINT_CONFIDENCE = 0.5
# A homography needs at least 4 point correspondences.
MIN_KEYPOINTS = 4
# Target field in meters, CENTER origin: x in [-52.5, 52.5], y in [-34, 34].
# SoccerPitchConfiguration is corner-origin and ships a 120x70 (cm-scale) template,
# so dividing its vertices by 100 alone yields wrong-origin [0,120]x[0,70] coords.
# We instead normalize the template vertices to [0,1] and remap them onto a standard
# 105x68 m field centered at (0,0). This keeps position_transformed consistent with
# pass_detector geometry (HALF_X=52.5, posts +/-3.66) and data_cleanup's /105, /68.
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
# Reject transformed points that land well outside the field. A homography can
# extrapolate a point near the image edge (e.g. a mis-detected ball) to wild
# coordinates; those are artifacts, not real positions. A few meters of slack
# keeps genuine near-boundary and just-out-of-play positions (needed for
# out-of-bounds / corner / goal detection) while dropping gross outliers.
FIELD_MARGIN_M = 6.0
_X_LIMIT = PITCH_LENGTH_M / 2 + FIELD_MARGIN_M
_Y_LIMIT = PITCH_WIDTH_M / 2 + FIELD_MARGIN_M


_MODEL_MISSING_MSG = (
    "Pitch keypoint model not found at: {path}\n"
    "PitchViewTransformer needs a YOLOv8 keypoint model that detects the 32 pitch\n"
    "vertices of roboflow sports' SoccerPitchConfiguration.\n\n"
    "Download the weights (≈140 MB) into models/pitch_keypoints.pt, e.g.:\n\n"
    "  curl -fsSL -o models/pitch_keypoints.pt \\\n"
    "    https://huggingface.co/martinjolif/yolo-football-pitch-detection/resolve/main/"
    "yolo-football-pitch-detection.pt\n\n"
    "Or pass model_path=... to PitchViewTransformer pointing at your own 32-keypoint\n"
    "pitch model. For testing without a model, use HardcodedViewTransformer instead."
)


class PitchViewTransformer():
    """Per-frame perspective transform driven by detected pitch keypoints.

    Runs a YOLO keypoint model on each frame, matches the confidently-detected
    pixel keypoints to their known real-world pitch coordinates (in meters), and
    builds a homography from those correspondences. Unlike the old hardcoded
    approach this adapts to camera motion and works on any broadcast video.

    Output is written to the same ``position_transformed`` key as before, in
    meters, so speed_and_distance_estimator and data_cleanup are unaffected.
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, detection_confidence=0.3,
                 keypoint_confidence=KEYPOINT_CONFIDENCE, min_keypoints=MIN_KEYPOINTS):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(_MODEL_MISSING_MSG.format(path=model_path))

        self.model = YOLO(str(model_path))
        self.config = SoccerPitchConfiguration()
        # Map the corner-origin template vertices onto a centered 105x68 m field
        # once up front: normalize to [0,1] by the template size, then scale and
        # shift so the field is centered at (0,0) -> x in [-52.5,52.5], y in [-34,34].
        verts = np.array(self.config.vertices, dtype=np.float32)
        norm_x = verts[:, 0] / float(self.config.length)
        norm_y = verts[:, 1] / float(self.config.width)
        self.pitch_vertices = np.column_stack([
            (norm_x - 0.5) * PITCH_LENGTH_M,
            (norm_y - 0.5) * PITCH_WIDTH_M,
        ]).astype(np.float32)
        self.detection_confidence = detection_confidence
        self.keypoint_confidence = keypoint_confidence
        self.min_keypoints = min_keypoints

    def get_transformer(self, frame):
        """Detect pitch keypoints in ``frame`` and return a fitted transformer.

        Returns a sports ViewTransformer (image pixels -> pitch meters), or None
        if fewer than ``min_keypoints`` vertices are detected confidently enough
        to solve a stable homography.
        """
        result = self.model.predict(frame, conf=self.detection_confidence, verbose=False)[0]
        key_points = sv.KeyPoints.from_ultralytics(result)

        if key_points.confidence is None or len(key_points.xy) == 0:
            return None

        # The filter mask aligns both arrays: detected pixel point i maps to
        # real-world vertex i (both models share the 32-vertex ordering).
        mask = key_points.confidence[0] > self.keypoint_confidence
        if int(mask.sum()) < self.min_keypoints:
            return None

        frame_reference_points = key_points.xy[0][mask].astype(np.float32)
        pitch_reference_points = self.pitch_vertices[mask].astype(np.float32)

        return SportsViewTransformer(
            source=frame_reference_points,
            target=pitch_reference_points,
        )

    def add_transformed_position_to_tracks(self, tracks, frames):
        """Transform every track's ``position_adjusted`` into pitch meters.

        ``frames`` is the list of frames aligned with the per-frame track lists
        (the new keypoint model needs the image to run inference). Inference is
        run once per frame and reused across all objects in that frame.
        """
        num_frames = len(frames)
        for frame_num in range(num_frames):
            transformer = self.get_transformer(frames[frame_num])

            for object, object_tracks in tracks.items():
                if frame_num >= len(object_tracks):
                    continue
                for track_id, track_info in object_tracks[frame_num].items():
                    position_transformed = None
                    position = track_info.get('position_adjusted')
                    if transformer is not None and position is not None:
                        point = np.array(position, dtype=np.float32).reshape(1, 2)
                        transformed = transformer.transform_points(point)[0]
                        # Drop gross extrapolation artifacts; keep near-field points.
                        if abs(transformed[0]) <= _X_LIMIT and abs(transformed[1]) <= _Y_LIMIT:
                            position_transformed = transformed.tolist()
                    track_info['position_transformed'] = position_transformed


class HardcodedViewTransformer():
    """Legacy transform with manually-picked pixel vertices (single video only).

    Kept as a fallback for testing when the pitch keypoint model is unavailable.
    Note its output range differs from PitchViewTransformer (a 23.32 x 68 m window
    rather than the full pitch), but the ``position_transformed`` key is the same.
    """

    def __init__(self):
        court_width = 68
        court_length = 23.32

        self.pixel_vertices = np.array([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915]
        ])

        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = int(point[0]), int(point[1])
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)

        return transform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
