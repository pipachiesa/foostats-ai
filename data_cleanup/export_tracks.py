"""Export foostats_ai pipeline tracks to a cleanup-ready tracking CSV.

This writer is intentionally decoupled from the detection/tracking pipeline in
``main.py``. It consumes the ``all_tracks`` dict that ``main()`` already builds
(keys: ``players``, ``goalkeepers``, ``referees``, ``ball`` — each a per-frame
list of ``{track_id: {"bbox": [...], "position_transformed": [x, y], ...}}``)
and emits the schema consumed by ``data_cleanup.lib.match.Match.import_raw_data``:

    Frame, Object, Object ID, Team, X1, Y1, X2, Y2, X_Pitch, Y_Pitch, X_MPLSoccer, Y_MPLSoccer

``X_Pitch`` / ``Y_Pitch`` come from the view transformer's ``position_transformed``
(field meters). ``X_MPLSoccer`` / ``Y_MPLSoccer`` are those values normalized to
[0, 1] by the standard pitch size, ready to drop into mplsoccer.
"""

import csv

# Keep these in sync with data_cleanup.lib.match. No SoccerPitchConfiguration
# exists in the project yet, so we hardcode the FIFA-standard pitch.
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def _row(frame_number, object_name, object_id, team, bbox, position):
    if bbox is None:
        bbox = [0, 0, 0, 0]
    x1, y1, x2, y2 = bbox

    if position is None:
        x_pitch, y_pitch = 0, 0
        x_mpl, y_mpl = 0, 0
    else:
        x_pitch, y_pitch = position[0], position[1]
        x_mpl = x_pitch / PITCH_LENGTH
        y_mpl = y_pitch / PITCH_WIDTH

    return [frame_number, object_name, object_id, team,
            x1, y1, x2, y2, x_pitch, y_pitch, x_mpl, y_mpl]


def tracks_to_csv(all_tracks, output_path):
    """Write ``all_tracks`` to ``output_path`` as a tracking CSV.

    Frame numbers are 1-indexed to match the cleanup library's expectations.
    Goalkeepers are written as ``player`` rows (they are players with a team for
    downstream analysis); referees are skipped (no team / not used by Match).
    """
    header = ["Frame", "Object", "Object ID", "Team", "X1", "Y1", "X2", "Y2",
              "X_Pitch", "Y_Pitch", "X_MPLSoccer", "Y_MPLSoccer"]

    players = all_tracks.get("players", [])
    goalkeepers = all_tracks.get("goalkeepers", [])
    ball = all_tracks.get("ball", [])
    num_frames = max(len(players), len(goalkeepers), len(ball))

    with open(output_path, "w", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for frame_idx in range(num_frames):
            frame_number = frame_idx + 1

            for object_name, source in (("player", players), ("player", goalkeepers)):
                if frame_idx >= len(source):
                    continue
                for track_id, info in source[frame_idx].items():
                    writer.writerow(_row(
                        frame_number, object_name, track_id,
                        info.get("team", 0),
                        info.get("bbox"),
                        info.get("position_transformed"),
                    ))

            if frame_idx < len(ball):
                for _, info in ball[frame_idx].items():
                    writer.writerow(_row(
                        frame_number, "ball", 0, 0,
                        info.get("bbox"),
                        info.get("position_transformed"),
                    ))

    return output_path
