from lib.moment import Moment
from lib.player import Player
from lib.ball import Ball
import csv
import os

# Standard pitch dimensions (meters) used to normalize pitch coordinates into
# the [0, 1] range expected by mplsoccer. foostats_ai has no SoccerPitchConfiguration
# yet, so we hardcode the FIFA-standard 105 x 68 here. If a config is added later,
# import its length/width and override these.
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


class Match():
    def __init__(self):
        self.players = []
        self.ball = Ball()
        self.current_frame = 1
        self.frames = 0
        self.source = None
        self.file_path = None
        self.file_name = None

    def import_metrica(self, file_path, file_name):
        self.source = "metrica"
        headers = []
        frames = []
        self.file_path = file_path
        self.file_name = file_name
        path = file_path + file_name
        with open(path, "r") as file:
            count = 0
            for line in file:
                if count < 3:
                    headers.append(line.strip().split(","))
                else:
                    frames.append(line.strip().split(","))
                count += 1

        team_row = headers[0]
        player_id_row = headers[1]
        player_row = headers[2]

        players = []
        ball_idx = 0
        for i in range(3, len(headers[0]), 2):
            if(player_row[i] == "Ball"):
                ball_idx = i
            else:
                players.append(Player({
                    "team" : team_row[i],
                    "id" : player_id_row[i],
                    "name" : player_row[i],
                }))

        self.frames = len(frames)
        for frame in frames:
            player_count = 0
            for i in range(3, len(frame), 2):
                frame_data = {
                    "frame" : frame[1],
                    "time" : frame[2]
                }
                if frame[i] == "0" or frame[i+1] == "0":
                    frame_data["empty"] = True
                else:
                    frame_data["x"] = frame[i]
                    frame_data["y"] = frame[i+1]

                if(i == ball_idx):
                    self.ball.source = "metrica"
                    self.ball.add_frame(frame_data)
                else:
                    players[player_count].source = "metrica"
                    players[player_count].add_frame(frame_data)
                    player_count += 1
        self.players = players

    def import_raw_data(self, file_path, file_name, fps=25):
        """Import a foostats_ai tracking CSV.

        Expected header (extra trailing columns such as X_MPLSoccer / Y_MPLSoccer
        are tolerated and ignored here):

            Frame, Object, Object ID, Team, X1, Y1, X2, Y2, X_Pitch, Y_Pitch, ...

        Columns are resolved by header name so the importer is robust to column
        reordering and to the legacy duplicate-"X2" header bug in the original
        upstream export. If the header cannot be mapped, we fall back to the
        original fixed-index layout.

        `fps` is the source video frame rate, used to derive per-frame timestamps.
        Defaults to 25 (the pipeline's tracker frame rate); pass the real fps if
        the footage differs.
        """
        self.file_path = file_path
        self.file_name = file_name
        path = file_path + file_name
        self.source = "raw"
        frames = []
        with open(path, "r") as file:
            for line in file:
                frames.append(line.strip().split(","))

        header = frames[0]
        col = self._resolve_columns(header)
        frames = frames[1:]

        players = {}
        for frame in frames:
            if len(frame) <= col["y_pitch"]:
                continue
            frame_number = int(frame[col["frame"]])
            obj = frame[col["object"]]

            frame_data = {
                "frame" : frame_number,
                "time" : frame_number / fps
            }

            x_pitch = frame[col["x_pitch"]]
            y_pitch = frame[col["y_pitch"]]
            if x_pitch == "0" or y_pitch == "0" or x_pitch == "" or y_pitch == "":
                frame_data["empty"] = True
            else:
                frame_data["frame_x1"] = frame[col["x1"]]
                frame_data["frame_y1"] = frame[col["y1"]]
                frame_data["frame_x2"] = frame[col["x2"]]
                frame_data["frame_y2"] = frame[col["y2"]]
                frame_data["x"] = x_pitch
                frame_data["y"] = y_pitch

            if(obj == "ball"):
                self.ball.source = "raw"
                self.ball.add_frame(frame_data)
                self.frames = self.frames + 1
            else:
                player_id = frame[col["object_id"]]
                team = float(frame[col["team"]])
                if( str(player_id) not in players):
                    players[str(player_id)] = Player({
                        "team" : team,
                        "id" : player_id,
                        "name" : "Player " + str(player_id),
                    })

                player = players[str(player_id)]
                player.source = "raw"
                player.add_frame(frame_data)

        self.players = []
        for player_id in players:
            player = players[player_id]
            self.players.append(player)

    def _resolve_columns(self, header):
        """Map our CSV header names to column indices, with a fixed-index fallback."""
        aliases = {
            "frame":     ["frame"],
            "object":    ["object"],
            "object_id": ["object id", "object_id", "id"],
            "team":      ["team"],
            "x1":        ["x1"],
            "y1":        ["y1"],
            "x2":        ["x2"],
            "y2":        ["y2"],
            "x_pitch":   ["x_pitch", "x pitch"],
            "y_pitch":   ["y_pitch", "y pitch"],
        }
        normalized = [h.strip().lower() for h in header]
        col = {}
        for key, names in aliases.items():
            idx = None
            for name in names:
                if name in normalized:
                    idx = normalized.index(name)
                    break
            col[key] = idx

        # y2 frequently collides with the upstream duplicate "x2" header bug; if
        # it failed to resolve but x2 did, y2 is the column right after x2.
        if col["y2"] is None and col["x2"] is not None:
            col["y2"] = col["x2"] + 1

        if any(v is None for v in col.values()):
            # Fall back to the original fixed layout:
            # Frame, Object, Object ID, Team, X1, Y1, X2, Y2, X_Pitch, Y_Pitch
            col = {
                "frame": 0, "object": 1, "object_id": 2, "team": 3,
                "x1": 4, "y1": 5, "x2": 6, "y2": 7, "x_pitch": 8, "y_pitch": 9,
            }
        return col

    def player(self, player_id):
        for player in self.players:
            if(str(player_id) == str(player.id)):
                return player

        return None

    def add_player(self, player):
        self.players.append(player)

    def remove_player(self, player):
        for i in range(len(self.players)):
            if str(self.players[i].id) == str(player.id):
                return self.players.pop(i)

    def team(self, team):
        players = []
        for player in self.players:
            if(player.team == team):
                players.append(player)
        return players

    def merge_players(self, player1, player2):
        new_player = Player({
            "id" : player1.id,
            "team" : player1.team,
            "name" : player1.name,
        })

        for frame_number in range(1, self.frames+1):
            player1_frame = player1.frame(frame_number)
            player2_frame = player2.frame(frame_number)
            if(player1_frame):
                new_player.add_frame(player1_frame)
            elif(player2_frame):
                new_player.add_frame(player2_frame)

        self.remove_player(player1)
        self.remove_player(player2)
        self.add_player(new_player)

        return new_player

    def frame(self, frame_number):
        if(frame_number > self.frames):
            return None

        ball_frame = self.ball.frame(frame_number)
        ball = {
            "object" : self.ball,
            "type" : "Ball",
            "frame" : ball_frame
        }

        players = []
        for player in self.players:
            players.append({
                "object" : player,
                "type" : "Player",
                "frame" : player.frame(frame_number)
            })

        return Moment({
            "ball" : ball,
            "players" : players,
            "frame" : ball_frame.frame,
            "time" : ball_frame.time
        })

    def has_next_frame(self):
        moment = self.frame(self.current_frame + 1)
        if moment == None:
            return False
        return True

    def next_frame(self):
        self.current_frame = self.current_frame + 1
        return self.frame(self.current_frame)

    def first_frame(self):
        self.current_frame = 1
        return self.frame(self.current_frame)

    def export(self):
        data = []
        if(self.source == "metrica"):
            data = self.__export_metrica()
        elif(self.source == "raw"):
            data = self.__export_raw()

        os.makedirs('./output', exist_ok=True)
        csv_writer = csv.writer(open('./output/' + self.file_name, 'w', newline='\n'))
        csv_writer.writerows(data)

    def __export_metrica(self):
        header = []
        header_row1 = ["","",""]
        header_row2 = ["","",""]
        header_row3 = ["Period","Frame","Time [s]"]
        for player in self.players:
            header_row1.append(player.team)
            header_row1.append("")
            header_row2.append(player.id)
            header_row2.append("")
            header_row3.append(player.name)
            header_row3.append("")
        header_row1.append("")
        header_row2.append("")
        header_row3.append("Ball")
        header_row3.append("")
        header.append(header_row1)
        header.append(header_row2)
        header.append(header_row3)

        data = []
        ball = self.ball
        for frame_number in range(1, self.frames + 1):
            row = []
            row.append(1)
            row.append(frame_number)
            row.append(frame_number * .04)
            for player in self.players:
                frame = player.frame(frame_number)
                if(frame):
                    coordinates = frame.coordinates
                    if(coordinates):
                        row.append(coordinates[0])
                        row.append(coordinates[1])
                    else:
                        row.append(0)
                        row.append(0)
                else:
                    row.append(0)
                    row.append(0)

            coordinates = ball.frame(frame_number).coordinates
            if(coordinates):
                row.append(coordinates[0])
                row.append(coordinates[1])
            else:
                row.append(0)
                row.append(0)
            data.append(row)

        return header + data

    def __export_raw(self):
        header = [[ "Frame", "Object", "Object ID", "Team", "X1", "Y1", "X2", "Y2",
                    "X_Pitch", "Y_Pitch", "X_MPLSoccer", "Y_MPLSoccer" ]]
        data = []

        def player_row(moment, player_entry):
            frame = player_entry["frame"]
            player = player_entry["object"]
            return [moment.frame, "player", player.id, player.team,
                    frame.frame_x1, frame.frame_y1, frame.frame_x2, frame.frame_y2,
                    frame.x, frame.y] + self._mplsoccer(frame.x, frame.y)

        def ball_row(moment, ball):
            return [moment.frame, "ball", 0, 0,
                    ball.frame_x1, ball.frame_y1, ball.frame_x2, ball.frame_y2,
                    ball.x, ball.y] + self._mplsoccer(ball.x, ball.y)

        moment = self.first_frame()
        ball = moment.ball["frame"]
        for player_entry in moment.players:
            if(player_entry["frame"]):
                data.append(player_row(moment, player_entry))
        data.append(ball_row(moment, ball))

        while(self.has_next_frame()):
            moment = self.next_frame()
            ball = moment.ball["frame"]
            for player_entry in moment.players:
                if(player_entry["frame"]):
                    data.append(player_row(moment, player_entry))
            data.append(ball_row(moment, ball))

        return header + data

    @staticmethod
    def _mplsoccer(x_pitch, y_pitch):
        """Normalize pitch coordinates into mplsoccer's [0, 1] space."""
        if x_pitch is None or y_pitch is None:
            return [0, 0]
        return [float(x_pitch) / PITCH_LENGTH, float(y_pitch) / PITCH_WIDTH]
