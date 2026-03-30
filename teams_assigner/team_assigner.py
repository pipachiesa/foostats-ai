from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        # Backward-compatible alias in case other code still references this name.
        self.assign_team_colors = self.team_colors
        self.player_team_dict = {}
        self.player_team_overrides = {}

    def get_clustering_model(self, image):
        # Reshape the image into a 2d array
        image_2d = image.reshape(-1, 3)

        # Perform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10).fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        top_half_image = image[:image.shape[0]//2, :]

        # Get the clustering model

        kmeans = self.get_clustering_model(top_half_image)

        # Get cluster labels for each pixel

        labels = kmeans.labels_

        # Reshape the labels back to the original image shape

        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Get the player cluster

        corner_clusters = [clustered_image[0,0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):

        player_colors = []

        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10).fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        if player_id in self.player_team_overrides:
            team_id = self.player_team_overrides[player_id]

        self.player_team_dict[player_id] = team_id

        return team_id

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
