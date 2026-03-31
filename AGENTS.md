# AGENTS.md

## Project Overview

This project aims to build an **automated football (soccer) analytics system using computer vision and machine learning**.

The system processes match videos and extracts structured data about players, the ball, and match events. The long-term goal is to generate **advanced football statistics automatically from video**.

The system combines:

* Object detection (YOLO)
* Multi-object tracking
* Ball tracking
* Team classification via color clustering
* Camera motion compensation
* Perspective transformation
* Event detection
* Player metric computation (speed, distance, etc.)

The output should resemble the type of analytics produced by platforms like **StatsBomb, Opta, or Wyscout**, but derived directly from video.

---

# Current System Architecture

## 1. Object Detection

We use **YOLO models** to detect:

* Players
* Referees
* Ball

The model is trained on a **custom dataset labeled with Roboflow**.

Dataset format:

```
YOLO format

class_id x_center y_center width height
```

Classes:

```
ball
goalkeeper
player
referee
```

Model options being tested:

* YOLOv5
* YOLOv8

Dataset size currently ~600 images.

Goal is to expand dataset using:

* frame extraction from match videos
* auto-labeling with the current detector
* manual correction.

---

## 2. Object Tracking

After detection, objects are tracked across frames using a tracking algorithm.

Possible trackers:

* ByteTrack
* BoT-SORT
* DeepSORT

Tracking allows us to maintain **persistent player IDs across frames**.

Tracking outputs:

```
player_id
bounding_box
frame_number
```

---

## 3. Team Classification

Players are assigned to teams using **shirt color clustering**.

Process:

1. Crop player bounding box
2. Extract shirt region
3. Convert to color space (HSV or RGB)
4. Use **KMeans clustering** to determine dominant color
5. Assign cluster to team A or B.

---

## 4. Ball Tracking

Ball detection is difficult because:

* the ball is small
* motion blur occurs
* frequent occlusion

The system tracks the ball using:

* YOLO detection
* tracking
* interpolation when the ball disappears

Improving ball detection is a major focus.

---

## 5. Camera Motion Compensation

Football broadcast cameras move continuously.

To correctly measure player movement we estimate **camera motion between frames**.

Methods being explored:

* Optical flow
* Feature matching (ORB / SIFT)
* Homography estimation

This allows stabilization of the coordinate system.

---

## 6. Perspective Transformation

A **homography transformation** converts image coordinates into approximate field coordinates.

This allows movement to be measured in **meters instead of pixels**.

Outputs include:

```
player_position_field_x
player_position_field_y
```

---

## 7. Player Metrics

Once player coordinates are known we compute:

* distance covered
* player speed
* acceleration
* heatmaps
* movement trajectories

---

# Long-Term Goal: Event Detection

The next stage is detecting **football events automatically**.

Examples:

* Passes
* Shots
* Ball recoveries
* Duels
* Possession changes

These events require reasoning over **temporal sequences**, not just frame-level detections.

Event detection will rely on:

* ball trajectory
* player-ball distance
* player velocity
* possession estimation
* spatial zones on the pitch.

---

# Event Detection Strategy

Events are not detected directly by YOLO.

Instead they are inferred using:

1. **Tracking data**
2. **Ball trajectory**
3. **Possession estimation**
4. **Temporal models or rule systems**

Example pass detection logic:

```
player A has ball
ball moves > threshold distance
player B gains possession
```

Example shot detection:

```
player possession
ball velocity spike
ball trajectory towards goal
```

---

# Dataset Expansion Strategy

The system needs more labeled data.

Approach:

1. Collect match videos.
2. Extract frames using ffmpeg.
3. Use current YOLO model for auto-labeling.
4. Manually correct labels.
5. Retrain detector.

Focus on improving:

* ball detection
* occlusion cases
* motion blur frames.

---

# What Codex Should Help With

Codex should assist with:

### System Design

* designing modular computer vision pipelines
* selecting tracking algorithms
* structuring data flow

### Machine Learning

* improving YOLO training
* dataset expansion strategies
* model evaluation

### Event Detection

* designing event classification logic
* suggesting temporal models
* improving possession estimation

### Performance

* optimizing inference speed
* scaling to full-match processing
* improving tracking stability

### Research Guidance

* suggesting relevant academic papers
* recommending state-of-the-art approaches

---

# Constraints

* The system must run on **standard GPUs (Colab / local GPU)**.
* Training datasets are initially **small**.
* Many modules will start with **heuristics before ML models**.

---

# Development Philosophy

The system will evolve iteratively:

1. Build simple working pipeline.
2. Validate on small video clips.
3. Improve detection/tracking.
4. Introduce event detection.
5. Scale to full match analysis.

Focus is on **practical results and incremental improvement**, not perfect models from the start.

---

# End Goal

Produce a system capable of generating:

```
Player statistics
Pass maps
Shot maps
Distance covered
Speed metrics
Possession statistics
Event timelines
```

from raw football match video automatically.