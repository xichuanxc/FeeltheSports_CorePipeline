# Tennis Strike Vision Prototype (Week 1 & Week 2 Spec)

## Project Overview
Build a lightweight, real-time Python prototype on macOS (Apple Silicon M4) that tracks two tennis players simultaneously, extracts wrist kinematic metrics using MediaPipe Pose and YOLOv8, and outputs kinematic event triggers for tennis swing detection.

This prototype covers **Phase 1 (Dual-Player Tracking)** and **Phase 2 (Kinematic Extraction)**. It does NOT include audio model integration or GUI polish.

---

## Technical Stack & Dependencies

- **Language:** Python 3.10+
- **Primary Libraries:**
  - `ultralytics` (YOLOv8)
  - `mediapipe` (Pose Landmarker / Solutions API)
  - `opencv-python` (`cv2`)
  - `numpy`
  - `torch` (with MPS support enabled)

### Installation Command
```bash
pip install ultralytics mediapipe opencv-python numpy torch
```

---

## System Architecture & Pipeline Flow

```
[ Video Frame (1080p @ 60fps) ]
              │
              ▼
    [ YOLOv8n Person Detector ]  <-- Filters class_id == 0, selects top 2 bounding boxes
              │
      ┌───────┴───────┐
      ▼               ▼
 [ Player 1 Crop ] [ Player 2 Crop ]  <-- Includes 15% padding around bounding boxes
      │               │
      ▼               ▼
 [ Pose Instance 1 ] [ Pose Instance 2 ]  <-- MediaPipe Pose (VIDEO running mode)
      │               │
      └───────┬───────┘
              ▼
    [ Remap Coordinates ]  <-- Maps normalized crop keypoints back to original frame space
              │
              ▼
  [ Kinematic Metric Engine ]  <-- Computes wrist velocity, angular acceleration, peak snaps
              │
              ▼
   [ CSV / JSON Log Output ]  <-- Structured frame-by-frame kinematic metrics
```

---

## Detailed Implementation Requirements

### Module 1: `player_tracker.py` (Dual-Player Tracking)

1. **YOLO Detection:**
   - Load `yolov8n.pt` initialized on GPU/MPS if available, fallback to CPU.
   - Run inference on full camera frames to identify person bounding boxes (`class_id == 0`).
   - Filter and retain the **top 2 person detections** based on bounding box confidence and area.
   - Assign consistent spatial IDs:
     - `Player_Near`: Bounding box with lower $Y_{	ext{max}}$ (closer to bottom of frame).
     - `Player_Far`: Bounding box with higher $Y_{	ext{max}}$ (closer to top/court horizon).

2. **Crop Padding & Extraction:**
   - Expand each detected bounding box by **15% padding** on all four sides (clip to frame boundaries).
   - Crop the frame region for each player.

3. **Dual MediaPipe Pose Instances:**
   - Maintain two separate `mediapipe.solutions.pose.Pose` instances (`pose_near`, `pose_far`).
   - Set parameters:
     - `static_image_mode=False`
     - `model_complexity=1` (or `2` for high quality)
     - `min_detection_confidence=0.5`
     - `min_tracking_confidence=0.5`
   - Pass cropped regions to their respective pose instances.

4. **Coordinate Global Remapping:**
   - Transform normalized keypoint coordinates $(x_{	ext{crop}}, y_{	ext{crop}})$ back to global image coordinates $(X_{	ext{global}}, Y_{	ext{global}})$:
     $$X_{	ext{global}} = X_{	ext{bbox\_min}} + x_{	ext{crop}} \cdot W_{	ext{bbox\_padded}}$$
     $$Y_{	ext{global}} = Y_{	ext{bbox\_min}} + y_{	ext{crop}} \cdot H_{	ext{bbox\_padded}}$$

---

### Module 2: `kinematics.py` (Wrist Motion Analysis)

1. **Sliding Buffer Window:**
   - Maintain a rolling historical buffer of the past **5 frames** for both players.

2. **Landmarks Inspected:**
   - Right Wrist (Index 16), Right Elbow (Index 14), Right Shoulder (Index 12)
   - Left Wrist (Index 15), Left Elbow (Index 13), Left Shoulder (Index 11)
   - Auto-select dominant/active arm based on whichever wrist exhibits higher velocity in the current window.

3. **Metric Calculations (Per Player, Per Frame):**

   - **Linear Wrist Speed ($v_{	ext{wrist}}$):**
     $$v_t = rac{\sqrt{(X_t - X_{t-1})^2 + (Y_t - Y_{t-1})^2}}{\Delta t}$$
     *(Where $\Delta t = 1 / 	ext{FPS}$ of video source)*

   - **Forearm Vector Angle ($	heta_{	ext{forearm}}$):**
     Angle of vector $ec{V}_{	ext{elbow}	o	ext{wrist}}$ relative to horizontal.

   - **Forearm Angular Velocity ($\omega_{	ext{forearm}}$):**
     $$\omega_t = rac{|	heta_t - 	heta_{t-1}|}{\Delta t}$$

   - **Kinematic Snap / Peak Detection Trigger:**
     A potential visual swing trigger flag `is_visual_snap = True` is raised if:
     1. $v_{	ext{wrist}} > 	ext{VelocityThreshold}$ (default initial value: $1.2$ normalized units/sec).
     2. Wrist velocity drops sharply by $\ge 20\%$ within 2 frames after a local max peak (deceleration step).

---

### Module 3: `main_prototype.py` (Pipeline & Log Output)

1. **Input:** Accepts a video path (`--video_path path/to/tennis_match.mp4`).
2. **Execution Loop:**
   - Process video frame-by-frame.
   - Run Module 1 (Dual-Player Tracking) and Module 2 (Kinematics).
   - Render diagnostic overlay on output frame:
     - Bounding boxes for Player Near / Far.
     - Skeleton landmarks drawn in global coordinates.
     - HUD text showing current $v_{	ext{wrist}}$ and `SNAP DETECTED` indicator flashes in red when triggered.
3. **Data Export:**
   - Output structured CSV log file (`kinematic_log.csv`) with the following columns:
     ```csv
     frame_id, timestamp_sec, player_id, wrist_speed, forearm_angular_vel, is_visual_snap
     ```

---

## Acceptance Criteria for the AI Agent

1. **Functional Tracking:** Runs on a 1080p tennis video without throwing exceptions when players enter or temporarily leave the frame.
2. **No ID Swapping:** Correctly maintains distinct IDs (`Player_Near` and `Player_Far`) across at least 90% of contiguous rally frames.
3. **Kinematic Metrics Generated:** Successfully outputs `kinematic_log.csv` containing numerical velocity values and peak snap boolean triggers.
4. **Performance:** Reaches at least **45+ FPS** execution speed on Apple Silicon M4 without rendering delays.

---

## Example Usage Command

```bash
python main_prototype.py --video_path inputs/rally_sample.mp4 --save_overlay --output_csv logs/kinematic_log.csv
```
