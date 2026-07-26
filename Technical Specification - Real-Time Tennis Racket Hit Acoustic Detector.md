**Document Version:** 1.0

**Target Environment:** Python 3.13 / PyTorch (Phase 1) $\rightarrow$ ONNX Runtime (Deployment)
**Hardware Target:** Apple Silicon (M-Series / MPS Acceleration) & Standard x86/ARM CPUs

## 1. System Overview & Core Objectives

This specification defines the architecture, dataset schema, and inference pipeline for a real-time, lightweight acoustic recognition system designed to detect tennis racket strikes from court video/audio feeds.

- **Primary Function:** Continuous monitoring of mono court audio to identify $T_0$ impact moments of tennis racket hits.
- **Input Stream:** 16 kHz Mono PCM Audio.
- **Target Latency:** End-to-end window evaluation in $< 1.0\text{ ms}$.
- **Deployment Model:** Two-stage workflow using **PyTorch (`.pth`)** during training/experimentation and **ONNX (`.onnx`)** for real-time app integration.

## 2. Dataset Strategy & Annotation Format

### 2.1 Storage Layer Architecture

To allow human auditing without audio driver buffer truncation, the dataset maintains a clean two-layer structure:

1. **Layer 1 (Human Inspection & Source Storage):** Full-length match audio/video files (`.mp4` / `.wav`) paired with 3-column CSV annotation files.
    
2. **Layer 2 (Model Training Layer):** On-the-fly extracted $150\text{ ms}$ tensor slices or compiled NumPy matrix dumps (`X_tennis.npy`, `y_tennis.npy`).
    

### 2.2 Annotation File Schema

Annotation files must be saved in standard **3-column, headered UTF-8 CSV** format sharing the base filename of the media source (e.g., `match_01.wav` $\rightarrow$ `match_01.csv`):

Code snippet

```
file_name,timestamp_sec,label
match_01.wav,14.235,racket_hit
match_01.wav,14.890,ball_bounce
match_01.wav,15.110,shoe_squeak
match_01.wav,18.400,racket_hit
match_01.wav,22.050,grunt_speech
```

### 2.3 Acoustic Class Taxonomy & Dataset Targets

Target dataset volume: **~1,500 to 2,000 total manual annotations** across varied court surfaces.

| **Label Code**  | **Category**     | **Acoustic Characteristics**                | **Target Volume** | **Source Strategy**                      |
| --------------- | ---------------- | ------------------------------------------- | ----------------- | ---------------------------------------- |
| `racket_hit`    | **Racket Hit**   | Sharp < 5 ms attack + 120–180 Hz frame flex | **800 – 1,000**   | Manual annotation (Positive Class)       |
| `ball_bounce`   | **Ball Bounce**  | Short dull impact, lacking frame resonance  | **300 – 400**     | Manual annotation (Hard Negative)        |
| `shoe_squeak`   | **Shoe Squeak**  | Friction sweep (1 kHz – 3 kHz)              | **200 – 300**     | Manual annotation (Hard Negative)        |
| `grunt_speech`  | **Speech/Grunt** | Human vocal formants (300 Hz – 3 kHz)       | **200 – 300**     | Manual annotation (Hard Negative)        |
| `ambient_noise` | **Background**   | Stationary court hum, crowd, rain, HVAC     | **1,000+**        | **Automated** random off-stroke sampling |

### 2.4 Surface Diversity Ratios

To prevent model overfitting to surface-specific acoustics (e.g., hard-court squeaks or clay-court sliding):
- **50% Hard Court** (High-frequency reflection, heavy shoe squeaks)
- **35% Clay Court** (Dampened bounce, sliding noise, heavy player grunts)
- **15% Grass Court** (Low bounce energy, fast muted acoustic profiles)

## 3. Signal Processing & Feature Extraction

Every raw audio window is transformed into a **2D Log Mel-Spectrogram** before entering the neural network.
```
Raw Audio Stream (Mono 16 kHz)
       │
       ├── Crop Window: [T_annotated - 30ms : T_annotated + 120ms] (2,400 samples / 150 ms)
       │
       ├── Normalization: Peak amplitude scaling (y = y / max(|y|))
       │
       └── STFT Transformation:
             • Sample Rate: 16,000 Hz
             • Window Duration: 150 ms
             • n_fft: 400 (~25 ms window)
             • hop_length: 133 (~8.3 ms time step)
             • n_mels: 64
             
Tensor Input Shape: (Batch, Channels=1, Mel_Bins=64, Time_Frames=18)
```

## 4. Phase 1 Model Architecture: Custom 3-Layer CNN

A lightweight 2D Convolutional Neural Network designed for rapid experimentation and zero external dependencies.

```
Input Tensor: (Batch, 1, 64, 18)
  │
  ├── Block 1: Conv2D(1 → 32, k=3, p=1) ──> BatchNorm2D ──> ReLU ──> MaxPool2D(2, 2)  [32, 32, 9]
  ├── Block 2: Conv2D(32 → 64, k=3, p=1) ──> BatchNorm2D ──> ReLU ──> MaxPool2D(2, 2)  [64, 16, 4]
  ├── Block 3: Conv2D(64 → 128, k=3, p=1) ──> BatchNorm2D ──> ReLU ──> AdaptiveAvgPool2D(1, 1) [128, 1, 1]
  │
  ├── Flatten (128)
  ├── Dropout(p=0.3)
  └── Linear(128 → 5)  ──> CrossEntropyLoss (Training) / Softmax (Inference)
```

- **Parameter Footprint:** ~$150\text{k}$ parameters (~$1.5\text{ MB}$).
- **Export Target (Phase 1):** Native PyTorch Checkpoint (`tennis_hit_model.pth`).
- **Phase 2 Upgrade Option:** MobileNetV3-Small / EfficientNet-B0 if multi-court noise requires expanded capacity.

## 5. Real-Time Streaming & Inference Architecture

```
[ Incoming 16kHz Stream ] ──> [ 150ms Rolling Buffer ] ──> (Eval every 10ms) ──> [ CNN ] ──> [ Threshold & NMS ]
```

1. **Rolling Ring Buffer:** Maintains a moving $150\text{ ms}$ FIFO queue of audio samples in memory.
2. **Evaluation Step (Hop):** Slides forward by $10\text{ ms}$ ($160$ samples at $16\text{ kHz}$) yielding 100 model evaluations per second.
3. **Decision Rule:**
$$\text{Hit Detected} = \begin{cases} \text{True}, & \text{if } P(\text{racket\_hit}) \ge 0.85 \text{ AND } \Delta T_{\text{last\_hit}} > 200\text{ ms} \\ \text{False}, & \text{otherwise} \end{cases}$$
4. **Non-Maximum Suppression (NMS / Debounce):** Hard $200\text{ ms}$ lock-out window following a valid trigger to prevent multi-frame double counting on a single hit.


## 6. GUI Implementation Stack Brief

1. **Framework:** PySide6
2. **Video & Waveform Sync:** 
   - Video surface: `QMediaPlayer` + `QVideoWidget` (or OpenCV `cv2.VideoCapture` rendered to `QLabel`).
   - Waveform view: `pyqtgraph` (super fast interactive waveform plotting).
3. **Audio Audit Engine:** 
   - Use `sounddevice` or `pygame.mixer` to handle the isolated 150 ms loop playback off the main thread so the UI never freezes.
4. **Key Features to Implement:**
   - **Keyboard Hotkeys:** `[1]` = Racket Hit, `[2]` = Ball Bounce, `[3]` = Shoe Squeak, `[4]` = Grunt/Speech, `[Space]` = Play/Pause.
   - **Peak Snapping:** When a user clicks near an impact transient on the waveform, automatically snap the timestamp $T$ to the local maximum audio energy within a $\pm 30\text{ ms}$ window.
   - **CSV Auto-Save:** Auto-save/sync the 3-column CSV (`file_name`, `timestamp_sec`, `label`) whenever a hotkey is pressed.

## 7. Implementation Milestones

- [ ] **Milestone 1:** Build custom desktop video annotator (handling 3-column CSV output + 150 ms loop playback).
- [ ] **Milestone 2:** Collect initial 300-hit Hard Court dataset and build Python pre-processing script (`librosa` / `torchaudio`).
- [ ] **Milestone 3:** Train Phase 1 Custom 3-Layer CNN in PyTorch (`.pth`) and evaluate confusion matrix on test split.
- [ ] **Milestone 4:** Expand dataset across Clay and Grass matches to reach ~1,500 manual samples.
- [ ] **Milestone 5:** Export validated PyTorch model to **ONNX (`tennis_hit_model.onnx`)** and integrate into real-time streaming player loop.