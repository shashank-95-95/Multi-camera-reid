# AI-Based Multi-Camera Person Re-Identification System

## Phase 2 — Multi-Camera Person Detection & Tracking

A modular Python application that detects and tracks people across **multiple camera streams** using **YOLOv8s** (Ultralytics) and **DeepSORT**. Each camera operates as an independent pipeline with its own detector, tracker, and output files.

---

## Features

- **Multi-camera support** — process any number of video streams
- **Independent tracking** — each camera has its own detector, tracker, and track IDs
- **Per-camera outputs** — separate video and JSON for every camera
- **Real-time display** — one OpenCV window per camera
- **Pluggable execution** — sequential now, threaded/parallel ready
- **YOLOv8s detection** with configurable confidence and bbox filtering
- **DeepSORT tracking** tuned for surveillance stability
- **Phase 3 ready** — clear extension points for OSNet ReID integration

---

## Project Structure

```
person-reid/
│
├── camera/                          # Phase 2 — multi-camera orchestration
│   ├── camera_pipeline.py           #   CameraPipeline — per-camera pipeline
│   └── camera_manager.py            #   CameraManager  — orchestrator
│
├── detection/
│   └── detector.py                  #   PersonDetector — YOLOv8s wrapper
│
├── tracking/
│   └── tracker.py                   #   DeepSortTracker — DeepSORT wrapper
│
├── utils/
│   ├── video.py                     #   VideoProcessor — video I/O & JSON
│   └── draw.py                      #   Drawing utilities (bboxes, HUD)
│
├── outputs/                         #   Generated at runtime
│   ├── camera_1/
│   │   ├── tracked_video.mp4
│   │   └── tracking_results.json
│   ├── camera_2/
│   │   ├── tracked_video.mp4
│   │   └── tracking_results.json
│   └── ...
│
├── sample_videos/                   #   Place test videos here
│
├── main.py                          #   CLI entry point
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Navigate to the project

```bash
cd person-reid
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

| OS            | Command                        |
|---------------|--------------------------------|
| Windows       | `venv\Scripts\activate`        |
| macOS / Linux | `source venv/bin/activate`     |

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will automatically download `yolov8s.pt` weights (~21 MB) from Ultralytics.

---

## Usage

### Single camera

```bash
python main.py --videos sample_videos/cam1.mp4
```

### Multiple cameras

```bash
python main.py --videos sample_videos/cam1.mp4 sample_videos/cam2.mp4 sample_videos/cam3.mp4
```

### All options

```bash
python main.py \
    --videos cam1.mp4 cam2.mp4 \
    --output outputs \
    --model yolov8s.pt \
    --confidence 0.6 \
    --device "" \
    --min-bbox-width 40 \
    --min-bbox-height 60 \
    --no-display
```

| Flag              | Default       | Description                                     |
|-------------------|---------------|-------------------------------------------------|
| `--videos`        | *(required)*  | One or more input video file paths              |
| `--output`        | `outputs`     | Base output directory                           |
| `--model`         | `yolov8s.pt`  | Path to YOLO weights                            |
| `--confidence`    | `0.6`         | Minimum detection confidence                    |
| `--device`        | `""` (auto)   | Compute device (`cpu`, `cuda`, `mps`)           |
| `--min-bbox-width`| `40`          | Minimum detection width (pixels)                |
| `--min-bbox-height`| `60`         | Minimum detection height (pixels)               |
| `--no-display`    | `False`       | Run headless (no OpenCV windows)                |

### Controls

- Press **Q** or **Esc** in any camera window to stop all pipelines.

---

## Output Format

### Per-camera directory

Each camera produces:

- `outputs/camera_N/tracked_video.mp4` — annotated video
- `outputs/camera_N/tracking_results.json` — structured tracking log

### JSON entry format

```json
{
    "camera_id": 1,
    "frame": 120,
    "track_id": 5,
    "bbox": [120, 200, 280, 550],
    "confidence": 0.94,
    "timestamp": 4.12
}
```

> **Note:** Track IDs are **local to each camera**. Camera 1 and Camera 2 may both have a Track ID `1` — these refer to different people.

---

## Architecture

### Execution Model

```
CameraManager
  ├── CameraPipeline (Camera 1) ── Detector + Tracker + VideoProcessor
  ├── CameraPipeline (Camera 2) ── Detector + Tracker + VideoProcessor
  └── CameraPipeline (Camera N) ── Detector + Tracker + VideoProcessor
```

- **CameraPipeline** exposes a single `run()` method
- **CameraManager** decides *how* to call it (sequential / threaded / parallel)
- Switching execution strategy requires changes **only** in CameraManager

### Phase 3 Integration Point

`CameraPipeline.run()` contains a clearly marked extension point:

```python
# ─── Phase 3 extension point — Person Re-Identification ───
# Future integration steps:
#   1. Crop each tracked person from the frame
#   2. Extract appearance embeddings via OSNet
#   3. Query / update a GlobalIdentityManager
#   4. Replace local track_id with global_id
```

---

## Roadmap

| Phase | Description                     | Status       |
|-------|---------------------------------|--------------|
| 1     | Detection & Tracking            | ✅ Complete   |
| 2     | Multi-Camera Support            | ✅ Complete   |
| 3     | OSNet Person Re-Identification  | 🔲 Planned   |
| 4     | FastAPI Backend                 | 🔲 Planned   |
| 5     | Dashboard                       | 🔲 Planned   |

---

## Tech Stack

- **Python** 3.11
- **Ultralytics** (YOLOv8s)
- **deep-sort-realtime**
- **OpenCV**
- **NumPy**

---

## License

This project is developed as part of an internship programme.
