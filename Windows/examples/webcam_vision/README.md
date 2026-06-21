# Webcam Vision — Sterling Example

USB webcam + YOLOv8 + face_recognition as a drop-in replacement for the HuskyLens2.
Exposes the same interface as `core/vision.py` so Sterling's main.py needs minimal changes.

---

## Setup

### 1. Install dependencies

```bash
source ster/bin/activate

# M1 Mac
brew install cmake
pip install dlib
pip install -r examples/webcam_vision/requirements.txt

# Jetson / Linux
pip install -r examples/webcam_vision/requirements.txt

# Windows
pip install -r examples/webcam_vision/requirements.txt
```

### 2. Enrol faces (optional but recommended)

Drop a clear photo of each person into the `faces/` folder.
The filename (without extension) becomes the person's name.

```
examples/webcam_vision/faces/
  jtb.jpg       → recognised as "jtb"
  guest.jpg     → recognised as "guest"
```

One good front-facing photo per person is enough.
Multiple photos per person improve accuracy — just name them `jtb_1.jpg`, `jtb_2.jpg` etc.

### 3. Run the demo

```bash
source ster/bin/activate
python examples/webcam_vision/vision_webcam.py
```

Opens a live preview window showing:
- Bounding boxes around detected objects and people
- Name labels on recognised faces
- "Unknown" label on unrecognised faces
- FPS counter

Press **Q** to quit.

---

## Integrating into Sterling

When ready to integrate properly:

1. Copy `vision_webcam.py` to `core/vision_webcam.py`
2. Copy `faces/` to the project root
3. Update `config.yaml`:
   ```yaml
   vision:
     backend: "webcam"
     device_index: 0
     yolo_model: "yolov8n.pt"
     face_recognition: true
     known_faces_dir: "faces/"
   ```
4. Update `main.py` `_init_vision()` to pick backend from config

---

## Hardware Notes

| Platform | YOLO backend | face_recognition |
|---|---|---|
| M1 Mac | Metal (MPS) — automatic | CPU |
| Jetson Orin | CUDA — automatic | CUDA via dlib |
| Windows RTX | CUDA — automatic | CPU or CUDA |

YOLO model recommendations:
- `yolov8n.pt` — fastest, good for always-on monitoring
- `yolov8s.pt` — better accuracy, still real-time on GPU
