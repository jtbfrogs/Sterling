"""
Sterling Vision — USB Webcam + YOLO + Face Recognition
=======================================================
Drop-in replacement for core/vision.py (HuskyLens2).
Exposes the same Block/Arrow interface so main.py needs minimal changes.

Architecture
------------
A background thread captures frames continuously from the webcam so
queries are instant — no lag waiting for a frame to arrive.

    CaptureThread:  VideoCapture → self._frame  (always fresh, locked)
    On query:       YOLO(self._frame) → detections
                    face_recognition(self._frame) → who is in frame

Face Enrollment
---------------
Drop a photo into the faces/ directory. Filename = person's name.
    faces/jtb.jpg     → recognised as "jtb"
    faces/guest.jpg   → recognised as "guest"
Multiple photos per person are supported: jtb_1.jpg, jtb_2.jpg etc.
All encodings for the same base name are averaged.

Running as a standalone demo
-----------------------------
    python vision_webcam.py
    python vision_webcam.py --camera 1 --model yolov8s.pt --no-faces

Dependencies
------------
    pip install ultralytics opencv-python face_recognition numpy
    # face_recognition also needs dlib:
    # M1 Mac:  brew install cmake && pip install dlib
    # Jetson:  pip install dlib  (CUDA picked up automatically)
    # Windows: pip install dlib  (pre-built wheel)
"""

import argparse
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Optional imports — degrade gracefully if not installed
# ─────────────────────────────────────────────────────────────────────────────

try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not installed. Run: pip install ultralytics")

try:
    import face_recognition as _fr
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("[WARNING] face_recognition not installed. Face ID will be disabled.")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures — match core/vision.py interface
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Block:
    """
    Detected object bounding box. Mirrors the HuskyLens Block dataclass
    in core/vision.py so the rest of Sterling needs no changes.
    """
    x:      int     # Centre X (pixels)
    y:      int     # Centre Y (pixels)
    width:  int
    height: int
    id:     int     # 0 = unknown / unrecognised, >0 = known face (index in known_names + 1)
    label:  str = ""    # Human-readable label e.g. "jtb", "person", "laptop"
    confidence: float = 0.0

    @property
    def is_learned(self) -> bool:
        """True if this detection maps to a known/enrolled person."""
        return self.id > 0


# ─────────────────────────────────────────────────────────────────────────────
# Webcam Vision
# ─────────────────────────────────────────────────────────────────────────────

# COCO class names for the 80 classes YOLOv8 detects
COCO_CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "potted plant", "bed", "dining table", "toilet", "tv monitor",
    "laptop", "mouse", "remote", "keyboard", "mobile phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


class WebcamVision:
    """
    USB webcam vision using YOLOv8 + face_recognition.

    Runs a background capture thread for zero-latency queries.
    Drop-in replacement for core/vision.HuskyLens.

    Usage:
        cam = WebcamVision(device_index=0, model_size="yolov8n.pt")
        cam.start()

        blocks, _ = cam.get_all()
        for b in blocks:
            print(b.label, "at", b.x, b.y)

        cam.disconnect()
    """

    def __init__(
        self,
        device_index:       int  = 0,
        model_size:         str  = "yolov8n.pt",
        face_recognition:   bool = True,
        known_faces_dir:    str  = "faces",
        confidence_thresh:  float = 0.45,
    ):
        """
        Args:
            device_index:       OpenCV camera index. 0 = first USB cam.
            model_size:         YOLO model file. yolov8n.pt is fastest.
            face_recognition:   Enable face ID. Requires face_recognition library.
            known_faces_dir:    Path to folder containing enrollment photos.
            confidence_thresh:  Minimum YOLO confidence to report a detection.
        """
        self._device_index      = device_index
        self._model_size        = model_size
        self._use_face_recog    = face_recognition and FACE_RECOGNITION_AVAILABLE
        self._known_faces_dir   = Path(known_faces_dir)
        self._conf_thresh       = confidence_thresh

        # YOLO model
        self._yolo = None
        if YOLO_AVAILABLE:
            print(f"Loading YOLO model: {model_size} ...")
            self._yolo = _YOLO(model_size)
            print("YOLO ready.")

        # Face recognition
        self._known_encodings:  list       = []
        self._known_names:      list[str]  = []
        if self._use_face_recog:
            self._load_known_faces()

        # Capture state
        self._cap:          Optional[cv2.VideoCapture] = None
        self._frame:        Optional[np.ndarray]       = None
        self._frame_lock    = threading.Lock()
        self._running       = False
        self._capture_thread: Optional[threading.Thread] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def start(self):
        """Open the camera and start the background capture thread."""
        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {self._device_index}. "
                "Check the device is connected and not in use."
            )
        # Reasonable capture settings
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="sterling-capture"
        )
        self._capture_thread.start()

        # Wait for first frame
        for _ in range(20):
            if self._frame is not None:
                break
            time.sleep(0.05)

        print(f"Camera online — device {self._device_index}")

    def disconnect(self):
        """Stop the capture thread and release the camera."""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        print("Camera offline.")

    def ping(self) -> bool:
        """Return True if the camera is open and delivering frames."""
        return self._cap is not None and self._cap.isOpened() and self._frame is not None

    # ─────────────────────────────────────────────────────────────────────────
    # Detection API — matches HuskyLens interface
    # ─────────────────────────────────────────────────────────────────────────

    def get_all(self) -> tuple[list[Block], list]:
        """
        Detect all objects and people in the current frame.
        Returns (blocks, []) — arrows not applicable for webcam vision.
        """
        return self._detect(), []

    def get_blocks(self) -> list[Block]:
        """All detected objects (people + things)."""
        return self._detect()

    def get_learned_blocks(self) -> list[Block]:
        """Only recognised (enrolled) people."""
        return [b for b in self._detect() if b.is_learned]

    def startup_scan(self, face_map: dict = None) -> str:
        """
        Take one detection pass and return a plain-English description.
        Used at boot time to confirm the camera is working.
        """
        time.sleep(0.5)  # let camera settle
        blocks, _ = self.get_all()
        if not blocks:
            return ""

        names = []
        for b in blocks:
            if b.is_learned:
                names.append(b.label)
            else:
                names.append(f"unrecognised {b.label}")

        return ", ".join(names)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal — detection
    # ─────────────────────────────────────────────────────────────────────────

    def _detect(self) -> list[Block]:
        """Run YOLO + face recognition on the latest frame."""
        frame = self._get_frame()
        if frame is None or not YOLO_AVAILABLE:
            return []

        results  = self._yolo(frame, verbose=False, conf=self._conf_thresh)
        boxes    = results[0].boxes
        detected: list[Block] = []

        # Map YOLO detections to Block objects
        for box in boxes:
            cls_id  = int(box.cls[0])
            label   = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "unknown"
            conf    = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx      = (x1 + x2) // 2
            cy      = (y1 + y2) // 2
            w       = x2 - x1
            h       = y2 - y1

            detected.append(Block(
                x=cx, y=cy, width=w, height=h,
                id=0, label=label, confidence=conf,
            ))

        # Overlay face recognition results on person detections
        if self._use_face_recog and self._known_encodings:
            detected = self._apply_face_recognition(frame, detected)

        return detected

    def _apply_face_recognition(
        self, frame: np.ndarray, blocks: list[Block]
    ) -> list[Block]:
        """
        Run face_recognition on the frame and update person blocks
        with the recognised name and ID.
        """
        rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations  = _fr.face_locations(rgb, model="hog")   # hog = CPU-friendly
        encodings  = _fr.face_encodings(rgb, locations)

        for (top, right, bottom, left), encoding in zip(locations, encodings):
            matches    = _fr.compare_faces(self._known_encodings, encoding, tolerance=0.55)
            distances  = _fr.face_distance(self._known_encodings, encoding)

            name    = "unknown"
            face_id = 0

            if True in matches:
                best = int(np.argmin(distances))
                if matches[best]:
                    name    = self._known_names[best]
                    face_id = best + 1  # 1-indexed, 0 = unknown

            face_cx = (left + right)  // 2
            face_cy = (top  + bottom) // 2
            face_w  = right  - left
            face_h  = bottom - top

            # Update an existing "person" block if it overlaps, else add new
            matched_block = None
            for b in blocks:
                if b.label == "person":
                    if abs(b.x - face_cx) < 80 and abs(b.y - face_cy) < 80:
                        matched_block = b
                        break

            if matched_block:
                matched_block.id    = face_id
                matched_block.label = name
            else:
                blocks.append(Block(
                    x=face_cx, y=face_cy,
                    width=face_w, height=face_h,
                    id=face_id, label=name,
                    confidence=1.0,
                ))

        return blocks

    # ─────────────────────────────────────────────────────────────────────────
    # Internal — face enrollment
    # ─────────────────────────────────────────────────────────────────────────

    def _load_known_faces(self):
        """
        Load face encodings from the known_faces_dir.
        Each image file's stem is used as the person's name.
        Multiple images with the same base name (jtb_1.jpg, jtb_2.jpg) are
        all enrolled under the same name.
        """
        if not self._known_faces_dir.exists():
            print(f"[Vision] No faces directory found at '{self._known_faces_dir}'. "
                  "Face recognition disabled.")
            return

        image_files = list(self._known_faces_dir.glob("*.jpg")) + \
                      list(self._known_faces_dir.glob("*.jpeg")) + \
                      list(self._known_faces_dir.glob("*.png"))

        if not image_files:
            print(f"[Vision] faces/ directory is empty — no faces enrolled.")
            return

        print(f"[Vision] Loading {len(image_files)} face image(s)...")

        for path in image_files:
            # Strip trailing _1, _2 etc. to get base name
            base_name = re.sub(r"_\d+$", "", path.stem).lower()

            try:
                image     = _fr.load_image_file(str(path))
                encodings = _fr.face_encodings(image)
                if not encodings:
                    print(f"  [!] No face found in {path.name} — skipping.")
                    continue
                self._known_encodings.append(encodings[0])
                self._known_names.append(base_name)
                print(f"  ✓ Enrolled: {base_name} ({path.name})")
            except Exception as e:
                print(f"  [!] Failed to load {path.name}: {e}")

        print(f"[Vision] {len(self._known_encodings)} face encoding(s) loaded.")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal — capture thread
    # ─────────────────────────────────────────────────────────────────────────

    def _capture_loop(self):
        """Background thread — continuously reads frames from the webcam."""
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._frame_lock:
                    self._frame = frame
            else:
                time.sleep(0.01)

    def _get_frame(self) -> Optional[np.ndarray]:
        """Return the latest captured frame (thread-safe)."""
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# Standalone demo
# ─────────────────────────────────────────────────────────────────────────────

def _draw_blocks(frame: np.ndarray, blocks: list[Block]) -> np.ndarray:
    """Draw bounding boxes and labels on a frame."""
    for b in blocks:
        x1 = b.x - b.width  // 2
        y1 = b.y - b.height // 2
        x2 = b.x + b.width  // 2
        y2 = b.y + b.height // 2

        colour = (0, 200, 0) if b.is_learned else (0, 140, 255)
        label  = f"{b.label} ({b.confidence:.0%})" if b.confidence else b.label

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            frame, label, (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2,
        )
    return frame


def main():
    parser = argparse.ArgumentParser(description="Sterling webcam vision demo")
    parser.add_argument("--camera",    type=int,   default=0,            help="Camera device index")
    parser.add_argument("--model",     type=str,   default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--faces-dir", type=str,   default="faces",      help="Known faces directory")
    parser.add_argument("--no-faces",  action="store_true",              help="Disable face recognition")
    parser.add_argument("--conf",      type=float, default=0.45,         help="Detection confidence threshold")
    args = parser.parse_args()

    cam = WebcamVision(
        device_index=args.camera,
        model_size=args.model,
        face_recognition=not args.no_faces,
        known_faces_dir=args.faces_dir,
        confidence_thresh=args.conf,
    )

    cam.start()
    print("\nLive preview — press Q to quit\n")

    fps_timer = time.time()
    fps       = 0.0
    frame_count = 0

    try:
        while True:
            frame = cam._get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            blocks, _ = cam.get_all()
            frame     = _draw_blocks(frame, blocks)

            # FPS counter
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()

            cv2.putText(
                frame, f"FPS: {fps:.1f}  Detections: {len(blocks)}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

            # Print detections to console
            if blocks:
                names = [b.label for b in blocks]
                print(f"\r  Seeing: {', '.join(names):<60}", end="", flush=True)

            cv2.imshow("Sterling Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("\n")
        cam.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
