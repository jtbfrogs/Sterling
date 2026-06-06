"""
Sterling Vision — USB Webcam
==============================
YOLOv8 object/person detection + face_recognition for face ID.

Face Enrollment
---------------
Drop a photo into vision/faces/. The filename becomes the person's name:
    vision/faces/jtb.jpg     → recognised as "jtb"
    vision/faces/guest.jpg   → recognised as "guest"
Multiple photos per person: jtb_1.jpg, jtb_2.jpg → both enrolled as "jtb"
Restart Sterling to pick up new faces.

Dependencies
------------
    pip install ultralytics opencv-python

    face_recognition (optional — enables face ID):
        M1 Mac:   brew install cmake && pip install dlib face_recognition
        Jetson:   pip install dlib face_recognition
        Windows:  pip install dlib face_recognition
"""

import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from utils.logger import setup_logger

logger = setup_logger("sterling.vision.webcam")

# ─────────────────────────────────────────────────────────────────────────────
# Optional dependencies
# ─────────────────────────────────────────────────────────────────────────────

try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed — webcam vision unavailable. Run: pip install ultralytics")

try:
    import face_recognition as _fr
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logger.info("face_recognition not installed — face ID disabled. Person detection still works.")


# ─────────────────────────────────────────────────────────────────────────────
# COCO class names (80 classes YOLOv8 detects out of the box)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Block — matches core/vision.py so main.py needs no changes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Block:
    x:          int
    y:          int
    width:      int
    height:     int
    id:         int           # 0 = unknown, >0 = enrolled face index (1-based)
    label:      str  = ""     # "jtb", "person", "laptop" etc.
    confidence: float = 0.0

    @property
    def is_learned(self) -> bool:
        """True if this block is a recognised/enrolled face."""
        return self.id > 0


# ─────────────────────────────────────────────────────────────────────────────
# WebcamVision
# ─────────────────────────────────────────────────────────────────────────────

class WebcamVision:
    """
    USB webcam vision using YOLOv8 + face_recognition.
    Configured via config.yaml under the vision section.

    Usage:
        cam = WebcamVision(device_index=0)
        cam.start()
        blocks, _ = cam.get_all()
        cam.disconnect()
    """

    def __init__(
        self,
        device_index:       int   = 0,
        model_size:         str   = "yolov8n.pt",
        face_recognition:   bool  = True,
        known_faces_dir:    str   = "vision/faces",
        confidence_thresh:  float = 0.45,
    ):
        """
        Args:
            device_index:       OpenCV camera index. 0 = first USB cam.
            model_size:         YOLO model weights file.
            face_recognition:   Enable face ID (requires face_recognition library).
            known_faces_dir:    Path to folder of enrollment photos.
            confidence_thresh:  Min YOLO confidence to report a detection.
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError(
                "ultralytics is required for webcam vision. "
                "Install it: pip install ultralytics"
            )

        self._device_index    = device_index
        self._conf_thresh     = confidence_thresh
        self._use_face_recog  = face_recognition and FACE_RECOGNITION_AVAILABLE
        self._known_faces_dir = Path(known_faces_dir)

        # YOLO
        logger.info(f"Loading YOLO model: {model_size} ...")
        self._yolo = _YOLO(model_size)
        logger.info("YOLO ready.")

        # Face recognition
        self._known_encodings: list      = []
        self._known_names:     list[str] = []
        if self._use_face_recog:
            self._load_known_faces()
        elif face_recognition and not FACE_RECOGNITION_AVAILABLE:
            logger.warning(
                "face_recognition library not installed — face ID disabled. "
                "Install: brew install cmake && pip install dlib face_recognition"
            )

        # Capture state
        self._cap:             Optional[cv2.VideoCapture] = None
        self._frame:           Optional[np.ndarray]       = None
        self._frame_lock       = threading.Lock()
        self._running          = False
        self._capture_thread:  Optional[threading.Thread] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def start(self):
        """Open the webcam and start the background capture thread."""
        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {self._device_index}. "
                "Check it's connected and not in use by another app."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="sterling-webcam"
        )
        self._capture_thread.start()

        # Wait up to 1s for the first frame
        for _ in range(20):
            if self._frame is not None:
                break
            time.sleep(0.05)

        logger.info(f"Webcam online — device index {self._device_index}")

    def disconnect(self):
        """Stop capture thread and release the camera."""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        logger.info("Webcam disconnected.")

    def ping(self) -> bool:
        """Return True if the camera is open and delivering frames."""
        return (
            self._cap is not None
            and self._cap.isOpened()
            and self._frame is not None
        )

    def switch_algorithm(self, algorithm: int):
        """No-op — included for interface compatibility."""
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Detection API
    # ─────────────────────────────────────────────────────────────────────────

    def get_all(self) -> tuple[list[Block], list]:
        """All detections. Returns (blocks, []) — arrows not used for webcam."""
        return self._detect(), []

    def get_blocks(self) -> list[Block]:
        """All detected objects and people."""
        return self._detect()

    def get_learned_blocks(self) -> list[Block]:
        """Only recognised (enrolled) faces."""
        return [b for b in self._detect() if b.is_learned]

    def startup_scan(self, face_map: dict = None) -> str:
        """
        One-shot scan at boot — returns a plain-English description.
        Confirms camera is working and shows initial room state.
        """
        time.sleep(0.5)
        blocks, _ = self.get_all()
        if not blocks:
            logger.info("Webcam startup scan: nothing detected.")
            return ""

        parts = []
        for b in blocks:
            if b.is_learned:
                parts.append(b.label)
            else:
                parts.append(f"unrecognised {b.label}")

        desc = ", ".join(parts)
        logger.info(f"Webcam startup scan: {desc}")
        return desc

    # ─────────────────────────────────────────────────────────────────────────
    # Detection internals
    # ─────────────────────────────────────────────────────────────────────────

    def _detect(self) -> list[Block]:
        """Run YOLO (+ optional face recognition) on the latest frame."""
        frame = self._get_frame()
        if frame is None:
            return []

        results  = self._yolo(frame, verbose=False, conf=self._conf_thresh)
        boxes    = results[0].boxes
        detected: list[Block] = []

        for box in boxes:
            cls_id = int(box.cls[0])
            label  = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "unknown"
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detected.append(Block(
                x      = (x1 + x2) // 2,
                y      = (y1 + y2) // 2,
                width  = x2 - x1,
                height = y2 - y1,
                id     = 0,
                label  = label,
                confidence = conf,
            ))

        if self._use_face_recog and self._known_encodings:
            detected = self._apply_face_recognition(frame, detected)

        return detected

    def _apply_face_recognition(
        self, frame: np.ndarray, blocks: list[Block]
    ) -> list[Block]:
        """
        Run face_recognition on the current frame and update person blocks
        with enrolled names and IDs. Unknown faces get id=0, label="unknown".
        """
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = _fr.face_locations(rgb, model="hog")
        encodings = _fr.face_encodings(rgb, locations)

        for (top, right, bottom, left), encoding in zip(locations, encodings):
            matches   = _fr.compare_faces(self._known_encodings, encoding, tolerance=0.55)
            distances = _fr.face_distance(self._known_encodings, encoding)

            name    = "unknown"
            face_id = 0

            if True in matches:
                best = int(np.argmin(distances))
                if matches[best]:
                    name    = self._known_names[best]
                    face_id = best + 1   # 1-indexed so 0 stays "unknown"

            face_cx = (left + right)  // 2
            face_cy = (top  + bottom) // 2

            # Update an overlapping "person" block rather than adding a duplicate
            merged = False
            for b in blocks:
                if b.label == "person" and abs(b.x - face_cx) < 100 and abs(b.y - face_cy) < 100:
                    b.id    = face_id
                    b.label = name
                    merged  = True
                    break

            if not merged:
                blocks.append(Block(
                    x=face_cx, y=face_cy,
                    width=right - left, height=bottom - top,
                    id=face_id, label=name, confidence=1.0,
                ))

        return blocks

    # ─────────────────────────────────────────────────────────────────────────
    # Face enrollment
    # ─────────────────────────────────────────────────────────────────────────

    def _load_known_faces(self):
        """
        Load face encodings from known_faces_dir.
        Filename stem = person's name. Multiple photos per person supported.
        """
        if not self._known_faces_dir.exists():
            logger.warning(
                f"Faces directory not found: '{self._known_faces_dir}'. "
                "Create it and add photos to enable face recognition. "
                "Example: vision/faces/jtb.jpg"
            )
            return

        image_files = (
            list(self._known_faces_dir.glob("*.jpg"))  +
            list(self._known_faces_dir.glob("*.jpeg")) +
            list(self._known_faces_dir.glob("*.png"))
        )

        if not image_files:
            logger.info("vision/faces/ is empty — no faces enrolled yet.")
            return

        logger.info(f"Loading face encodings from {self._known_faces_dir} ...")

        for path in sorted(image_files):
            base_name = re.sub(r"_\d+$", "", path.stem).lower()
            try:
                image     = _fr.load_image_file(str(path))
                encodings = _fr.face_encodings(image)
                if not encodings:
                    logger.warning(f"  No face detected in {path.name} — skipping.")
                    continue
                self._known_encodings.append(encodings[0])
                self._known_names.append(base_name)
                logger.info(f"  ✓ Enrolled: {base_name} ({path.name})")
            except Exception as e:
                logger.warning(f"  Failed to load {path.name}: {e}")

        logger.info(f"Face recognition ready — {len(self._known_encodings)} encoding(s) loaded.")

    # ─────────────────────────────────────────────────────────────────────────
    # Capture thread
    # ─────────────────────────────────────────────────────────────────────────

    def _capture_loop(self):
        """Background thread — keeps self._frame updated continuously."""
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._frame_lock:
                    self._frame = frame
            else:
                time.sleep(0.01)

    def _get_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None
