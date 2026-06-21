"""
Sterling Gesture Detector
==========================
Background gesture recognition using YOLOv8n-pose keypoints.
Runs in a daemon thread at configurable intervals; fires registered
callbacks only after a gesture has been held for a sustain window
(prevents single-frame false positives).

Supported gestures
------------------
    wave        — either wrist rises above the nose line
    hands_up    — both wrists above shoulder line
    point_right — right arm extended horizontally to the right
    point_left  — left arm extended horizontally to the left

Note on thumbs-up / thumbs-down
---------------------------------
These require per-finger keypoints that YOLOv8-pose doesn't provide
(it only gives 17 body-level keypoints).  Thumbs detection needs a
dedicated hand-landmark model such as MediaPipe Hands — tracked in
FUTURE_ITERATIONS.md as a medium-term addition.

The pose model (yolov8n-pose.pt, ~7 MB) is downloaded automatically
by ultralytics on first use and cached in the ultralytics model dir.
"""

import threading
import time
from typing import Callable, Optional

import numpy as np
from utils.logger import setup_logger

logger = setup_logger("sterling.vision.gesture")

# COCO-Pose 17-keypoint indices
_NOSE           = 0
_LEFT_SHOULDER  = 5
_RIGHT_SHOULDER = 6
_LEFT_ELBOW     = 7
_RIGHT_ELBOW    = 8
_LEFT_WRIST     = 9
_RIGHT_WRIST    = 10

try:
    from ultralytics import YOLO as _YOLO
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class GestureDetector:
    """
    Background gesture detector.

    Usage::

        gd = GestureDetector(frame_fn=webcam.get_frame)
        gd.on_gesture("wave",       lambda: sterling.gesture_wake())
        gd.on_gesture("hands_up",   lambda: sterling.tts.stop())
        gd.on_gesture("point_right",lambda: sterling.spotify.skip())
        gd.on_gesture("point_left", lambda: sterling.spotify.previous())
        gd.start()
        ...
        gd.stop()

    Callbacks run in short-lived daemon threads so they never block the
    detection loop.
    """

    def __init__(
        self,
        frame_fn:          Callable,
        model_size:        str   = "yolov8n-pose.pt",
        poll_interval:     float = 1.0,
        sustain_seconds:   float = 1.5,
        confidence:        float = 0.45,
        point_min_distance:int   = 120,
        point_max_height:  int   = 80,
    ):
        """
        Args:
            frame_fn:           Callable → np.ndarray | None.  Latest camera frame.
            model_size:         YOLOv8-pose weights.  Auto-downloaded on first use.
            poll_interval:      Seconds between gesture checks.
            sustain_seconds:    Seconds a gesture must be held before the callback fires.
            confidence:         Minimum keypoint confidence to trust.
            point_min_distance: Min pixel arm extension to register as a point gesture.
            point_max_height:   Max vertical deviation for a horizontal point gesture.
        """
        if not _AVAILABLE:
            raise RuntimeError(
                "ultralytics is required for gesture detection.  "
                "It should already be installed; reinstall if missing."
            )

        self._get_frame    = frame_fn
        self._model_size   = model_size
        self._interval     = poll_interval
        self._sustain      = sustain_seconds
        self._conf         = confidence
        self._point_min_dx = point_min_distance
        self._point_max_dy = point_max_height

        self._model: Optional[object]                = None
        self._callbacks: dict[str, list[Callable]]   = {}
        self._running    = False
        self._thread: Optional[threading.Thread]     = None

        # Debounce state
        self._current: Optional[str] = None
        self._since:   float         = 0.0
        self._fired:   bool          = False

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def on_gesture(self, gesture: str, callback: Callable) -> None:
        """Register a callback for a named gesture."""
        self._callbacks.setdefault(gesture, []).append(callback)

    def start(self) -> None:
        """Load the pose model (auto-download if needed) and start the loop."""
        logger.info(f"Loading gesture model: {self._model_size}")
        self._model   = _YOLO(self._model_size)
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="sterling-gesture"
        )
        self._thread.start()
        logger.info("Gesture detector running.")

    def stop(self) -> None:
        """Stop the background loop and release the model."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self._model = None
        logger.debug("Gesture detector stopped.")

    # ─────────────────────────────────────────────────────────────────────────
    # Background loop
    # ─────────────────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        _next = time.monotonic()
        while self._running:
            _next += self._interval
            frame = self._get_frame()
            if frame is not None:
                try:
                    gesture = self._detect_gesture(frame)
                    self._debounce(gesture)
                except Exception as e:
                    logger.debug(f"Gesture detection error: {e}")
            slack = _next - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            else:
                _next = time.monotonic()

    # ─────────────────────────────────────────────────────────────────────────
    # Gesture classification
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_gesture(self, frame: np.ndarray) -> Optional[str]:
        """
        Run the pose model and classify the dominant gesture, or None.

        Keypoint layout (COCO-Pose 17):
            0 nose | 5/6 shoulders | 7/8 elbows | 9/10 wrists
        """
        results = self._model(frame, verbose=False)

        if (not results or results[0].keypoints is None
                or results[0].keypoints.data is None):
            return None

        kdata = results[0].keypoints.data
        if len(kdata) == 0:
            return None

        # Use the first (largest / highest-confidence) detected person
        kpts = kdata[0].cpu().numpy()   # shape (17, 3): x, y, confidence

        def pt(idx) -> Optional[np.ndarray]:
            """Return keypoint if confidence meets threshold, else None."""
            return kpts[idx] if kpts[idx][2] >= self._conf else None

        nose      = pt(_NOSE)
        l_wrist   = pt(_LEFT_WRIST)
        r_wrist   = pt(_RIGHT_WRIST)
        l_shoulder = pt(_LEFT_SHOULDER)
        r_shoulder = pt(_RIGHT_SHOULDER)

        if nose is None:
            return None   # can't classify without a reference point

        # ── Wave: either wrist above the nose ────────────────────────────────
        if l_wrist is not None and l_wrist[1] < nose[1]:
            return "wave"
        if r_wrist is not None and r_wrist[1] < nose[1]:
            return "wave"

        # ── Both hands up: wrists above shoulder line ─────────────────────────
        if (l_wrist   is not None and l_shoulder is not None and
                r_wrist is not None and r_shoulder is not None):
            if l_wrist[1] < l_shoulder[1] and r_wrist[1] < r_shoulder[1]:
                return "hands_up"

        # ── Point right: right wrist clearly right of right shoulder ──────────
        if r_wrist is not None and r_shoulder is not None:
            dx = r_wrist[0] - r_shoulder[0]
            dy = abs(r_wrist[1] - r_shoulder[1])
            if dx > self._point_min_dx and dy < self._point_max_dy:
                return "point_right"

        # ── Point left: left wrist clearly left of left shoulder ──────────────
        if l_wrist is not None and l_shoulder is not None:
            dx = l_shoulder[0] - l_wrist[0]
            dy = abs(l_wrist[1] - l_shoulder[1])
            if dx > self._point_min_dx and dy < self._point_max_dy:
                return "point_left"

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Debounce + callback dispatch
    # ─────────────────────────────────────────────────────────────────────────

    def _debounce(self, gesture: Optional[str]) -> None:
        """
        Fire a callback only after the same gesture has been held
        continuously for sustain_seconds.  Resets when gesture changes.
        """
        now = time.monotonic()

        if gesture != self._current:
            # Gesture changed (or became None) — reset state
            self._current = gesture
            self._since   = now
            self._fired   = False
            return

        if gesture is None or self._fired:
            return

        if now - self._since >= self._sustain:
            self._fired = True  # block re-firing until gesture breaks
            logger.info(f"Gesture detected: {gesture}")
            for cb in self._callbacks.get(gesture, []):
                threading.Thread(target=self._safe_call, args=(cb,), daemon=True).start()

    @staticmethod
    def _safe_call(cb: Callable) -> None:
        try:
            cb()
        except Exception as e:
            logger.debug(f"Gesture callback raised: {e}")
