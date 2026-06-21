"""
Sterling Object Tracker
========================
Persists the last-seen position and timestamp of trackable everyday
objects across sessions.  Updated passively every time the camera is
queried — no background YOLO loop.

Answers questions like "where's my phone?" or "have you seen my keys?"
by injecting last-seen context into the LLM prompt.

Stored in object_tracker.json (project root, gitignored).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.logger import setup_logger

logger = setup_logger("sterling.vision.tracker")

# YOLO COCO label → canonical friendly name we track.
# Only everyday items worth remembering.
TRACKABLE: dict[str, str] = {
    "cell phone":   "phone",
    "laptop":       "laptop",
    "book":         "book",
    "cup":          "cup",
    "bottle":       "bottle",
    "remote":       "remote",
    "mouse":        "mouse",
    "keyboard":     "keyboard",
    "backpack":     "bag",
    "handbag":      "bag",
    "suitcase":     "luggage",
    "umbrella":     "umbrella",
    "scissors":     "scissors",
    "clock":        "clock",
    "tv":           "tv",
    "couch":        "couch",
    "chair":        "chair",
    "bicycle":      "bike",
    "wine glass":   "glass",
}

# Natural speech aliases the user might say → canonical name
_ALIASES: dict[str, list[str]] = {
    "phone":    ["phone", "cell", "mobile", "iphone", "android", "cellphone"],
    "laptop":   ["laptop", "computer", "mac", "macbook", "pc", "notebook"],
    "book":     ["book", "notebook", "magazine"],
    "cup":      ["cup", "mug", "coffee", "tea"],
    "bottle":   ["bottle", "water bottle", "water"],
    "remote":   ["remote", "controller", "clicker"],
    "mouse":    ["mouse"],
    "keyboard": ["keyboard"],
    "bag":      ["bag", "backpack", "purse", "tote"],
    "luggage":  ["luggage", "suitcase"],
    "tv":       ["tv", "television", "screen"],
    "glasses":  ["glasses", "sunglasses"],
    "keys":     ["keys", "key"],
    "wallet":   ["wallet"],
    "bike":     ["bike", "bicycle"],
    "glass":    ["glass", "wine glass", "cup"],
}


class ObjectTracker:
    """
    Lightweight, file-backed last-seen tracker.

    Updated by passing raw YOLO detection dicts to update().
    Queried by passing free-form user text to find().

    Thread-safe for read; writes are atomic (tmp → rename).
    """

    def __init__(self, path: str = "object_tracker.json"):
        self._path = Path(path)
        self._data: dict = self._load()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, detections: list[dict]) -> None:
        """
        Record positions of trackable objects from a list of detection dicts.
        Each dict should have at least 'label', 'h_pos', and 'depth' keys,
        as produced by WebcamVision.get_scene_description internals.
        """
        now     = datetime.now().isoformat(timespec="seconds")
        changed = False

        for det in detections:
            label     = det.get("label", "")
            canonical = TRACKABLE.get(label)
            if canonical:
                self._data[canonical] = {
                    "last_seen": now,
                    "position":  det.get("h_pos", ""),
                    "depth":     det.get("depth", ""),
                    "raw_label": label,
                }
                changed = True

        if changed:
            self._save()

    def find(self, query: str) -> Optional[str]:
        """
        Return a plain-English description of where the item was last seen,
        or None if this item has never been tracked.
        Suitable for direct injection into LLM context.
        """
        canonical = self._resolve(query)
        if not canonical:
            return None

        entry = self._data.get(canonical)
        if not entry:
            return None

        # Build a human-readable time-ago string
        try:
            dt    = datetime.fromisoformat(entry["last_seen"])
            mins  = int((datetime.now() - dt).total_seconds() // 60)
            if mins < 2:
                when = "just now"
            elif mins < 60:
                when = f"about {mins} minute{'s' if mins != 1 else ''} ago"
            elif mins < 120:
                when = "about an hour ago"
            else:
                hours = mins // 60
                when = f"about {hours} hour{'s' if hours != 1 else ''} ago"
        except Exception:
            when = "recently"

        pos   = entry.get("position", "")
        depth = entry.get("depth", "")
        loc_parts = [p for p in [pos, depth] if p]
        location  = ", ".join(loc_parts) if loc_parts else "somewhere in frame"

        return f"Your {canonical} was last seen {when}, {location} of frame."

    def summary(self) -> str:
        """
        Return a compact summary of all tracked items — useful for
        injecting full inventory context into the LLM.
        """
        if not self._data:
            return "No objects tracked yet."
        lines = []
        for canonical, entry in sorted(self._data.items()):
            try:
                dt   = datetime.fromisoformat(entry["last_seen"])
                mins = int((datetime.now() - dt).total_seconds() // 60)
                when = f"{mins}m ago" if mins < 60 else f"{mins//60}h ago"
            except Exception:
                when = "unknown"
            pos  = entry.get("position", "")
            lines.append(f"{canonical}: {pos}, {when}")
        return "; ".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve(self, query: str) -> Optional[str]:
        """Map free-form user speech to a canonical tracked item name."""
        q = query.lower()
        for canonical, aliases in _ALIASES.items():
            for alias in aliases:
                if alias in q:
                    return canonical
        # Fall back to direct canonical match
        for canonical in self._data:
            if canonical in q:
                return canonical
        return None

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Object tracker load failed ({e}) — starting fresh.")
            return {}

    def _save(self) -> None:
        try:
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            tmp.rename(self._path)
        except Exception as e:
            logger.debug(f"Object tracker save failed: {e}")
