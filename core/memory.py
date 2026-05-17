"""
Sterling Memory Manager
=======================
Manages conversation context for the LLM.

Session (RAM)
-------------
Stores a sliding window of conversation turns (user + assistant pairs).
The system prompt is always prepended to every LLM call.
Oldest turns are dropped automatically to stay within the model's token limit.

Persistent (JSON)
-----------------
Every message is written to memory.json immediately after it's added.
Each session is stamped with start/end times and per-message timestamps.
On startup, the last N turns from previous sessions are recalled into the
active context window so Sterling remembers across restarts.

memory.json structure
---------------------
{
  "sessions": [
    {
      "session_id": "2026-05-15_10-30-00",
      "started":    "2026-05-15T10:30:00",
      "ended":      "2026-05-15T10:45:12",   ← null if session crashed
      "turn_count": 5,
      "messages": [
        {"role": "user",      "content": "...", "timestamp": "2026-05-15T10:30:05"},
        {"role": "assistant", "content": "...", "timestamp": "2026-05-15T10:30:08"}
      ]
    }
  ]
}
"""

import json
import os
from datetime import datetime
from pathlib import Path

from utils.logger import setup_logger

logger = setup_logger("sterling.memory")


class Memory:
    """
    Conversation context manager with optional JSON persistence.

    Maintains a list of {"role": ..., "content": ...} dicts for the LLM.
    Enforces a max history window to keep context within the model's token limit.
    Optionally persists every message to disk and recalls previous sessions on startup.
    """

    def __init__(
        self,
        system_prompt: str = "",
        max_history: int = 20,
        persist: bool = True,
        memory_file: str = "memory.json",
        recall_turns: int = 10,
    ):
        """
        Args:
            system_prompt:  Sterling's personality / instruction prompt.
                            Prepended to every LLM message list.
            max_history:    Max conversation turns (each = 1 user + 1 assistant message)
                            to keep in the active context window. Older turns are dropped.
            persist:        Write every message to memory_file immediately.
            memory_file:    Path to the JSON memory file.
            recall_turns:   On startup, inject this many turns from previous sessions
                            into the context window so Sterling remembers past convos.
                            Set to 0 to start fresh every time.
        """
        self._system_prompt = system_prompt
        self._max_messages = max_history * 2        # each turn = 2 messages
        self._history: list[dict] = []              # active context (no timestamps — LLM-ready)
        self._session_start = datetime.now()

        # Persistence
        self._persist = persist
        self._memory_file = Path(memory_file)
        self._session_id = self._session_start.strftime("%Y-%m-%d_%H-%M-%S")
        self._session_start_iso = self._session_start.isoformat(timespec="seconds")
        self._current_session_msgs: list[dict] = []  # timestamped, for JSON only

        if persist:
            self._load_and_recall(recall_turns)

        logger.info(
            f"Memory initialized — max history: {max_history} turns, "
            f"persist: {persist}"
            + (f", memory file: {memory_file}, recall: {recall_turns} turns" if persist else "")
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Adding messages
    # ─────────────────────────────────────────────────────────────────────────

    def add_user(self, text: str):
        """Append a user message to conversation history."""
        self._history.append({"role": "user", "content": text})
        self._trim()
        if self._persist:
            self._track_and_save("user", text)
        logger.debug(f"Memory: user message added ({len(self._history)} total messages)")

    def add_assistant(self, text: str):
        """Append an assistant response to conversation history."""
        self._history.append({"role": "assistant", "content": text})
        self._trim()
        if self._persist:
            self._track_and_save("assistant", text)
        logger.debug(f"Memory: assistant message added ({len(self._history)} total messages)")

    # ─────────────────────────────────────────────────────────────────────────
    # Retrieving context
    # ─────────────────────────────────────────────────────────────────────────

    def get_messages(self) -> list[dict]:
        """
        Build the full message list for the LLM.

        Returns:
            List of message dicts with system prompt prepended.
        """
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(self._history)
        return messages

    def get_history(self) -> list[dict]:
        """Return just the conversation history (no system prompt)."""
        return list(self._history)

    # ─────────────────────────────────────────────────────────────────────────
    # Session management
    # ─────────────────────────────────────────────────────────────────────────

    def end_session(self):
        """
        Stamp the current session with an end time and write a final save.
        Call this from Sterling's shutdown() so every session has a clean end time.
        If the process crashes the session will have ended: null in the JSON,
        which is still fully readable.
        """
        if not self._persist or not self._current_session_msgs:
            return
        self._save(ended=True)
        logger.info(
            f"Memory session closed — {self.turn_count} turns saved to {self._memory_file}"
        )

    def clear(self):
        """Clear conversation history. System prompt is preserved. JSON is NOT wiped."""
        count = len(self._history)
        self._history = []
        logger.info(f"Memory cleared — removed {count} messages from active context.")

    def update_system_prompt(self, prompt: str):
        """Replace the system prompt (takes effect on next LLM call)."""
        self._system_prompt = prompt
        logger.debug("System prompt updated.")

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def turn_count(self) -> int:
        """Number of complete conversation turns (user + assistant pairs)."""
        return len(self._history) // 2

    @property
    def message_count(self) -> int:
        """Total number of individual messages in history."""
        return len(self._history)

    @property
    def session_duration(self) -> str:
        """Human-readable session duration since Memory was created."""
        delta = datetime.now() - self._session_start
        minutes, seconds = divmod(int(delta.total_seconds()), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence — internal
    # ─────────────────────────────────────────────────────────────────────────

    def _track_and_save(self, role: str, content: str):
        """Add a timestamped entry to the current session log and auto-save."""
        self._current_session_msgs.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        self._save()

    def _save(self, ended: bool = False):
        """
        Atomically write the current session to memory.json.
        Reads the existing file first so previous sessions are preserved.
        Uses a .tmp file + rename to prevent corruption on crash.
        """
        # Load existing data (or start fresh)
        data = self._load_raw()

        # Find or create the entry for this session
        sessions: list[dict] = data.setdefault("sessions", [])
        session_entry = next(
            (s for s in sessions if s.get("session_id") == self._session_id), None
        )
        if session_entry is None:
            session_entry = {
                "session_id": self._session_id,
                "started": self._session_start_iso,
                "ended": None,
                "turn_count": 0,
                "messages": [],
            }
            sessions.append(session_entry)

        session_entry["messages"] = self._current_session_msgs
        session_entry["turn_count"] = len(self._current_session_msgs) // 2
        if ended:
            session_entry["ended"] = datetime.now().isoformat(timespec="seconds")

        # Atomic write: .tmp → rename
        tmp_path = self._memory_file.with_suffix(".tmp")
        try:
            self._memory_file.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            tmp_path.rename(self._memory_file)
        except Exception as e:
            logger.error(f"Memory save failed: {e}")
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _load_raw(self) -> dict:
        """Read the raw JSON from disk. Returns empty structure on any failure."""
        if not self._memory_file.exists():
            return {"sessions": []}
        try:
            with open(self._memory_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read memory file ({e}) — starting with empty store.")
            return {"sessions": []}

    def _load_and_recall(self, recall_turns: int):
        """
        On startup, load the last `recall_turns` conversation turns from all
        previous sessions and inject them into the active context window.
        This gives Sterling continuity across restarts.
        """
        if recall_turns <= 0:
            return

        data = self._load_raw()
        past_sessions = [
            s for s in data.get("sessions", [])
            if s.get("session_id") != self._session_id  # exclude current (shouldn't exist yet)
        ]

        if not past_sessions:
            logger.info("Memory: no previous sessions found.")
            return

        # Collect messages newest-first from previous sessions, then reverse
        recalled: list[dict] = []
        for session in reversed(past_sessions):
            msgs = session.get("messages", [])
            recalled = msgs + recalled          # prepend older session
            if len(recalled) // 2 >= recall_turns:
                break

        # Trim to the last recall_turns pairs (newest)
        max_msgs = recall_turns * 2
        recalled = recalled[-max_msgs:]

        # Strip timestamps — LLM only needs role + content
        for msg in recalled:
            self._history.append({"role": msg["role"], "content": msg["content"]})

        total_sessions = len(past_sessions)
        logger.info(
            f"Memory: recalled {len(recalled)} messages from {total_sessions} previous "
            f"session{'s' if total_sessions != 1 else ''}."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    def _trim(self):
        """
        Keep history within the max_messages limit.
        Drops the oldest messages first.
        Always drops in pairs (user + assistant) to keep roles balanced.
        """
        if len(self._history) > self._max_messages:
            excess = len(self._history) - self._max_messages
            drop = excess + (excess % 2)        # round up to nearest pair
            self._history = self._history[drop:]
            logger.debug(f"Memory trimmed — dropped {drop} messages, {len(self._history)} remain.")
