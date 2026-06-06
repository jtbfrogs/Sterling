"""
Sterling Memory Manager
=======================
Two-layer memory architecture:

  Layer 1 — Session window (RAM)
  --------------------------------
  A sliding window of the current conversation's recent turns.
  Fast, always accurate for what's happening right now.
  Cleared when Sterling restarts.

  Layer 2 — Semantic long-term memory (ChromaDB)
  -----------------------------------------------
  Every completed exchange (user + assistant pair) is embedded and stored.
  Before each LLM call, the current user message is used to query ChromaDB
  for semantically similar past exchanges. Only relevant context is injected
  — not a blind dump of the last N messages.
  Persists across restarts in the .chroma directory.

  Layer 3 — JSON archive
  ----------------------
  Full timestamped history of every session written to memory.json.
  Used for ChromaDB backfill and audit trail. Not injected directly
  into the LLM (ChromaDB handles recall instead).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.logger import setup_logger

logger = setup_logger("sterling.memory")


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB semantic memory
# ─────────────────────────────────────────────────────────────────────────────

class ChromaMemory:
    """
    Semantic long-term memory backed by ChromaDB.
    Stores user+assistant exchange pairs as embedded documents.
    Retrieves the most contextually relevant past exchanges on demand.
    """

    def __init__(self, persist_dir: str = ".chroma", n_results: int = 3):
        """
        Args:
            persist_dir: Directory for ChromaDB files. Created if it doesn't exist.
            n_results:   Max past exchanges to inject per LLM call.
        """
        import chromadb
        self._n_results = n_results
        try:
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._client.get_or_create_collection(
                name="sterling_conversations",
                metadata={"hnsw:space": "cosine"},
            )
            count = self._collection.count()
            logger.info(
                f"ChromaDB ready — {count} past exchange{'s' if count != 1 else ''} stored "
                f"at '{persist_dir}'"
            )
        except Exception as e:
            logger.warning(f"ChromaDB init failed: {e}")
            raise

    def add(self, user_msg: str, assistant_msg: str, session_id: str):
        """Store a completed exchange in ChromaDB."""
        doc_id    = f"{session_id}_{datetime.now().strftime('%H%M%S%f')}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        document  = f"[{timestamp}] User: {user_msg}\nSterling: {assistant_msg}"
        try:
            self._collection.upsert(
                documents=[document],
                ids=[doc_id],
                metadatas=[{
                    "session_id": session_id,
                    "timestamp":  datetime.now().isoformat(timespec="seconds"),
                }],
            )
        except Exception as e:
            logger.warning(f"ChromaDB add failed: {e}")

    def query(self, text: str) -> list[str]:
        """
        Retrieve the most semantically relevant past exchanges for a given query.

        Returns:
            List of document strings (each is "User: ...\nSterling: ...").
            Empty list if nothing relevant found or ChromaDB is empty.
        """
        count = self._collection.count()
        if count == 0:
            return []
        try:
            n       = min(self._n_results, count)
            results = self._collection.query(query_texts=[text], n_results=n)
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            logger.warning(f"ChromaDB query failed: {e}")
            return []

    @property
    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:
            return 0


# ─────────────────────────────────────────────────────────────────────────────
# Main memory manager
# ─────────────────────────────────────────────────────────────────────────────

class Memory:
    """
    Sterling conversation memory manager.

    Combines a short-term session window with optional ChromaDB semantic recall.
    Falls back gracefully to JSON-based recall if ChromaDB is unavailable.
    """

    def __init__(
        self,
        system_prompt:  str  = "",
        max_history:    int  = 10,
        persist:        bool = True,
        memory_file:    str  = "memory.json",
        recall_turns:   int  = 2,
        chroma_enabled: bool = True,
        chroma_path:    str  = ".chroma",
        chroma_results: int  = 3,
    ):
        """
        Args:
            system_prompt:  Sterling's personality prompt — prepended to every LLM call.
            max_history:    Max turns to keep in the active session window.
            persist:        Write every message to memory_file.
            memory_file:    Path to the JSON archive.
            recall_turns:   Turns to inject from JSON on startup if ChromaDB is disabled.
                            Kept small (2) when ChromaDB is active.
            chroma_enabled: Enable semantic long-term memory via ChromaDB.
            chroma_path:    Directory for ChromaDB data files.
            chroma_results: How many relevant past exchanges ChromaDB injects per call.
        """
        self._system_prompt  = system_prompt
        self._max_messages   = max_history * 2
        self._history:  list[dict] = []
        self._session_start  = datetime.now()
        self._session_id     = self._session_start.strftime("%Y-%m-%d_%H-%M-%S")
        self._last_user_msg: Optional[str] = None

        # JSON persistence
        self._persist        = persist
        self._memory_file    = Path(memory_file)
        self._current_session_msgs: list[dict] = []

        # ChromaDB semantic memory
        self._chroma: Optional[ChromaMemory] = None
        if chroma_enabled:
            try:
                self._chroma = ChromaMemory(
                    persist_dir=chroma_path,
                    n_results=chroma_results,
                )
            except Exception as e:
                logger.warning(
                    f"ChromaDB unavailable ({e}) — falling back to JSON recall."
                )

        # Recall previous context on startup
        if persist:
            recall = recall_turns if self._chroma is None else recall_turns
            self._load_and_recall(recall)

        logger.info(
            f"Memory ready — window: {max_history} turns, "
            f"chroma: {'enabled' if self._chroma else 'disabled'}, "
            f"persist: {persist}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Adding messages
    # ─────────────────────────────────────────────────────────────────────────

    def add_user(self, text: str):
        """Append a user message and cache it for ChromaDB storage after response."""
        self._history.append({"role": "user", "content": text})
        self._last_user_msg = text
        self._trim()
        if self._persist:
            self._track_and_save("user", text)

    def add_assistant(self, text: str):
        """Append assistant response and store the completed exchange in ChromaDB."""
        self._history.append({"role": "assistant", "content": text})
        self._trim()
        if self._persist:
            self._track_and_save("assistant", text)

        # Store completed exchange in ChromaDB
        if self._chroma and self._last_user_msg:
            self._chroma.add(self._last_user_msg, text, self._session_id)
            self._last_user_msg = None

    # ─────────────────────────────────────────────────────────────────────────
    # Retrieving context for LLM
    # ─────────────────────────────────────────────────────────────────────────

    def get_messages(self) -> list[dict]:
        """
        Build the full message list for the LLM.

        Order:
          1. System prompt (personality + instructions)
          2. Relevant past context from ChromaDB (if available and relevant)
          3. Current session window
        """
        messages = []

        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        # Semantic recall — inject only relevant past exchanges
        if self._chroma and self._last_user_msg:
            relevant = self._chroma.query(self._last_user_msg)
            if relevant:
                context_block = "\n---\n".join(relevant)
                messages.append({
                    "role": "system",
                    "content": (
                        "The following are past conversations for background context only. "
                        "They have already happened — do not treat them as current or ongoing. "
                        "Use them only if directly relevant to what the user is asking right now.\n\n"
                        f"{context_block}"
                    ),
                })
                logger.debug(f"ChromaDB injected {len(relevant)} relevant past exchange(s).")

        messages.extend(self._history)
        return messages

    # ─────────────────────────────────────────────────────────────────────────
    # Session management
    # ─────────────────────────────────────────────────────────────────────────

    def end_session(self):
        """Stamp current session with end time and write final save."""
        if not self._persist or not self._current_session_msgs:
            return
        self._save(ended=True)
        logger.info(
            f"Memory session closed — {self.turn_count} turns saved to {self._memory_file}"
        )

    def clear(self):
        """Clear active session window. JSON and ChromaDB are NOT wiped."""
        count = len(self._history)
        self._history = []
        self._last_user_msg = None
        logger.info(f"Memory cleared — removed {count} messages from session window.")

    def update_system_prompt(self, prompt: str):
        """Replace the system prompt (takes effect on next LLM call)."""
        self._system_prompt = prompt

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def turn_count(self) -> int:
        return len(self._history) // 2

    @property
    def message_count(self) -> int:
        return len(self._history)

    @property
    def session_duration(self) -> str:
        delta   = datetime.now() - self._session_start
        minutes, seconds = divmod(int(delta.total_seconds()), 60)
        hours, minutes   = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    @property
    def chroma_count(self) -> int:
        """Total exchanges stored in ChromaDB."""
        return self._chroma.count if self._chroma else 0

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _track_and_save(self, role: str, content: str):
        self._current_session_msgs.append({
            "role":      role,
            "content":   content,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        self._save()

    def _save(self, ended: bool = False):
        """Atomically write current session to memory.json (.tmp → rename)."""
        data     = self._load_raw()
        sessions = data.setdefault("sessions", [])

        entry = next(
            (s for s in sessions if s.get("session_id") == self._session_id), None
        )
        if entry is None:
            entry = {
                "session_id": self._session_id,
                "started":    self._session_start.isoformat(timespec="seconds"),
                "ended":      None,
                "turn_count": 0,
                "messages":   [],
            }
            sessions.append(entry)

        entry["messages"]   = self._current_session_msgs
        entry["turn_count"] = len(self._current_session_msgs) // 2
        if ended:
            entry["ended"]  = datetime.now().isoformat(timespec="seconds")

        tmp = self._memory_file.with_suffix(".tmp")
        try:
            self._memory_file.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            tmp.rename(self._memory_file)
        except Exception as e:
            logger.error(f"Memory save failed: {e}")
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def _load_raw(self) -> dict:
        if not self._memory_file.exists():
            return {"sessions": []}
        try:
            with open(self._memory_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read memory file ({e}) — starting fresh.")
            return {"sessions": []}

    def _load_and_recall(self, recall_turns: int):
        """
        Inject the last N turns from previous sessions into the session window.
        Used as a lightweight recent-context anchor alongside ChromaDB semantic recall.
        Kept small (default 2) when ChromaDB is active.
        """
        if recall_turns <= 0:
            return

        data         = self._load_raw()
        past         = [
            s for s in data.get("sessions", [])
            if s.get("session_id") != self._session_id
        ]

        if not past:
            logger.info("Memory: no previous sessions found.")
            return

        recalled: list[dict] = []
        for session in reversed(past):
            recalled = session.get("messages", []) + recalled
            if len(recalled) // 2 >= recall_turns:
                break

        recalled = recalled[-(recall_turns * 2):]

        # Wrap recalled turns in a system message so the LLM knows they are
        # from a past session — NOT the current conversation.  Injecting them
        # as bare user/assistant turns caused the LLM to treat stale topics
        # (e.g. a Jarvis discussion days ago) as the active subject.
        lines = []
        for msg in recalled:
            speaker = "User" if msg["role"] == "user" else "Sterling"
            lines.append(f"{speaker}: {msg['content']}")

        self._history.append({
            "role": "system",
            "content": (
                "The following exchange is from a PREVIOUS session that has already ended. "
                "Treat it as background context only — do NOT continue it or treat it as "
                "part of the current conversation.\n\n"
                + "\n".join(lines)
            ),
        })

        logger.info(
            f"Memory: recalled {len(recalled)} messages from previous session(s) as background context."
        )

    def _trim(self):
        """Drop oldest messages to stay within the session window limit."""
        if len(self._history) > self._max_messages:
            excess = len(self._history) - self._max_messages
            drop   = excess + (excess % 2)
            self._history = self._history[drop:]
