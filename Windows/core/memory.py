"""
Sterling Memory Manager
=======================
Three-layer memory architecture designed to work reliably on Python 3.14
(where ChromaDB's native bindings are unavailable).

  Layer 1 — Session window (RAM)
  -------------------------------
  A sliding window of the *current* conversation's recent turns. Stores ONLY
  the clean user utterance and Sterling's reply — never the bracketed system
  context that gets injected for a single generation. This is the fix for the
  "ask for music, then ask for a joke, get a music answer" bug: ephemeral
  instructions no longer pollute future turns.

  Layer 2 — Keyword recall (long-term, JSON-backed)
  -------------------------------------------------
  Every completed exchange is archived to memory.json. On each turn the current
  user message is scored against past exchanges by weighted keyword overlap and
  only the most relevant ones are injected as background context. This is a
  pure-Python stand-in for semantic search that needs no native dependencies.

  Layer 3 — Facts / profile store (facts.json)
  --------------------------------------------
  Durable, explicit facts ("my name is jtb", "I like jazz"). Always available,
  retrieved by relevance, and survives restarts. This is what makes Sterling
  feel like it actually *knows* you.

ChromaDB remains optional: if it ever imports cleanly it is used in addition to
keyword recall, otherwise everything degrades gracefully.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.logger import setup_logger

logger = setup_logger("sterling.memory")


# ─────────────────────────────────────────────────────────────────────────────
# Stop words for keyword recall (kept small + fast)
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "at", "for", "with", "as", "by",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "this", "that",
    "these", "those", "do", "does", "did", "can", "could", "would", "should",
    "will", "shall", "may", "might", "must", "have", "has", "had", "what",
    "when", "where", "who", "why", "how", "which", "whom", "if", "then",
    "so", "just", "about", "up", "down", "out", "get", "got", "im", "ive",
    "youre", "dont", "thats", "whats", "okay", "ok", "yeah", "yes", "no",
    "hey", "please", "thanks", "thank", "now", "like", "want", "tell", "say",
}

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set[str]:
    """Lowercase content words with stopwords removed."""
    return {
        w for w in _WORD_RE.findall(text.lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB semantic memory (optional — best-effort, never required)
# ─────────────────────────────────────────────────────────────────────────────

class ChromaMemory:
    """Semantic long-term memory backed by ChromaDB. Optional."""

    def __init__(self, persist_dir: str = ".chroma", n_results: int = 3):
        import chromadb
        self._n_results = n_results
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name="sterling_conversations",
            metadata={"hnsw:space": "cosine"},
        )
        count = self._collection.count()
        logger.info(f"ChromaDB ready — {count} past exchange(s) at '{persist_dir}'")

    def add(self, user_msg: str, assistant_msg: str, session_id: str):
        doc_id    = f"{session_id}_{datetime.now().strftime('%H%M%S%f')}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        document  = f"[{timestamp}] User: {user_msg}\nSterling: {assistant_msg}"
        try:
            self._collection.upsert(
                documents=[document], ids=[doc_id],
                metadatas=[{"session_id": session_id,
                            "timestamp": datetime.now().isoformat(timespec="seconds")}],
            )
        except Exception as e:
            logger.warning(f"ChromaDB add failed: {e}")

    def query(self, text: str) -> list[str]:
        try:
            count = self._collection.count()
            if count == 0:
                return []
            n = min(self._n_results, count)
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
# Keyword recall — pure-Python long-term retrieval (the real workhorse)
# ─────────────────────────────────────────────────────────────────────────────

class KeywordRecall:
    """
    Lightweight relevance search over archived exchanges.

    Scores each past exchange against the query by weighted keyword overlap
    (Jaccard-ish, biased toward rarer words and recency). No native deps,
    works everywhere, and is genuinely useful for "what did we talk about".
    """

    def __init__(self, memory_file: Path, n_results: int = 3, current_session_id: str = ""):
        self._memory_file = memory_file
        self._n_results   = n_results
        self._current_id  = current_session_id
        # Each entry: {"user", "assistant", "tokens", "timestamp", "session_id"}
        self._exchanges: list[dict] = []
        self._df: dict[str, int]    = {}   # document frequency per token
        self._load()

    def _load(self):
        if not self._memory_file.exists():
            return
        try:
            with open(self._memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        for session in data.get("sessions", []):
            sid  = session.get("session_id", "")
            msgs = session.get("messages", [])
            # Pair user→assistant turns
            i = 0
            while i < len(msgs) - 1:
                if msgs[i].get("role") == "user" and msgs[i + 1].get("role") == "assistant":
                    user = msgs[i].get("content", "")
                    asst = msgs[i + 1].get("content", "")
                    toks = _tokens(user) | _tokens(asst)
                    if toks:
                        self._exchanges.append({
                            "user": user, "assistant": asst, "tokens": toks,
                            "timestamp": msgs[i].get("timestamp", ""), "session_id": sid,
                        })
                        for t in toks:
                            self._df[t] = self._df.get(t, 0) + 1
                    i += 2
                else:
                    i += 1
        logger.info(f"KeywordRecall: indexed {len(self._exchanges)} past exchange(s).")

    def add(self, user_msg: str, assistant_msg: str, session_id: str):
        toks = _tokens(user_msg) | _tokens(assistant_msg)
        if not toks:
            return
        self._exchanges.append({
            "user": user_msg, "assistant": assistant_msg, "tokens": toks,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id,
        })
        for t in toks:
            self._df[t] = self._df.get(t, 0) + 1

    def query(self, text: str, min_score: float = 0.12) -> list[str]:
        q = _tokens(text)
        if not q or not self._exchanges:
            return []

        import math
        total = len(self._exchanges)
        scored = []
        for ex in self._exchanges:
            # Skip exchanges from the live session (the window already has them)
            if ex["session_id"] == self._current_id:
                continue
            overlap = q & ex["tokens"]
            if not overlap:
                continue
            # Inverse-document-frequency weighting: rarer shared words count more
            score = sum(math.log(1 + total / self._df.get(t, 1)) for t in overlap)
            score /= math.sqrt(len(ex["tokens"]))   # normalise for long exchanges
            scored.append((score, ex))

        if not scored:
            return []
        scored.sort(key=lambda s: s[0], reverse=True)
        top = [ex for sc, ex in scored[: self._n_results] if sc >= min_score]

        out = []
        for ex in top:
            ts = ex["timestamp"][:16].replace("T", " ")
            out.append(f"[{ts}] User: {ex['user']}\nSterling: {ex['assistant']}")
        return out

    @property
    def count(self) -> int:
        return len(self._exchanges)


# ─────────────────────────────────────────────────────────────────────────────
# Facts / profile store
# ─────────────────────────────────────────────────────────────────────────────

class FactStore:
    """Durable, explicit facts about the user. Survives restarts."""

    def __init__(self, path: str = "facts.json", max_inject: int = 6):
        self._path = Path(path)
        self._max_inject = max_inject
        self._facts: list[dict] = []   # {"text", "tokens", "timestamp"}
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for item in raw.get("facts", []):
                txt = item.get("text", "").strip()
                if txt:
                    self._facts.append({
                        "text": txt, "tokens": _tokens(txt),
                        "timestamp": item.get("timestamp", ""),
                    })
            logger.info(f"FactStore: loaded {len(self._facts)} fact(s).")
        except Exception as e:
            logger.warning(f"FactStore load failed: {e}")

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"facts": [{"text": x["text"], "timestamp": x["timestamp"]}
                                     for x in self._facts]}, f, indent=2, ensure_ascii=False)
            tmp.rename(self._path)
        except Exception as e:
            logger.warning(f"FactStore save failed: {e}")

    def add(self, text: str) -> bool:
        text = text.strip().rstrip(".")
        if not text:
            return False
        # De-dupe on high token overlap
        new_toks = _tokens(text)
        for fct in self._facts:
            if fct["tokens"] and new_toks and len(new_toks & fct["tokens"]) / len(new_toks | fct["tokens"]) > 0.7:
                fct["text"] = text  # refresh wording
                fct["timestamp"] = datetime.now().isoformat(timespec="seconds")
                self._save()
                return True
        self._facts.append({
            "text": text, "tokens": new_toks,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        self._save()
        return True

    def relevant(self, query: str) -> list[str]:
        """Most-recent facts plus any that overlap the query."""
        if not self._facts:
            return []
        q = _tokens(query)
        recent = self._facts[-self._max_inject:]
        relevant = [f for f in self._facts if q & f["tokens"]]
        # Merge, preserve order, cap
        seen, out = set(), []
        for f in relevant + list(reversed(recent)):
            if f["text"] not in seen:
                seen.add(f["text"])
                out.append(f["text"])
            if len(out) >= self._max_inject:
                break
        return out

    def clear(self) -> int:
        n = len(self._facts)
        self._facts = []
        self._save()
        return n

    @property
    def count(self) -> int:
        return len(self._facts)


# ─────────────────────────────────────────────────────────────────────────────
# Main memory manager
# ─────────────────────────────────────────────────────────────────────────────

class Memory:
    """
    Sterling conversation memory manager.

    Combines a clean short-term session window with keyword recall + a facts
    store. The session window stores only clean utterances; per-turn system
    context is injected ephemerally via get_messages(ephemeral_context=...).
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
        facts_file:     str  = "facts.json",
        recall_results: int  = 3,
    ):
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

        # Layer 2 — keyword recall (always available)
        self._recall = KeywordRecall(
            memory_file=self._memory_file,
            n_results=recall_results,
            current_session_id=self._session_id,
        )

        # Layer 3 — facts store
        self._facts = FactStore(path=facts_file)

        # Optional ChromaDB (best-effort; keyword recall is the real engine)
        self._chroma: Optional[ChromaMemory] = None
        if chroma_enabled:
            try:
                self._chroma = ChromaMemory(persist_dir=chroma_path, n_results=chroma_results)
            except Exception as e:
                logger.info(f"ChromaDB unavailable ({type(e).__name__}) — using keyword recall.")

        # Anchor a couple of recent turns from last session for continuity
        if persist and recall_turns > 0:
            self._load_recent_anchor(recall_turns)

        logger.info(
            f"Memory ready — window={max_history} turns, "
            f"recall={self._recall.count} exchanges, facts={self._facts.count}, "
            f"chroma={'on' if self._chroma else 'off'}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Adding messages
    # ─────────────────────────────────────────────────────────────────────────

    def add_user(self, text: str):
        """Append a CLEAN user message (no bracketed system context)."""
        self._history.append({"role": "user", "content": text})
        self._last_user_msg = text
        self._trim()
        if self._persist:
            self._track_and_save("user", text)

    def pop_last_user(self):
        """Remove the most recent user turn (called when TTS was interrupted)."""
        if self._history and self._history[-1]["role"] == "user":
            self._history.pop()
            self._last_user_msg = None
            logger.debug("Popped interrupted user message from session window.")

    def add_assistant(self, text: str):
        """Append assistant reply and archive the completed exchange."""
        self._history.append({"role": "assistant", "content": text})
        self._trim()
        if self._persist:
            self._track_and_save("assistant", text)
        if self._last_user_msg:
            self._recall.add(self._last_user_msg, text, self._session_id)
            if self._chroma:
                self._chroma.add(self._last_user_msg, text, self._session_id)
            self._last_user_msg = None

    # ─────────────────────────────────────────────────────────────────────────
    # Facts
    # ─────────────────────────────────────────────────────────────────────────

    def remember_fact(self, text: str) -> bool:
        return self._facts.add(text)

    def forget_facts(self) -> int:
        return self._facts.clear()

    @property
    def fact_count(self) -> int:
        return self._facts.count

    # ─────────────────────────────────────────────────────────────────────────
    # Retrieving context for LLM
    # ─────────────────────────────────────────────────────────────────────────

    def get_messages(self, ephemeral_context: Optional[str] = None) -> list[dict]:
        """
        Build the message list for the LLM.

        Order:
          1. System prompt
          2. Facts / profile (durable)
          3. Relevant past exchanges (keyword recall + optional Chroma)
          4. Session window (clean turns)

        Args:
            ephemeral_context: Per-turn system data (weather, vision, action
                confirmations, etc). Injected as a transient system message for
                THIS generation only — never stored in the window. This is what
                stops one turn's context from bleeding into the next.
        """
        messages = [{"role": "system", "content": self._system_prompt}] if self._system_prompt else []

        query = self._last_user_msg or ""

        # Facts
        facts = self._facts.relevant(query)
        if facts:
            messages.append({
                "role": "system",
                "content": "What you know about the user (durable facts):\n- " + "\n- ".join(facts),
            })

        # Long-term recall
        recalled = self._recall.query(query) if query else []
        if self._chroma and query:
            for doc in self._chroma.query(query):
                if doc not in recalled:
                    recalled.append(doc)
        if recalled:
            messages.append({
                "role": "system",
                "content": (
                    "Relevant snippets from PAST conversations (already happened — "
                    "use only as background, do not treat as the current topic):\n\n"
                    + "\n---\n".join(recalled[:4])
                ),
            })

        # Session window
        messages.extend(self._history)

        # Ephemeral per-turn context — appended to the last user message copy
        if ephemeral_context and messages and messages[-1]["role"] == "user":
            tail = dict(messages[-1])
            tail["content"] = f"{tail['content']}\n\n{ephemeral_context}"
            messages[-1] = tail
        elif ephemeral_context:
            messages.append({"role": "system", "content": ephemeral_context})

        return messages

    # ─────────────────────────────────────────────────────────────────────────
    # Session management
    # ─────────────────────────────────────────────────────────────────────────

    def end_session(self):
        if not self._persist or not self._current_session_msgs:
            return
        self._save(ended=True)
        logger.info(f"Memory session closed — {self.turn_count} turns saved.")

    def clear(self):
        count = len(self._history)
        self._history = []
        self._last_user_msg = None
        logger.info(f"Memory cleared — removed {count} messages from session window.")

    def update_system_prompt(self, prompt: str):
        self._system_prompt = prompt

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def turn_count(self) -> int:
        return len([m for m in self._history if m["role"] == "user"])

    @property
    def message_count(self) -> int:
        return len(self._history)

    @property
    def session_duration(self) -> str:
        delta = datetime.now() - self._session_start
        minutes, seconds = divmod(int(delta.total_seconds()), 60)
        hours, minutes   = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    @property
    def chroma_count(self) -> int:
        return self._chroma.count if self._chroma else self._recall.count

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _track_and_save(self, role: str, content: str):
        self._current_session_msgs.append({
            "role": role, "content": content,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        self._save()

    def _save(self, ended: bool = False):
        data     = self._load_raw()
        sessions = data.setdefault("sessions", [])
        entry = next((s for s in sessions if s.get("session_id") == self._session_id), None)
        if entry is None:
            entry = {
                "session_id": self._session_id,
                "started":    self._session_start.isoformat(timespec="seconds"),
                "ended":      None, "turn_count": 0, "messages": [],
            }
            sessions.append(entry)

        entry["messages"]   = self._current_session_msgs
        entry["turn_count"] = len([m for m in self._current_session_msgs if m["role"] == "user"])
        if ended:
            entry["ended"] = datetime.now().isoformat(timespec="seconds")

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

    def _load_recent_anchor(self, recall_turns: int):
        """Inject the last few turns of the previous session for conversational continuity."""
        data = self._load_raw()
        past = [s for s in data.get("sessions", []) if s.get("session_id") != self._session_id]
        if not past:
            return

        recalled: list[dict] = []
        for session in reversed(past):
            recalled = session.get("messages", []) + recalled
            if len([m for m in recalled if m["role"] == "user"]) >= recall_turns:
                break
        recalled = recalled[-(recall_turns * 2):]
        if not recalled:
            return

        lines = [
            f"{'User' if m['role'] == 'user' else 'Sterling'}: {m['content']}"
            for m in recalled
        ]
        self._history.append({
            "role": "system",
            "content": (
                "The exchange below is from a PREVIOUS session that already ended. "
                "Background only — do not continue it as the current topic.\n\n"
                + "\n".join(lines)
            ),
        })
        logger.info(f"Memory: anchored {len(recalled)} message(s) from last session.")

    def _trim(self):
        """Drop oldest turns, but never drop a leading system anchor."""
        # Count non-system messages
        body = [m for m in self._history if m["role"] != "system"]
        if len(body) <= self._max_messages:
            return
        # Keep system anchors + the most recent max_messages body turns
        keep_body = set(id(m) for m in body[-self._max_messages:])
        self._history = [
            m for m in self._history
            if m["role"] == "system" or id(m) in keep_body
        ]
