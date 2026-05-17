"""
Sterling Text Utilities
Cleans LLM output so it sounds natural when read aloud by TTS.
Strips markdown, special characters, and other TTS-unfriendly artifacts.
"""

import re


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def clean_for_tts(text: str) -> str:
    """
    Remove markdown and formatting artifacts from text before passing to TTS.
    Returns clean, speakable prose.
    """
    text = _strip_code_blocks(text)
    text = _strip_inline_code(text)
    text = _strip_markdown_formatting(text)
    text = _strip_markdown_headers(text)
    text = _convert_lists_to_prose(text)
    text = _strip_urls(text)
    text = _normalize_whitespace(text)
    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into individual sentences for streaming TTS.
    Tries to avoid splitting on common abbreviations.
    """
    # Simple sentence boundary detection
    # Split on . ! ? followed by whitespace and capital letter, or end of string
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(])', text)
    return [p.strip() for p in parts if p.strip()]


def truncate_for_display(text: str, max_chars: int = 80) -> str:
    """Truncate text for console display with an ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_code_blocks(text: str) -> str:
    """Replace code blocks with a brief verbal placeholder."""
    # ```lang\n...\n``` → "[code block]"
    text = re.sub(r"```[\w]*\n.*?```", "[code block]", text, flags=re.DOTALL)
    text = re.sub(r"~~~[\w]*\n.*?~~~", "[code block]", text, flags=re.DOTALL)
    return text


def _strip_inline_code(text: str) -> str:
    """Strip backtick inline code markers but keep the content."""
    return re.sub(r"`([^`]+)`", r"\1", text)


def _strip_markdown_formatting(text: str) -> str:
    """Remove bold, italic, strikethrough markers."""
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)   # ***bold italic***
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)        # **bold**
    text = re.sub(r"\*(.+?)\*", r"\1", text)             # *italic*
    text = re.sub(r"___(.+?)___", r"\1", text)           # ___bold italic___
    text = re.sub(r"__(.+?)__", r"\1", text)             # __bold__
    text = re.sub(r"_(.+?)_", r"\1", text)               # _italic_
    text = re.sub(r"~~(.+?)~~", r"\1", text)             # ~~strikethrough~~
    return text


def _strip_markdown_headers(text: str) -> str:
    """Remove # heading markers."""
    return re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)


def _convert_lists_to_prose(text: str) -> str:
    """
    Convert markdown bullet / numbered list items to comma-separated prose
    or simple sentence structure.
    """
    lines = text.split("\n")
    result = []
    list_items = []
    in_list = False

    for line in lines:
        # Detect list item: - item, * item, 1. item, 1) item
        list_match = re.match(r"^\s*[-*•]\s+(.+)$", line)
        numbered_match = re.match(r"^\s*\d+[.)]\s+(.+)$", line)

        if list_match or numbered_match:
            item = (list_match or numbered_match).group(1).strip()
            list_items.append(item)
            in_list = True
        else:
            if in_list and list_items:
                # Flush the accumulated list items as prose
                result.append(_list_items_to_prose(list_items))
                list_items = []
                in_list = False
            result.append(line)

    # Flush any remaining list items
    if list_items:
        result.append(_list_items_to_prose(list_items))

    return "\n".join(result)


def _list_items_to_prose(items: list[str]) -> str:
    """Convert a list of strings to a natural spoken sentence."""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]}, and {items[1]}."
    body = ", ".join(items[:-1])
    return f"{body}, and {items[-1]}."


def _strip_urls(text: str) -> str:
    """Replace URLs with a brief placeholder."""
    return re.sub(r"https?://\S+", "[link]", text)


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple blank lines and normalize spacing."""
    # Collapse 2+ newlines → single newline (one paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace remaining single newlines with spaces (for flow)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text
