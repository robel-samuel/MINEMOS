"""Host-side segmentation for long messages.

The resolver reads one message at a time and never truncates text.  A host
that accepts long messages therefore has to split them before resolution so
every message handed to :class:`~minemos.resolution.pipeline.Resolver` fits the
service's 8,192-character segment cap.  :func:`segment_message` performs that
split and returns chunks whose offsets address the original message, so a
consumer can always map a record back to its source text.

The split prefers sentence boundaries, then newlines, and hard-cuts only when
neither exists.  Every non-whitespace character of the original message ends up
in exactly one chunk; whitespace between chunks is skipped, not dropped from
the source, because ``char_start``/``char_end`` are offsets into the original
text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

MAX_MESSAGE_CHARS = 8192

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class MessageChunk:
    """One resolution-sized piece of a longer source message."""

    message_id: str
    text: str
    char_start: int
    char_end: int
    chunk_index: int
    chunk_count: int

    def __post_init__(self) -> None:
        if not self.message_id:
            raise ValueError("chunk message_id is required")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("chunk text must be a non-empty string")
        if self.char_start < 0 or self.char_end <= self.char_start:
            raise ValueError("chunk offsets must describe a non-empty range")
        if self.chunk_count < 1 or not 0 <= self.chunk_index < self.chunk_count:
            raise ValueError("chunk index must fall inside chunk_count")

    @property
    def identity(self) -> str:
        """Stable chunk identity; the first chunk keeps the message id."""
        if self.chunk_count == 1:
            return self.message_id
        return f"{self.message_id}#{self.chunk_index}"


def _split_block(text: str, offset: int, max_chars: int
                 ) -> List[Tuple[str, int, int]]:
    pieces: List[Tuple[str, int, int]] = []
    position = 0
    while position < len(text):
        window_end = min(len(text), position + max_chars)
        if window_end == len(text):
            end = len(text)
        else:
            split_at = -1
            for sentence in _SENTENCE_SPLIT.finditer(
                    text[position:window_end]):
                split_at = position + sentence.end()
            if split_at <= position:
                split_at = text.rfind("\n", position, window_end)
            if split_at <= position:
                split_at = window_end
            end = split_at
        piece = text[position:end]
        if piece.strip():
            pieces.append((piece, offset + position, offset + end))
        position = end
        while position < len(text) and text[position].isspace():
            position += 1
    return pieces


def segment_message(text: str, message_id: str,
                    max_chars: int = MAX_MESSAGE_CHARS
                    ) -> Tuple[MessageChunk, ...]:
    """Splits one message into chunks of at most ``max_chars`` characters.

    Returns an empty tuple for blank text.  ``char_start``/``char_end`` index
    into ``text``, so ``text[chunk.char_start:chunk.char_end] == chunk.text``.
    A message that already fits is returned as a single chunk, and that chunk's
    ``identity`` is exactly ``message_id``.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(message_id, str) or not message_id.strip():
        raise ValueError("message_id must be a non-empty string")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise TypeError("max_chars must be an integer")
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if not text.strip():
        return ()
    pieces = _split_block(text, 0, max_chars)
    count = len(pieces)
    return tuple(
        MessageChunk(
            message_id=message_id,
            text=piece,
            char_start=start,
            char_end=end,
            chunk_index=index,
            chunk_count=count,
        )
        for index, (piece, start, end) in enumerate(pieces)
    )


__all__ = ["MAX_MESSAGE_CHARS", "MessageChunk", "segment_message"]
