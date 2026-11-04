"""Real-source parsing for the evaluation environment.

The engine consumes structured observations, not arbitrary documents.  This
module therefore preserves *source structure* explicitly: every segment keeps
its source, document or conversation boundary, speaker (when present), ordinal
position, character offsets and original timestamp.  Extraction into
observations is a separate, explicit step (verbatim segment memories or an
authorized upstream structured extractor).  No template heuristic is presented
as general-purpose fact extraction.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from minemos_service.config import SOURCE_TYPES, validate_tenant_id

SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
DOCUMENT_ID_RE = SOURCE_ID_RE
_PARAGRAPH_SPLIT = re.compile(r"\n[ \t]*\n+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HEADING = re.compile(r"^#{1,6}\s+\S")


class SourceFormatError(ValueError):
    """A source file does not match its declared format."""


@dataclass(frozen=True)
class Observation:
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    timestamp: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("subject", "predicate", "object"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SourceFormatError(f"observation {name} must be non-empty")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise SourceFormatError("observation confidence must be in [0, 1]")
        if self.timestamp is not None:
            value = float(self.timestamp)
            if value != value or value in (float("inf"), float("-inf")):
                raise SourceFormatError("observation timestamp must be finite")
            object.__setattr__(self, "timestamp", value)
        object.__setattr__(self, "confidence", float(self.confidence))


@dataclass(frozen=True)
class Segment:
    source_id: str
    source_type: str
    document_id: str
    segment_index: int
    text: str
    original_timestamp: Optional[float] = None
    conversation_id: Optional[str] = None
    speaker: Optional[str] = None
    title: Optional[str] = None
    char_start: int = 0
    char_end: int = 0
    observations: Tuple[Observation, ...] = ()

    def __post_init__(self) -> None:
        if not SOURCE_ID_RE.match(self.source_id):
            raise SourceFormatError("source_id has an invalid format")
        if not DOCUMENT_ID_RE.match(self.document_id):
            raise SourceFormatError("document_id has an invalid format")
        if self.source_type not in SOURCE_TYPES:
            raise SourceFormatError(f"unknown source_type {self.source_type!r}")
        if self.segment_index < 0:
            raise SourceFormatError("segment_index must be non-negative")
        if not isinstance(self.text, str) or not self.text.strip():
            raise SourceFormatError("segment text must be non-empty")
        if self.original_timestamp is not None:
            value = float(self.original_timestamp)
            if value != value or value in (float("inf"), float("-inf")):
                raise SourceFormatError("segment timestamp must be finite")
            object.__setattr__(self, "original_timestamp", value)

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def segment_key(self) -> str:
        return f"{self.source_id}:{self.document_id}:{self.segment_index}"

    @property
    def timestamp_inferred(self) -> bool:
        return self.original_timestamp is None


def _validate_timestamp(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SourceFormatError(f"{label} must be a number")
    return float(value)


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceFormatError(f"{label} must be a non-empty string")
    return value


def _resolve_source_id(line: Dict[str, Any], fallback: Optional[str],
                       path: Path) -> str:
    value = line.get("source_id") or fallback or path.stem
    value = str(value).strip()
    if not SOURCE_ID_RE.match(value):
        raise SourceFormatError(
            f"source_id {value!r} must match {SOURCE_ID_RE.pattern}")
    return value


def split_text(text: str, max_chars: int = 2000
               ) -> List[Tuple[str, int, int]]:
    """Splits text at paragraph boundaries, then sentence boundaries.

    Returns ``(segment_text, char_start, char_end)`` with offsets into the
    original document, so a segment never loses its position in the source.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    chunks: List[Tuple[str, int, int]] = []
    cursor = 0
    for match in list(_PARAGRAPH_SPLIT.finditer(text)) + [None]:
        end = match.start() if match else len(text)
        block = text[cursor:end]
        if block.strip():
            chunks.extend(_split_block(block, cursor, max_chars))
        if match is not None:
            cursor = match.end()
    return chunks


def _split_block(block: str, offset: int, max_chars: int
                 ) -> List[Tuple[str, int, int]]:
    if len(block) <= max_chars:
        return [(block, offset, offset + len(block))]
    pieces: List[Tuple[str, int, int]] = []
    position = 0
    while position < len(block):
        window_end = min(len(block), position + max_chars)
        if window_end == len(block):
            end = len(block)
        else:
            split_at = -1
            for sentence in _SENTENCE_SPLIT.finditer(
                    block[position:window_end]):
                split_at = position + sentence.end()
            if split_at <= position:
                split_at = block.rfind("\n", position, window_end)
            if split_at <= position:
                split_at = window_end
            end = split_at
        piece = block[position:end]
        if piece.strip():
            pieces.append((piece, offset + position, offset + end))
        position = end
        while position < len(block) and block[position].isspace():
            position += 1
    return pieces


def _parse_conversation_line(
    line: Dict[str, Any], source_id: str, index: int,
    conversation_counters: Dict[str, int],
) -> List[Segment]:
    conversation_id = str(
        line.get("conversation_id") or line.get("conversation") or "").strip()
    if not conversation_id:
        raise SourceFormatError(
            "conversation sources require conversation_id")
    if not DOCUMENT_ID_RE.match(conversation_id):
        raise SourceFormatError("conversation_id has an invalid format")
    line_timestamp = _validate_timestamp(
        line.get("timestamp"), "conversation timestamp")
    messages = line.get("messages")
    if messages is None:
        messages = [line]
    elif not isinstance(messages, list):
        raise SourceFormatError("messages must be a list")
    out: List[Segment] = []
    for message in messages:
        if not isinstance(message, dict):
            raise SourceFormatError("each message must be an object")
        text = message.get("content", message.get("text"))
        text = _require_text(text, "message content")
        speaker = message.get("role") or message.get("speaker")
        timestamp = _validate_timestamp(
            message.get("timestamp"), "message timestamp")
        if timestamp is None:
            timestamp = line_timestamp
        segment_index = conversation_counters.get(conversation_id, 0)
        conversation_counters[conversation_id] = segment_index + 1
        out.append(Segment(
            source_id=source_id,
            source_type="conversation",
            document_id=conversation_id,
            conversation_id=conversation_id,
            segment_index=segment_index,
            speaker=str(speaker) if speaker is not None else None,
            text=text,
            original_timestamp=timestamp,
            char_start=0,
            char_end=len(text),
        ))
    return out


def _parse_document_line(line: Dict[str, Any], source_id: str,
                         max_chars: int) -> List[Segment]:
    timestamp = _validate_timestamp(
        line.get("timestamp"), "document timestamp")
    title = line.get("title")
    title = str(title) if title is not None else None
    pieces = line.get("segments")
    out: List[Segment] = []
    document_id = str(
        line.get("document_id") or line.get("id") or "").strip()
    if not document_id:
        if pieces is None:
            seed = _require_text(line.get("text"), "document text")
        else:
            seed = "\x1f".join(
                str(piece.get("text", piece.get("content")))
                if isinstance(piece, dict) else str(piece)
                for piece in pieces)
        document_id = hashlib.sha256(
            seed.encode("utf-8")).hexdigest()[:16]
    if not DOCUMENT_ID_RE.match(document_id):
        raise SourceFormatError("document_id has an invalid format")
    if pieces is not None:
        if not isinstance(pieces, list):
            raise SourceFormatError("document segments must be a list")
        for index, piece in enumerate(pieces):
            if isinstance(piece, dict):
                text = _require_text(
                    piece.get("text", piece.get("content")),
                    "segment text")
                offset = int(piece.get("char_start", 0))
            else:
                text = _require_text(piece, "segment text")
                offset = 0
            out.append(Segment(
                source_id=source_id,
                source_type="document",
                document_id=document_id,
                segment_index=index,
                text=text,
                original_timestamp=timestamp,
                title=title,
                char_start=offset,
                char_end=offset + len(text),
            ))
        return out
    text = _require_text(line.get("text"), "document text")
    for index, (chunk, start, end) in enumerate(split_text(text, max_chars)):
        out.append(Segment(
            source_id=source_id,
            source_type="document",
            document_id=document_id,
            segment_index=index,
            text=chunk,
            original_timestamp=timestamp,
            title=title,
            char_start=start,
            char_end=end,
        ))
    return out


def _parse_structured_entry(line: Dict[str, Any], source_id: str,
                            document_id: Optional[str]) -> Segment:
    subject = _require_text(line.get("subject"), "subject")
    predicate = _require_text(line.get("predicate"), "predicate")
    obj = _require_text(line.get("object"), "object")
    confidence = line.get("confidence", 1.0)
    timestamp = _validate_timestamp(line.get("timestamp"), "timestamp")
    document_id = str(document_id or line.get("document_id") or source_id)
    if not DOCUMENT_ID_RE.match(document_id):
        raise SourceFormatError("document_id has an invalid format")
    segment_index = line.get("segment_index", line.get("index"))
    if segment_index is None:
        segment_index = line.get("_line", 0)
    if isinstance(segment_index, bool) or not isinstance(segment_index, int):
        raise SourceFormatError("segment_index must be an integer")
    text = f"{subject} {predicate} {obj}."
    observation = Observation(
        subject=subject, predicate=predicate, object=obj,
        confidence=confidence, timestamp=timestamp)
    return Segment(
        source_id=source_id,
        source_type="observation",
        document_id=document_id,
        segment_index=segment_index,
        text=text,
        original_timestamp=timestamp,
        observations=(observation,),
        char_start=0,
        char_end=len(text),
    )


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SourceFormatError(f"cannot read {path}: {exc}") from exc
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SourceFormatError(
                f"{path}: line {number} is not valid JSON: {exc}") from exc
        if not isinstance(entry, dict):
            raise SourceFormatError(
                f"{path}: line {number} must be a JSON object")
        entry.setdefault("_line", len(entries))
        entries.append(entry)
    return entries


def parse_source_file(path: str | Path, source_type: str,
                      source_id: Optional[str] = None,
                      max_chars: int = 2000) -> List[Segment]:
    """Parses one source file into boundary-preserving segments."""
    path = Path(path)
    if not path.is_file():
        raise SourceFormatError(f"source file does not exist: {path}")
    declared = str(source_type).strip().lower()
    if declared in ("jsonl", "json"):
        declared = "document"
    if declared not in SOURCE_TYPES:
        raise SourceFormatError(
            f"source_type must be one of {', '.join(SOURCE_TYPES)}")
    if declared in ("conversation",):
        entries = _read_jsonl(path)
        source = _resolve_source_id(entries[0] if entries else {}, source_id,
                                    path)
        counters: Dict[str, int] = {}
        segments: List[Segment] = []
        for entry in entries:
            segments.extend(_parse_conversation_line(
                entry, source, entry.get("_line", 0), counters))
        return segments
    if declared == "document":
        entries = _read_jsonl(path)
        source = _resolve_source_id(entries[0] if entries else {}, source_id,
                                    path)
        segments = []
        for entry in entries:
            segments.extend(_parse_document_line(entry, source, max_chars))
        return segments
    if declared == "observation":
        entries = _read_jsonl(path)
        source = _resolve_source_id(entries[0] if entries else {}, source_id,
                                    path)
        return [_parse_structured_entry(entry, source, None)
                for entry in entries]
    # Notes, journals, technical documents, research documents and project
    # histories are all treated as prose documents with explicit boundaries.
    source = source_id or path.stem
    if not SOURCE_ID_RE.match(str(source)):
        raise SourceFormatError("source_id has an invalid format")
    text = path.read_text(encoding="utf-8")
    document_id = hashlib.sha256(
        f"{source}\x1f{path.name}".encode("utf-8")).hexdigest()[:24]
    pieces = split_text(text, max_chars)
    return [
        Segment(
            source_id=str(source),
            source_type=declared,
            document_id=document_id,
            segment_index=index,
            text=chunk,
            original_timestamp=None,
            title=path.stem,
            char_start=start,
            char_end=end,
        )
        for index, (chunk, start, end) in enumerate(pieces)
    ]


def load_observation_mapping(path: str | Path) -> Dict[str, List[Observation]]:
    """Loads authorized upstream extraction aligned to segment keys.

    Each line supplies ``segment_key`` plus subject/predicate/object.  This is
    the supported path for real extraction: the extractor runs outside the
    service, its outputs are reviewable, and no heuristic is presented as
    universal NLP.
    """
    entries = _read_jsonl(Path(path))
    mapping: Dict[str, List[Observation]] = {}
    for entry in entries:
        key = entry.get("segment_key")
        if not isinstance(key, str) or not key.strip():
            raise SourceFormatError(
                "observation mapping requires segment_key")
        observation = Observation(
            subject=_require_text(entry.get("subject"), "subject"),
            predicate=_require_text(entry.get("predicate"), "predicate"),
            object=_require_text(entry.get("object"), "object"),
            confidence=entry.get("confidence", 1.0),
            timestamp=_validate_timestamp(
                entry.get("timestamp"), "observation timestamp"),
        )
        mapping.setdefault(key, []).append(observation)
    return mapping


def verbatim_observations(segments: Sequence[Segment]) -> List[Observation]:
    """Turns source segments into retrievable memories without inventing facts.

    Each segment becomes one structured record: the document or conversation
    is the subject, the ordinal position is the predicate, and the verbatim
    segment is the object.  This preserves context and supports retention,
    retrieval and temporal evaluation; it deliberately does not claim semantic
    extraction or contradiction detection.
    """
    observations: List[Observation] = []
    for segment in segments:
        if segment.conversation_id:
            subject = f"conversation:{segment.conversation_id}"
        else:
            subject = f"document:{segment.document_id}"
        observations.append(Observation(
            subject=subject,
            predicate=f"segment_{segment.segment_index}",
            object=segment.text,
            timestamp=segment.original_timestamp,
        ))
    return observations


def observation_segments(segments: Sequence[Segment],
                         mapping: Dict[str, List[Observation]]) -> List[Segment]:
    """Attaches extracted observations to their source segments."""
    out: List[Segment] = []
    missing: List[str] = []
    for segment in segments:
        observations = tuple(mapping.get(segment.segment_key, ()))
        if not observations:
            missing.append(segment.segment_key)
        out.append(Segment(
            source_id=segment.source_id,
            source_type=segment.source_type,
            document_id=segment.document_id,
            segment_index=segment.segment_index,
            text=segment.text,
            original_timestamp=segment.original_timestamp,
            conversation_id=segment.conversation_id,
            speaker=segment.speaker,
            title=segment.title,
            char_start=segment.char_start,
            char_end=segment.char_end,
            observations=observations,
        ))
    if missing and mapping:
        raise SourceFormatError(
            "observation mapping does not cover segment keys: "
            + ", ".join(missing[:5]))
    return out


def order_key(segment: Segment) -> Tuple[int, float, str, int]:
    """Chronological order that keeps untimed sources deterministic."""
    if segment.original_timestamp is None:
        return (1, 0.0, segment.document_id, segment.segment_index)
    return (0, segment.original_timestamp, segment.document_id,
            segment.segment_index)


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(".").strip().casefold()
