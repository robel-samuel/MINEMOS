"""Published JSON Schema for ``minemos.resolution/1`` plus a light validator.

The schema document is the contract; :func:`validate_record` in
:mod:`minemos.resolution.contracts` enforces the same rules over the dataclass
(including span bounds, which JSON Schema cannot express here).

No third-party validator is required: :func:`validate_with_schema` checks the
structure that matters to an external consumer — required keys, closed
objects, enumerations and value types — and raises
:class:`ResolutionContractError` on the first violation.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    ATTRIBUTION_STATUSES,
    IDENTITY_TRANSITION_KINDS,
    RELATIONSHIP_LABELS,
    STAGE_NAMES,
    SUBJECT_KINDS,
    ResolutionContractError,
)

SCHEMA_PATH = Path(__file__).with_name("schema.json")


@lru_cache(maxsize=1)
def load_schema() -> Mapping[str, Any]:
    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


SCHEMA = load_schema()

_REQUIRED = ("version", "source", "original_text", "normalized", "language",
             "entities", "facts", "attributions", "relationships",
             "stage_confidence", "ambiguities", "abstentions", "provider")


def _require(document: Mapping[str, Any], key: str, kind: type,
             label: str) -> Any:
    if key not in document:
        raise ResolutionContractError(f"{label} is missing {key!r}")
    value = document[key]
    if not isinstance(value, kind):
        raise ResolutionContractError(
            f"{label}.{key} must be {kind.__name__}")
    return value


def validate_with_schema(document: Mapping[str, Any]) -> None:
    """Validates the JSON shape of a record produced by ``to_dict``."""
    if not isinstance(document, Mapping):
        raise ResolutionContractError("record must be a mapping")
    unknown = set(document) - set(SCHEMA["properties"])
    if unknown:
        raise ResolutionContractError(
            f"unknown top-level properties: {sorted(unknown)}")
    for key in _REQUIRED:
        if key not in document:
            raise ResolutionContractError(
                f"record is missing required property {key!r}")
    if document["version"] != SCHEMA["properties"]["version"]["const"]:
        raise ResolutionContractError(
            f"unexpected record version {document['version']!r}")
    _require(document, "source", Mapping, "record")
    _require(document, "normalized", Mapping, "record")
    _require(document, "language", Mapping, "record")
    _require(document, "stage_confidence", Mapping, "record")
    _require(document, "provider", str, "record")
    _require(document, "original_text", str, "record")
    if not document["original_text"].strip():
        raise ResolutionContractError("original_text cannot be empty")
    for key in ("entities", "facts", "attributions", "relationships"):
        items = _require(document, key, list, "record")
        if not all(isinstance(item, Mapping) for item in items):
            raise ResolutionContractError(f"{key} must hold objects")
    if not document["attributions"]:
        raise ResolutionContractError(
            "attributions must contain at least one entry")
    for key in ("ambiguities", "abstentions"):
        items = _require(document, key, list, "record")
        if not all(isinstance(item, str) and item for item in items):
            raise ResolutionContractError(
                f"{key} must hold non-empty strings")

    stages = document["stage_confidence"]
    if set(stages) != set(STAGE_NAMES):
        raise ResolutionContractError(
            f"stage_confidence keys must be {sorted(STAGE_NAMES)}")
    for name, value in stages.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ResolutionContractError(f"stage_confidence.{name} is not "
                                          "numeric")
        if not 0.0 <= float(value) <= 1.0:
            raise ResolutionContractError(
                f"stage_confidence.{name} is outside [0, 1]")

    for fact in document["facts"]:
        subject = fact.get("subject")
        if not isinstance(subject, Mapping):
            raise ResolutionContractError("fact.subject must be an object")
        if subject.get("kind") not in SUBJECT_KINDS:
            raise ResolutionContractError("fact.subject.kind is not allowed")
        if subject.get("status") not in ATTRIBUTION_STATUSES:
            raise ResolutionContractError("fact.subject.status is not allowed")
        if not fact.get("evidence"):
            raise ResolutionContractError(
                "every fact needs at least one evidence entry")

    for attribution in document["attributions"]:
        if attribution.get("status") not in ATTRIBUTION_STATUSES:
            raise ResolutionContractError(
                "attribution.status is not allowed")
        speaker = attribution.get("speaker")
        if not isinstance(speaker, Mapping) or not all(
                key in speaker for key in ("id", "label", "confidence",
                                           "source")):
            raise ResolutionContractError(
                "attribution.speaker is incomplete")
        scope = attribution.get("scope", "conversation")
        if scope not in ("message", "conversation", "cross_conversation"):
            raise ResolutionContractError("attribution.scope is not allowed")

    for relationship in document["relationships"]:
        if relationship.get("label") not in RELATIONSHIP_LABELS:
            raise ResolutionContractError("relationship.label is not allowed")

    for transition in document.get("transitions", []):
        if transition.get("kind") not in IDENTITY_TRANSITION_KINDS:
            raise ResolutionContractError("transition.kind is not allowed")
        if not transition.get("message_id"):
            raise ResolutionContractError(
                "transition.message_id is required")


__all__ = ["SCHEMA", "SCHEMA_PATH", "load_schema", "validate_with_schema"]
