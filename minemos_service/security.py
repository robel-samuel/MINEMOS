"""Authentication, authorization, request validation and safe logging.

Everything here is stdlib-only and independent of the memory engine.  The
service treats every caller as untrusted until an API key resolves to a
principal, and every tenant boundary is derived from that principal, never
from request content.
"""

from __future__ import annotations

import hmac
import json
import logging
import math
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .config import (
    ApiKeySpec,
    ServiceConfig,
    ServiceConfigurationError,
    SOURCE_TYPES,
    validate_tenant_id,
)

# ── Authentication ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Principal:
    tenant: str
    key_id: str
    roles: Tuple[str, ...]

    def can(self, role: str) -> bool:
        return "admin" in self.roles or role in self.roles

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tenant": self.tenant,
            "key_id": self.key_id,
            "roles": list(self.roles),
        }


class ApiKeyRegistry:
    """Constant-time API key verification.

    Every configured key is compared with :func:`hmac.compare_digest` and the
    loop deliberately does not short-circuit on a match, so response timing
    does not reveal which key or position matched.
    """

    def __init__(self, specs: Iterable[ApiKeySpec]):
        entries = tuple(specs)
        seen_ids: set[str] = set()
        seen_keys: set[str] = set()
        for spec in entries:
            if spec.key_id in seen_ids:
                raise ServiceConfigurationError(
                    f"duplicate api key id {spec.key_id!r}")
            if spec.key in seen_keys:
                raise ServiceConfigurationError(
                    "duplicate api key material; every key must be unique")
            seen_ids.add(spec.key_id)
            seen_keys.add(spec.key)
        self._entries = entries

    @classmethod
    def from_config(cls, config: ServiceConfig) -> "ApiKeyRegistry":
        return cls(config.api_keys)

    @property
    def configured(self) -> bool:
        return bool(self._entries)

    def tenants(self) -> Tuple[str, ...]:
        return tuple(sorted({spec.tenant for spec in self._entries}))

    def authenticate(self, authorization: Optional[str]) -> Optional[Principal]:
        if not isinstance(authorization, str):
            return None
        parts = authorization.strip().split(None, 1)
        if len(parts) != 2 or parts[0].strip().lower() != "bearer":
            return None
        token = parts[1].strip()
        if not token or len(token) > 1024:
            return None
        matched: Optional[ApiKeySpec] = None
        token_bytes = token.encode("utf-8")
        for spec in self._entries:
            candidate = spec.key.encode("utf-8")
            same_length = len(candidate) == len(token_bytes)
            padded = token_bytes if same_length else b""
            equal = hmac.compare_digest(candidate, padded) if same_length else False
            if equal and matched is None:
                matched = spec
        if matched is None:
            return None
        return Principal(
            tenant=matched.tenant,
            key_id=matched.key_id,
            roles=tuple(matched.roles),
        )


# ── Rate limiting ──────────────────────────────────────────────────────────


class RateLimiter:
    """Thread-safe token bucket per credential.

    ``rate_per_minute`` refills continuously and ``burst`` bounds a short
    spike.  A rejected request returns the whole seconds to wait so the server
    can emit ``Retry-After`` without exposing internal counters.
    """

    def __init__(self, rate_per_minute: int, burst: int,
                 clock=time.monotonic):
        if int(rate_per_minute) < 1 or int(burst) < 1:
            raise ValueError("rate limits must be positive")
        self.rate_per_second = float(rate_per_minute) / 60.0
        self.burst = float(burst)
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: Dict[str, Tuple[float, float]] = {}

    def acquire(self, key: str, cost: float = 1.0) -> float:
        """Consumes ``cost``; returns 0.0 when allowed or seconds to wait."""
        if cost <= 0:
            return 0.0
        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.burst, now))
            elapsed = max(0.0, now - last)
            tokens = min(self.burst, tokens + elapsed * self.rate_per_second)
            if tokens >= cost:
                self._buckets[key] = (tokens - cost, now)
                return 0.0
            deficit = cost - tokens
            self._buckets[key] = (tokens, now)
            return math.ceil(deficit / self.rate_per_second)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "bucket_count": len(self._buckets),
                "rate_per_minute": int(round(self.rate_per_second * 60)),
                "burst": int(self.burst),
            }


# ── Sensitive content ──────────────────────────────────────────────────────

_CREDENTIAL_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("jwt", re.compile(
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\."
        r"[A-Za-z0-9_-]{10,}\b")),
    ("bearer_token", re.compile(
        r"\bBearer\s+[A-Za-z0-9\-._~+/]{20,}=*", re.IGNORECASE)),
    ("api_key_literal", re.compile(
        r"\b(?:sk|pk|rk)[-_](?:live|test)?[-_]?[A-Za-z0-9]{16,}\b")),
    ("credential_assignment", re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|pwd)\b"
        r"\s*[:=]\s*[\"']?[^\s\"']{8,}")),
)

REDACTED = "[REDACTED]"


def find_sensitive(text: str) -> Tuple[str, ...]:
    """Returns distinct credential/PII-relevant pattern names, never values."""
    if not isinstance(text, str) or not text:
        return ()
    kinds = []
    for name, pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(text):
            kinds.append(name)
    return tuple(kinds)


def redact_sensitive(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    out = text
    for name, pattern in _CREDENTIAL_PATTERNS:
        out = pattern.sub(f"{REDACTED}:{name}", out)
    return out


# ── Request validation ─────────────────────────────────────────────────────

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MEMORY_ID_RE = re.compile(r"^[A-Za-z0-9-]{1,128}$")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class RequestError(Exception):
    """A client-correctable error with a stable, non-sensitive code."""

    def __init__(self, status: int, code: str, message: str,
                 headers: Optional[Mapping[str, str]] = None):
        super().__init__(message)
        self.status = int(status)
        self.code = code
        self.message = safe_message(message)
        self.headers = dict(headers or {})


def safe_message(message: Any, limit: int = 300) -> str:
    text = str(message or "").replace("\r", " ").replace("\n", " ")
    text = _CONTROL_RE.sub("", text).strip()
    return text[:limit]


def _reject_unknown(payload: Mapping[str, Any], allowed: Iterable[str],
                    label: str = "request") -> None:
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise RequestError(
            400, "invalid_request",
            f"{label} has unsupported fields: {', '.join(unknown)}")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RequestError(400, "invalid_request", f"{label} must be an object")
    return value


def _require_str(payload: Mapping[str, Any], name: str, max_len: int,
                 *, required: bool = True, allow_empty: bool = False,
                 label: Optional[str] = None) -> Optional[str]:
    value = payload.get(name)
    if value is None:
        if required:
            raise RequestError(
                400, "invalid_request", f"{label or name} is required")
        return None
    if not isinstance(value, str):
        raise RequestError(
            400, "invalid_request", f"{label or name} must be a string")
    if not allow_empty and not value.strip():
        raise RequestError(
            400, "invalid_request", f"{label or name} must be non-empty")
    if len(value) > max_len:
        raise RequestError(
            400, "invalid_request",
            f"{label or name} exceeds {max_len} characters")
    if _CONTROL_RE.search(value):
        raise RequestError(
            400, "invalid_request",
            f"{label or name} contains control characters")
    return value


def _optional_float(payload: Mapping[str, Any], name: str) -> Optional[float]:
    value = payload.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError(400, "invalid_request", f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise RequestError(400, "invalid_request", f"{name} must be finite")
    return value


def _validate_timestamp(value: Optional[float], name: str = "timestamp") -> Optional[float]:
    if value is None:
        return None
    if value < -62135596800.0 or value > 253402300799.999:
        raise RequestError(
            400, "invalid_request",
            f"{name} is outside the supported Unix timestamp range")
    return value


def _validate_source(payload: Mapping[str, Any]) -> Dict[str, Any]:
    source = _require_mapping(payload.get("source"), "source")
    _reject_unknown(
        source,
        {"source_id", "source_type", "document_id", "conversation_id",
         "segment_index", "original_timestamp"},
        label="source")
    source_id = _require_str(source, "source_id", 128, label="source.source_id")
    if not _ID_RE.match(source_id):
        raise RequestError(
            400, "invalid_request",
            "source.source_id must match ^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    source_type = _require_str(
        source, "source_type", 64, required=False, label="source.source_type"
    ) or "observation"
    if source_type not in SOURCE_TYPES:
        raise RequestError(
            400, "invalid_request",
            f"source.source_type must be one of {', '.join(SOURCE_TYPES)}")
    out: Dict[str, Any] = {
        "source_id": source_id,
        "source_type": source_type,
    }
    for field in ("document_id", "conversation_id"):
        value = _require_str(source, field, 128, required=False,
                             label=f"source.{field}")
        if value is not None:
            if not _ID_RE.match(value):
                raise RequestError(
                    400, "invalid_request",
                    f"source.{field} has an invalid identifier format")
            out[field] = value
    segment_index = source.get("segment_index")
    if segment_index is not None:
        if (isinstance(segment_index, bool)
                or not isinstance(segment_index, int)
                or segment_index < 0
                or segment_index > 2**31):
            raise RequestError(
                400, "invalid_request",
                "source.segment_index must be a non-negative integer")
        out["segment_index"] = segment_index
    original_timestamp = _optional_float(source, "original_timestamp")
    out["original_timestamp"] = _validate_timestamp(
        original_timestamp, "source.original_timestamp")
    return out


def _scan_sensitive(values: Iterable[str], config: ServiceConfig) -> List[str]:
    kinds: List[str] = []
    for value in values:
        kinds.extend(find_sensitive(value))
    if not kinds:
        return []
    if config.credential_policy == "reject":
        raise RequestError(
            422, "sensitive_content",
            "observation appears to contain credentials or secret material; "
            "remove it before ingestion")
    return sorted(set(kinds))


def validate_ingest_payload(payload: Any,
                            config: ServiceConfig) -> Dict[str, Any]:
    payload = _require_mapping(payload, "request body")
    _reject_unknown(payload, {"observations", "source"}, label="request body")
    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise RequestError(
            400, "invalid_request", "observations must be a non-empty list")
    if len(observations) > config.max_observations_per_request:
        raise RequestError(
            413, "payload_too_large",
            f"at most {config.max_observations_per_request} observations "
            "are accepted per request")
    default_source = payload.get("source")
    validated: List[Dict[str, Any]] = []
    for index, item in enumerate(observations):
        label = f"observations[{index}]"
        item = _require_mapping(item, label)
        _reject_unknown(
            item,
            {"subject", "predicate", "object", "timestamp", "confidence",
             "source", "ingestion_id"},
            label=label)
        subject = _require_str(item, "subject", 256, label=f"{label}.subject")
        predicate = _require_str(
            item, "predicate", 128, label=f"{label}.predicate")
        obj = _require_str(
            item, "object", config.max_object_chars, label=f"{label}.object")
        timestamp = _validate_timestamp(
            _optional_float(item, "timestamp"), f"{label}.timestamp")
        confidence = _optional_float(item, "confidence")
        if confidence is None:
            confidence = 1.0
        if not 0.0 <= confidence <= 1.0:
            raise RequestError(
                400, "invalid_request",
                f"{label}.confidence must be between 0 and 1")
        ingestion_id = _require_str(
            item, "ingestion_id", 128, required=False,
            label=f"{label}.ingestion_id")
        if ingestion_id is not None and not _ID_RE.match(ingestion_id):
            raise RequestError(
                400, "invalid_request",
                f"{label}.ingestion_id has an invalid format")
        source_payload = item.get("source", default_source)
        if source_payload is None:
            raise RequestError(
                400, "invalid_request",
                f"{label}.source is required (top-level or per-observation)")
        source = _validate_source({"source": source_payload})

        # Identifiers are never rewritten: a credential in an identifier is
        # always rejected, because redacting it would silently change which
        # source the memory is attributed to.
        identifier_kinds: List[str] = []
        for value in (source["source_id"], source.get("document_id"),
                      source.get("conversation_id")):
            if value:
                identifier_kinds.extend(find_sensitive(value))
        if identifier_kinds:
            raise RequestError(
                422, "sensitive_content",
                "source identifiers appear to contain credentials; use "
                "opaque identifiers instead")

        scans = _scan_sensitive([subject, predicate, obj], config)
        if config.credential_policy == "redact":
            subject = redact_sensitive(subject)
            predicate = redact_sensitive(predicate)
            obj = redact_sensitive(obj)

        validated.append({
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "timestamp": timestamp,
            "confidence": confidence,
            "ingestion_id": ingestion_id,
            "source": source,
            "redacted_kinds": scans,
        })
    return {"observations": validated}


def validate_query_payload(payload: Any, config: ServiceConfig) -> Dict[str, Any]:
    payload = _require_mapping(payload, "request body")
    _reject_unknown(payload, {"query", "top_k", "as_of"}, label="request body")
    query = _require_str(payload, "query", config.max_query_chars)
    top_k = payload.get("top_k", 5)
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise RequestError(400, "invalid_request", "top_k must be an integer")
    if not 1 <= top_k <= 100:
        raise RequestError(
            400, "invalid_request", "top_k must be between 1 and 100")
    as_of = _validate_timestamp(_optional_float(payload, "as_of"), "as_of")
    return {"query": query, "top_k": top_k, "as_of": as_of}


def validate_history_payload(payload: Any,
                             config: ServiceConfig) -> Dict[str, Any]:
    payload = _require_mapping(payload, "request body")
    _reject_unknown(payload, {"subject", "predicate"}, label="request body")
    subject = _require_str(payload, "subject", 256)
    predicate = _require_str(payload, "predicate", 128, required=False)
    return {"subject": subject, "predicate": predicate}


def validate_memory_payload(payload: Any, config: ServiceConfig) -> Dict[str, Any]:
    payload = _require_mapping(payload, "request body")
    _reject_unknown(payload, {"memory_id"}, label="request body")
    memory_id = _require_str(payload, "memory_id", 128)
    if not _MEMORY_ID_RE.match(memory_id):
        raise RequestError(
            400, "invalid_request", "memory_id has an invalid format")
    return {"memory_id": memory_id}


def validate_source_payload(payload: Any, config: ServiceConfig) -> Dict[str, Any]:
    payload = _require_mapping(payload, "request body")
    _reject_unknown(payload, {"source_id"}, label="request body")
    source_id = _require_str(payload, "source_id", 128)
    if not _ID_RE.match(source_id):
        raise RequestError(
            400, "invalid_request", "source_id has an invalid format")
    return {"source_id": source_id}


def validate_checkpoint_payload(payload: Any,
                                config: ServiceConfig) -> Dict[str, Any]:
    if payload is None:
        return {}
    payload = _require_mapping(payload, "request body")
    _reject_unknown(payload, set(), label="request body")
    return {}


# ── Logging ────────────────────────────────────────────────────────────────


class JsonLogFormatter(logging.Formatter):
    """One-line JSON logs with request context and credential scrubbing."""

    _RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
        "message", "asctime", "taskName"}

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(
                record.created, tz=timezone.utc).isoformat(
                    timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_sensitive(record.getMessage()),
        }
        for name, value in record.__dict__.items():
            if name in self._RESERVED or name.startswith("_"):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[name] = redact_sensitive(value) if isinstance(
                    value, str) else value
            elif isinstance(value, (list, tuple)) and all(
                    isinstance(v, (str, int, float, bool)) for v in value):
                payload[name] = [
                    redact_sensitive(v) if isinstance(v, str) else v
                    for v in value]
        if record.exc_info:
            payload["exception"] = redact_sensitive(
                self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(level: str = "INFO",
                      stream=None) -> logging.Logger:
    logger = logging.getLogger("minemos_service")
    logger.setLevel(str(level).upper())
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonLogFormatter())
    logger.handlers[:] = [handler]
    logger.propagate = False
    return logger


def request_id_from(header: Optional[str]) -> str:
    if isinstance(header, str) and re.match(r"^[A-Za-z0-9._-]{1,128}$", header):
        return header
    return uuid.uuid4().hex
