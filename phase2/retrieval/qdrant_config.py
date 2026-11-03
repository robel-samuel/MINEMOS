"""Configuration for the optional Qdrant vector-retrieval backend.

This module is stdlib-only so canonical MINEMOS never imports ``qdrant-client``.
Credentials are read from explicit configuration or environment variables and
are deliberately excluded from ``repr`` and from every stats payload.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_COLLECTION = "minemos_memories"
DEFAULT_TENANT = "default"


@dataclass(frozen=True)
class QdrantConfig:
    """Connection and index settings for one MINEMOS tenant.

    Exactly one of ``url`` (remote server) or ``path`` (embedded persistent
    Qdrant) must be supplied.  ``":memory:"`` selects an ephemeral embedded
    instance for tests.
    """

    url: Optional[str] = None
    path: Optional[str] = None
    api_key: Optional[str] = field(default=None, repr=False)
    collection: str = DEFAULT_COLLECTION
    tenant_id: str = DEFAULT_TENANT
    timeout: float = 5.0
    prefer_grpc: bool = False
    grpc_port: Optional[int] = None
    recreate: bool = False
    candidate_limit: int = 128
    batch_size: int = 256
    max_retries: int = 2
    retry_backoff: float = 0.25
    on_disk: bool = False
    ensure_payload_indexes: bool = True

    def __post_init__(self) -> None:
        if bool(self.url) == bool(self.path):
            raise ValueError(
                "QdrantConfig requires exactly one of 'url' (remote) or "
                "'path' (embedded)")
        if not str(self.tenant_id or "").strip():
            raise ValueError("QdrantConfig.tenant_id must be non-empty")
        if not str(self.collection or "").strip():
            raise ValueError("QdrantConfig.collection must be non-empty")
        if float(self.timeout) <= 0:
            raise ValueError("QdrantConfig.timeout must be > 0")
        if int(self.candidate_limit) < 1:
            raise ValueError("QdrantConfig.candidate_limit must be >= 1")
        if int(self.batch_size) < 1:
            raise ValueError("QdrantConfig.batch_size must be >= 1")
        if int(self.max_retries) < 0:
            raise ValueError("QdrantConfig.max_retries must be >= 0")
        if float(self.retry_backoff) < 0:
            raise ValueError("QdrantConfig.retry_backoff must be >= 0")
        object.__setattr__(self, "tenant_id", str(self.tenant_id).strip())
        object.__setattr__(self, "collection", str(self.collection).strip())
        object.__setattr__(self, "timeout", float(self.timeout))
        object.__setattr__(self, "candidate_limit", int(self.candidate_limit))
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "max_retries", int(self.max_retries))
        object.__setattr__(self, "retry_backoff", float(self.retry_backoff))
        object.__setattr__(self, "prefer_grpc", bool(self.prefer_grpc))
        object.__setattr__(self, "recreate", bool(self.recreate))
        object.__setattr__(self, "on_disk", bool(self.on_disk))
        object.__setattr__(self, "ensure_payload_indexes",
                           bool(self.ensure_payload_indexes))
        if self.grpc_port is not None:
            object.__setattr__(self, "grpc_port", int(self.grpc_port))

    @classmethod
    def from_value(cls, value: Any) -> "QdrantConfig":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.from_env()
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(
            f"qdrant_config must be QdrantConfig, dict or None, got {type(value)!r}")

    @classmethod
    def from_env(cls) -> "QdrantConfig":
        """Build config from ``QDRANT_*`` environment variables.

        ``QDRANT_API_KEY`` is consumed but never echoed by stats or logs.
        """
        url = os.environ.get("QDRANT_URL") or None
        path = os.environ.get("QDRANT_PATH") or None
        if not url and not path:
            raise ValueError(
                "Qdrant backend requires qdrant_config={url|path} or the "
                "QDRANT_URL / QDRANT_PATH environment variable")
        return cls(
            url=url,
            path=None if url else path,
            api_key=os.environ.get("QDRANT_API_KEY") or None,
            collection=os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION),
            tenant_id=os.environ.get("QDRANT_TENANT_ID", DEFAULT_TENANT),
            timeout=float(os.environ.get("QDRANT_TIMEOUT", "5.0")),
            prefer_grpc=os.environ.get("QDRANT_PREFER_GRPC", "").lower()
            in ("1", "true", "yes"),
            grpc_port=(
                int(os.environ["QDRANT_GRPC_PORT"])
                if os.environ.get("QDRANT_GRPC_PORT") else None),
            recreate=os.environ.get("QDRANT_RECREATE", "").lower()
            in ("1", "true", "yes"),
            candidate_limit=int(
                os.environ.get("QDRANT_CANDIDATE_LIMIT", "128")),
            batch_size=int(os.environ.get("QDRANT_BATCH_SIZE", "256")),
            max_retries=int(os.environ.get("QDRANT_MAX_RETRIES", "2")),
            retry_backoff=float(os.environ.get("QDRANT_RETRY_BACKOFF", "0.25")),
            on_disk=os.environ.get("QDRANT_ON_DISK", "").lower()
            in ("1", "true", "yes"),
            ensure_payload_indexes=os.environ.get(
                "QDRANT_PAYLOAD_INDEXES", "true").lower()
            not in ("0", "false", "no"),
        )
