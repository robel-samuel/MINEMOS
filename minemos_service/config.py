"""Environment-driven configuration for the MINEMOS service boundary.

Configuration is deliberately explicit.  It never reads a ``.env`` file on
its own and it never invents credentials: deployment tooling is responsible
for supplying secrets through the process environment or a restricted key
file.  ``repr`` and :meth:`ServiceConfig.describe` never print key material.
"""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

ROLES = ("reader", "writer", "admin")
ENVIRONMENTS = ("development", "production")
CREDENTIAL_POLICIES = ("reject", "redact")
SOURCE_TYPES = (
    "conversation",
    "document",
    "notes",
    "journal",
    "technical_doc",
    "research_doc",
    "project_history",
    "observation",
    "other",
)

_TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_KEY_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


class ServiceConfigurationError(ValueError):
    """The service configuration is incomplete or unsafe."""


def validate_tenant_id(value: Any) -> str:
    if not isinstance(value, str) or not _TENANT_RE.match(value.strip()):
        raise ServiceConfigurationError(
            "tenant ids must match ^[a-z0-9][a-z0-9_-]{0,63}$")
    return value.strip()


def is_loopback_host(host: str) -> bool:
    value = (host or "").strip().strip("[]").lower()
    if value in _LOOPBACK_HOSTS:
        return True
    return value.startswith("127.")


def _env_bool(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ServiceConfigurationError(
            f"{name} must be an integer, got {raw!r}") from exc


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ServiceConfigurationError(
            f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class ApiKeySpec:
    """One provisioned API key.  ``key`` is excluded from repr."""

    key_id: str
    key: str = field(repr=False)
    tenant: str
    roles: Tuple[str, ...] = ("reader",)

    def __post_init__(self) -> None:
        if not isinstance(self.key_id, str) or not _KEY_ID_RE.match(
                self.key_id.strip()):
            raise ServiceConfigurationError(
                "api key ids must match ^[A-Za-z0-9._:-]{1,64}$")
        if not isinstance(self.key, str) or not self.key:
            raise ServiceConfigurationError("api key material must be a string")
        validate_tenant_id(self.tenant)
        roles = tuple(str(r).strip().lower() for r in self.roles)
        if not roles:
            raise ServiceConfigurationError(
                f"api key {self.key_id!r} needs at least one role")
        unknown = sorted(set(roles) - set(ROLES))
        if unknown:
            raise ServiceConfigurationError(
                f"api key {self.key_id!r} has unknown roles {unknown}")
        object.__setattr__(self, "key_id", self.key_id.strip())
        object.__setattr__(self, "roles", roles)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key_id": self.key_id,
            "tenant": self.tenant,
            "roles": list(self.roles),
        }


def parse_api_keys(raw: Any) -> Tuple[ApiKeySpec, ...]:
    """Parses ``{"keys": [...]}`` or a bare list of key objects."""
    if isinstance(raw, dict):
        items = raw.get("keys")
    else:
        items = raw
    if not isinstance(items, list):
        raise ServiceConfigurationError(
            "api keys must be a list or an object with a 'keys' list")
    specs: List[ApiKeySpec] = []
    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ServiceConfigurationError("each api key must be an object")
        unknown = set(item) - {"key_id", "key", "tenant", "roles"}
        if unknown:
            raise ServiceConfigurationError(
                f"api key has unknown fields: {sorted(unknown)}")
        try:
            spec = ApiKeySpec(
                key_id=item["key_id"],
                key=item["key"],
                tenant=item["tenant"],
                roles=tuple(item.get("roles", ("reader",))),
            )
        except KeyError as exc:
            raise ServiceConfigurationError(
                f"api key is missing field {exc.args[0]!r}") from exc
        if spec.key_id in seen_ids:
            raise ServiceConfigurationError(
                f"duplicate api key id {spec.key_id!r}")
        if spec.key in seen_keys:
            raise ServiceConfigurationError(
                "duplicate api key material; every key must be unique")
        seen_ids.add(spec.key_id)
        seen_keys.add(spec.key)
        specs.append(spec)
    return tuple(specs)


def _load_key_file(path: Path) -> Tuple[ApiKeySpec, ...]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ServiceConfigurationError(
            f"cannot read api key file {path}: {exc}") from exc
    if os.name == "posix":
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise ServiceConfigurationError(
                f"api key file {path} is accessible by group/other "
                f"(mode {oct(mode)}); restrict it to the service user")
    try:
        return parse_api_keys(json.loads(text))
    except json.JSONDecodeError as exc:
        raise ServiceConfigurationError(
            f"api key file {path} is not valid JSON: {exc}") from exc


def _safe_url(url: Optional[str]) -> Optional[str]:
    """Drops userinfo (which could carry credentials) from a display URL."""
    if not url:
        return url
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return parsed._replace(netloc=netloc).geturl()


@dataclass(frozen=True)
class ServiceConfig:
    """Immutable service configuration.

    ``api_keys`` is excluded from ``repr``.  The engine never receives an API
    key; only the Qdrant client receives ``qdrant_api_key``.
    """

    host: str = "127.0.0.1"
    port: int = 8080
    environment: str = "development"
    data_dir: Path = Path("var/minemos")
    checkpoint_dir: Optional[Path] = None
    provenance_dir: Optional[Path] = None
    tls_cert_file: Optional[Path] = None
    tls_key_file: Optional[Path] = None
    allow_insecure_http: bool = False
    allow_unauthenticated_dev: bool = False
    max_body_bytes: int = 1_048_576
    max_observations_per_request: int = 256
    max_object_chars: int = 8192
    max_query_chars: int = 4096
    request_timeout: float = 30.0
    rate_limit_per_minute: int = 600
    rate_limit_burst: int = 100
    engine_capacity: int = 10_000
    engine_dim: int = 4096
    retrieval_mode: Optional[str] = None
    qdrant_url: Optional[str] = None
    qdrant_path: Optional[str] = None
    qdrant_api_key: Optional[str] = field(default=None, repr=False)
    qdrant_collection_prefix: str = "minemos_memories"
    qdrant_timeout: float = 5.0
    persist_after_mutation: bool = True
    credential_policy: str = "reject"
    log_level: str = "INFO"
    api_keys: Tuple[ApiKeySpec, ...] = field(default=(), repr=False)

    # -- derived paths ---------------------------------------------------- #
    @property
    def resolved_data_dir(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()

    @property
    def resolved_checkpoint_dir(self) -> Path:
        base = self.checkpoint_dir or (self.resolved_data_dir / "checkpoints")
        return Path(base).expanduser().resolve()

    @property
    def resolved_provenance_dir(self) -> Path:
        base = self.provenance_dir or (self.resolved_data_dir / "provenance")
        return Path(base).expanduser().resolve()

    @property
    def production(self) -> bool:
        return self.environment == "production"

    @property
    def tls_enabled(self) -> bool:
        return bool(self.tls_cert_file and self.tls_key_file)

    @property
    def binding_non_loopback(self) -> bool:
        return not is_loopback_host(self.host)

    # -- validation ------------------------------------------------------- #
    def validate(self) -> "ServiceConfig":
        environment = str(self.environment).strip().lower()
        if environment not in ENVIRONMENTS:
            raise ServiceConfigurationError(
                f"environment must be one of {ENVIRONMENTS}")
        object.__setattr__(self, "environment", environment)

        if not isinstance(self.host, str) or not self.host.strip():
            raise ServiceConfigurationError("host must be non-empty")
        if not 0 <= int(self.port) <= 65535:
            raise ServiceConfigurationError("port must be in 0..65535")
        if self.production and int(self.port) == 0:
            raise ServiceConfigurationError(
                "production cannot bind an ephemeral port")
        if int(self.max_body_bytes) < 1024:
            raise ServiceConfigurationError("max_body_bytes must be >= 1024")
        if not 1 <= int(self.max_observations_per_request) <= 10_000:
            raise ServiceConfigurationError(
                "max_observations_per_request must be in 1..10000")
        if int(self.max_object_chars) < 1:
            raise ServiceConfigurationError("max_object_chars must be >= 1")
        if int(self.max_query_chars) < 1:
            raise ServiceConfigurationError("max_query_chars must be >= 1")
        if float(self.request_timeout) <= 0:
            raise ServiceConfigurationError("request_timeout must be > 0")
        if int(self.rate_limit_per_minute) < 1:
            raise ServiceConfigurationError(
                "rate_limit_per_minute must be >= 1")
        if int(self.rate_limit_burst) < 1:
            raise ServiceConfigurationError("rate_limit_burst must be >= 1")
        if int(self.engine_capacity) < 1:
            raise ServiceConfigurationError("engine_capacity must be >= 1")
        if int(self.engine_dim) < 1:
            raise ServiceConfigurationError("engine_dim must be >= 1")
        if str(self.credential_policy).strip().lower() not in CREDENTIAL_POLICIES:
            raise ServiceConfigurationError(
                f"credential_policy must be one of {CREDENTIAL_POLICIES}")
        object.__setattr__(self, "credential_policy",
                           str(self.credential_policy).strip().lower())

        if self.production and self.allow_unauthenticated_dev:
            raise ServiceConfigurationError(
                "allow_unauthenticated_dev is forbidden in production")
        if not self.api_keys and not self.allow_unauthenticated_dev:
            raise ServiceConfigurationError(
                "no API keys are configured; provision MINEMOS_API_KEYS_FILE "
                "or MINEMOS_API_KEYS (or explicitly enable "
                "MINEMOS_DEV_INSECURE_NO_AUTH for local development)")

        if self.production:
            for spec in self.api_keys:
                if len(spec.key) < 32:
                    raise ServiceConfigurationError(
                        f"api key {spec.key_id!r} is shorter than 32 "
                        "characters; production keys must be high entropy")
            if not Path(self.data_dir).is_absolute():
                raise ServiceConfigurationError(
                    "production data_dir must be an absolute path")
            if self.binding_non_loopback and not self.tls_enabled:
                if not self.allow_insecure_http:
                    raise ServiceConfigurationError(
                        "refusing to bind a non-loopback host without TLS; "
                        "set MINEMOS_TLS_CERT_FILE and MINEMOS_TLS_KEY_FILE "
                        "or explicitly set MINEMOS_ALLOW_INSECURE_HTTP=true "
                        "for a trusted private network")
        if self.tls_enabled:
            for label, path in (("tls_cert_file", self.tls_cert_file),
                                ("tls_key_file", self.tls_key_file)):
                if path is None or not Path(path).expanduser().is_file():
                    raise ServiceConfigurationError(
                        f"{label} does not exist: {path}")

        if self.qdrant_path and self.production:
            raise ServiceConfigurationError(
                "embedded Qdrant (MINEMOS_QDRANT_PATH) is not permitted in "
                "production; use a server-mode QDRANT_URL")
        if self.qdrant_url and self.qdrant_path:
            raise ServiceConfigurationError(
                "configure either MINEMOS_QDRANT_URL or "
                "MINEMOS_QDRANT_PATH, not both")
        if self.retrieval_mode is not None:
            mode = str(self.retrieval_mode).strip().lower()
            if mode not in ("legacy", "production", "canonical"):
                raise ServiceConfigurationError(
                    "retrieval_mode must be legacy, production or canonical")
            object.__setattr__(self, "retrieval_mode", mode)
        if (self.retrieval_mode == "canonical"
                and (self.qdrant_url or self.qdrant_path)):
            raise ServiceConfigurationError(
                "retrieval_mode=canonical cannot be combined with a Qdrant "
                "configuration")
        if self.retrieval_mode == "production" and not self.qdrant_url:
            raise ServiceConfigurationError(
                "retrieval_mode=production requires MINEMOS_QDRANT_URL")
        if self.production and self.retrieval_mode != "production":
            raise ServiceConfigurationError(
                "production environment requires "
                "MINEMOS_RETRIEVAL_MODE=production")
        if self.qdrant_url:
            parsed = urlsplit(self.qdrant_url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ServiceConfigurationError(
                    "qdrant_url must be an http(s) URL with a host")
            if (not is_loopback_host(parsed.hostname)
                    and parsed.scheme != "https"
                    and not self.allow_insecure_http):
                raise ServiceConfigurationError(
                    "non-loopback Qdrant URLs must use https unless "
                    "MINEMOS_ALLOW_INSECURE_HTTP=true")
            if (not is_loopback_host(parsed.hostname)
                    and not self.qdrant_api_key):
                raise ServiceConfigurationError(
                    "non-loopback Qdrant requires an API key")
        collection = str(self.qdrant_collection_prefix or "").strip()
        if not collection or not re.match(r"^[A-Za-z0-9_.-]{1,120}$", collection):
            raise ServiceConfigurationError(
                "qdrant_collection_prefix must match ^[A-Za-z0-9_.-]{1,120}$")
        return self

    # -- reporting -------------------------------------------------------- #
    def describe(self) -> Dict[str, Any]:
        """A secret-free description suitable for logs and diagnostics."""
        return {
            "host": self.host,
            "port": int(self.port),
            "environment": self.environment,
            "data_dir": str(self.resolved_data_dir),
            "checkpoint_dir": str(self.resolved_checkpoint_dir),
            "provenance_dir": str(self.resolved_provenance_dir),
            "tls_enabled": self.tls_enabled,
            "allow_insecure_http": bool(self.allow_insecure_http),
            "max_body_bytes": int(self.max_body_bytes),
            "rate_limit_per_minute": int(self.rate_limit_per_minute),
            "rate_limit_burst": int(self.rate_limit_burst),
            "engine_capacity": int(self.engine_capacity),
            "engine_dim": int(self.engine_dim),
            "retrieval_mode": self.retrieval_mode,
            "qdrant_url": _safe_url(self.qdrant_url),
            "qdrant_collection_prefix": self.qdrant_collection_prefix,
            "persist_after_mutation": bool(self.persist_after_mutation),
            "credential_policy": self.credential_policy,
            "api_keys": [spec.as_dict() for spec in self.api_keys],
        }

    def tenant_configs(self) -> Tuple[str, ...]:
        return tuple(sorted({spec.tenant for spec in self.api_keys}))

    # -- loading ---------------------------------------------------------- #
    @classmethod
    def load(
        cls,
        env: Optional[Mapping[str, str]] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> "ServiceConfig":
        raw = os.environ if env is None else env
        api_keys: Tuple[ApiKeySpec, ...] = ()
        key_file = raw.get("MINEMOS_API_KEYS_FILE")
        if key_file:
            api_keys = _load_key_file(Path(key_file).expanduser())
        else:
            inline = raw.get("MINEMOS_API_KEYS")
            if inline:
                try:
                    api_keys = parse_api_keys(json.loads(inline))
                except json.JSONDecodeError as exc:
                    raise ServiceConfigurationError(
                        f"MINEMOS_API_KEYS is not valid JSON: {exc}") from exc

        config = cls(
            host=raw.get("MINEMOS_HOST", "127.0.0.1"),
            port=_env_int(raw, "MINEMOS_PORT", 8080),
            environment=raw.get("MINEMOS_ENVIRONMENT", "development"),
            data_dir=Path(raw.get("MINEMOS_DATA_DIR", "var/minemos")),
            checkpoint_dir=(
                Path(raw["MINEMOS_CHECKPOINT_DIR"])
                if raw.get("MINEMOS_CHECKPOINT_DIR") else None),
            provenance_dir=(
                Path(raw["MINEMOS_PROVENANCE_DIR"])
                if raw.get("MINEMOS_PROVENANCE_DIR") else None),
            tls_cert_file=(
                Path(raw["MINEMOS_TLS_CERT_FILE"])
                if raw.get("MINEMOS_TLS_CERT_FILE") else None),
            tls_key_file=(
                Path(raw["MINEMOS_TLS_KEY_FILE"])
                if raw.get("MINEMOS_TLS_KEY_FILE") else None),
            allow_insecure_http=_env_bool(
                raw, "MINEMOS_ALLOW_INSECURE_HTTP", False),
            allow_unauthenticated_dev=_env_bool(
                raw, "MINEMOS_DEV_INSECURE_NO_AUTH", False),
            max_body_bytes=_env_int(
                raw, "MINEMOS_MAX_BODY_BYTES", 1_048_576),
            max_observations_per_request=_env_int(
                raw, "MINEMOS_MAX_OBSERVATIONS_PER_REQUEST", 256),
            max_object_chars=_env_int(raw, "MINEMOS_MAX_OBJECT_CHARS", 8192),
            max_query_chars=_env_int(raw, "MINEMOS_MAX_QUERY_CHARS", 4096),
            request_timeout=_env_float(raw, "MINEMOS_REQUEST_TIMEOUT", 30.0),
            rate_limit_per_minute=_env_int(
                raw, "MINEMOS_RATE_LIMIT_PER_MINUTE", 600),
            rate_limit_burst=_env_int(
                raw, "MINEMOS_RATE_LIMIT_BURST", 100),
            engine_capacity=_env_int(raw, "MINEMOS_ENGINE_CAPACITY", 10_000),
            engine_dim=_env_int(raw, "MINEMOS_ENGINE_DIM", 4096),
            retrieval_mode=raw.get("MINEMOS_RETRIEVAL_MODE") or None,
            qdrant_url=raw.get("MINEMOS_QDRANT_URL") or None,
            qdrant_path=raw.get("MINEMOS_QDRANT_PATH") or None,
            qdrant_api_key=raw.get("MINEMOS_QDRANT_API_KEY") or None,
            qdrant_collection_prefix=raw.get(
                "MINEMOS_QDRANT_COLLECTION_PREFIX", "minemos_memories"),
            qdrant_timeout=_env_float(raw, "MINEMOS_QDRANT_TIMEOUT", 5.0),
            persist_after_mutation=_env_bool(
                raw, "MINEMOS_PERSIST_AFTER_MUTATION", True),
            credential_policy=raw.get("MINEMOS_CREDENTIAL_POLICY", "reject"),
            log_level=raw.get("MINEMOS_LOG_LEVEL", "INFO"),
            api_keys=api_keys,
        )
        if overrides:
            unknown = set(overrides) - set(cls.__dataclass_fields__)
            if unknown:
                raise ServiceConfigurationError(
                    f"unknown config overrides: {sorted(unknown)}")
            config = replace(config, **overrides)
        return config.validate()
