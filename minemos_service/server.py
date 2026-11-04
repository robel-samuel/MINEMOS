"""HTTP boundary for the MINEMOS service.

The server is intentionally small: stdlib ``ThreadingHTTPServer``, strict JSON
validation, bearer authentication, per-credential rate limiting, tenant-scoped
routing and JSON logging without request bodies or secrets.  It is not a
general-purpose application framework and it does not expose the engine
directly.

Deployment rule: do not bind a non-loopback address without TLS and without
provisioned API keys.  ``ServiceConfig.validate`` enforces that rule in the
production profile and refuses to start otherwise.
"""

from __future__ import annotations

import json
import logging
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from .config import ServiceConfig, is_loopback_host, validate_tenant_id
from .engine_pool import (
    EngineInitializationError,
    EnginePool,
    EnginePoolError,
    PersistenceError,
)
from .provenance import ProvenanceError
from .security import (
    ApiKeyRegistry,
    Principal,
    RateLimiter,
    RequestError,
    request_id_from,
    safe_message,
    validate_checkpoint_payload,
    validate_history_payload,
    validate_ingest_payload,
    validate_memory_payload,
    validate_query_payload,
    validate_source_payload,
)

LOGGER_NAME = "minemos_service"


@dataclass(frozen=True)
class Route:
    handler: Callable[[Optional[Principal], Any], Tuple[int, Dict[str, Any]]]
    role: Optional[str] = None
    public: bool = False


class ServiceApp:
    """Authentication, authorization and route handlers for one process."""

    def __init__(self, config: ServiceConfig, registry: ApiKeyRegistry,
                 pool: EnginePool, limiter: Optional[RateLimiter] = None,
                 logger: Optional[logging.Logger] = None):
        self.config = config
        self.registry = registry
        self.pool = pool
        self.limiter = limiter or RateLimiter(
            config.rate_limit_per_minute, config.rate_limit_burst)
        self.logger = logger or logging.getLogger(LOGGER_NAME)
        self.routes: Dict[Tuple[str, str], Route] = {
            ("GET", "/healthz"): Route(self._health, public=True),
            ("GET", "/readyz"): Route(self._ready, public=True),
            ("GET", "/v1/status"): Route(self._status, role="reader"),
            ("POST", "/v1/ingest"): Route(self._ingest, role="writer"),
            ("POST", "/v1/query"): Route(self._query, role="reader"),
            ("POST", "/v1/history"): Route(self._history, role="reader"),
            ("POST", "/v1/redact"): Route(self._redact, role="writer"),
            ("POST", "/v1/delete"): Route(self._delete, role="writer"),
            ("POST", "/v1/source/redact"): Route(
                self._source_redact, role="writer"),
            ("POST", "/v1/source/delete"): Route(
                self._source_delete, role="writer"),
            ("POST", "/v1/admin/checkpoint"): Route(
                self._checkpoint, role="admin"),
        }

    # -- request context -------------------------------------------------- #
    def authenticate(self, headers: Mapping[str, str]) -> Principal:
        principal = self.registry.authenticate(headers.get("Authorization"))
        if principal is not None:
            return principal
        if not self.registry.configured and self.config.allow_unauthenticated_dev:
            tenant = headers.get("X-MINEMOS-TENANT")
            if tenant:
                try:
                    tenant = validate_tenant_id(tenant)
                except Exception:
                    tenant = None
                if tenant:
                    return Principal(
                        tenant=tenant, key_id="dev-unauthenticated",
                        roles=("admin",))
        raise RequestError(
            401, "unauthorized", "a valid bearer token is required",
            {"WWW-Authenticate": "Bearer"})

    def enforce_rate(self, principal: Principal) -> None:
        retry_after = self.limiter.acquire(principal.key_id)
        if retry_after > 0:
            raise RequestError(
                429, "rate_limited", "request rate limit exceeded",
                {"Retry-After": str(int(retry_after))})

    def resolve(self, method: str, path: str) -> Route:
        route = self.routes.get((method, path))
        if route is None:
            known_path = any(p == path for _, p in self.routes)
            if known_path:
                raise RequestError(
                    405, "method_not_allowed", "method not allowed for route",
                    {"Allow": ", ".join(
                        m for (m, p) in self.routes if p == path)})
            raise RequestError(404, "not_found", "unknown route")
        return route

    # -- handlers --------------------------------------------------------- #
    def _health(self, principal: Optional[Principal],
                body: Any) -> Tuple[int, Dict[str, Any]]:
        health = self.pool.health()
        return 200, {"status": health["status"]}

    def _ready(self, principal: Optional[Principal],
               body: Any) -> Tuple[int, Dict[str, Any]]:
        readiness = self.pool.readiness()
        return 200, {
            "ready": readiness["ready"],
            "degraded": readiness["degraded"],
            "models_loaded": readiness["models_loaded"],
            "checkpoint_dir_writable": readiness["checkpoint_dir_writable"],
        }

    def _status(self, principal: Principal,
                body: Any) -> Tuple[int, Dict[str, Any]]:
        return 200, self.pool.status(principal.tenant)

    def _ingest(self, principal: Principal,
                body: Any) -> Tuple[int, Dict[str, Any]]:
        validated = validate_ingest_payload(body, self.config)
        results = []
        ingested = 0
        skipped = 0
        failed = 0
        persistence_failure = False
        processed = 0
        for observation in validated["observations"]:
            processed += 1
            try:
                result = self.pool.absorb(
                    principal.tenant,
                    subject=observation["subject"],
                    predicate=observation["predicate"],
                    object=observation["object"],
                    timestamp=observation["timestamp"],
                    confidence=observation["confidence"],
                    source=observation["source"],
                    ingestion_id=observation["ingestion_id"],
                )
            except PersistenceError as exc:
                failed += 1
                persistence_failure = True
                results.append({
                    "error": {
                        "code": "persistence_unavailable",
                        "message": safe_message(exc),
                    },
                })
                break
            except EngineInitializationError as exc:
                failed += 1
                persistence_failure = True
                results.append({
                    "error": {
                        "code": "engine_unavailable",
                        "message": safe_message(exc),
                    },
                })
                break
            except ProvenanceError:
                failed += 1
                persistence_failure = True
                results.append({
                    "error": {
                        "code": "provenance_unavailable",
                        "message": "ingestion could not be traced; "
                                   "the observation was not retained",
                    },
                })
                break
            except ValueError as exc:
                failed += 1
                results.append({
                    "error": {
                        "code": "invalid_observation",
                        "message": safe_message(exc),
                    },
                })
                continue
            result["content_redacted"] = bool(observation["redacted_kinds"])
            results.append(_public_record(result))
            if result.get("skipped"):
                skipped += 1
            else:
                ingested += 1
        unprocessed = len(validated["observations"]) - processed
        payload = {
            "tenant": principal.tenant,
            "results": results,
            "ingested": ingested,
            "skipped": skipped,
            "failed": failed,
            "unprocessed": unprocessed,
            "partial": bool(failed),
        }
        if persistence_failure and ingested == 0:
            return 503, payload
        if failed and ingested == 0:
            return 422, payload
        if failed:
            return 207, payload
        return 200, payload

    def _query(self, principal: Principal,
               body: Any) -> Tuple[int, Dict[str, Any]]:
        validated = validate_query_payload(body, self.config)
        rows = self.pool.query(
            principal.tenant, validated["query"], top_k=validated["top_k"],
            as_of=validated["as_of"])
        return 200, {"rows": [_public_record(r) for r in rows],
                     "count": len(rows)}

    def _history(self, principal: Principal,
                 body: Any) -> Tuple[int, Dict[str, Any]]:
        validated = validate_history_payload(body, self.config)
        rows = self.pool.history(
            principal.tenant, validated["subject"],
            predicate=validated["predicate"])
        return 200, {"rows": [_public_record(r) for r in rows],
                     "count": len(rows)}

    def _redact(self, principal: Principal,
                body: Any) -> Tuple[int, Dict[str, Any]]:
        validated = validate_memory_payload(body, self.config)
        changed = self.pool.redact_memory(
            principal.tenant, validated["memory_id"])
        return 200, {
            "memory_id": validated["memory_id"],
            "changed": bool(changed),
        }

    def _delete(self, principal: Principal,
                body: Any) -> Tuple[int, Dict[str, Any]]:
        validated = validate_memory_payload(body, self.config)
        changed = self.pool.delete_memory(
            principal.tenant, validated["memory_id"])
        return 200, {
            "memory_id": validated["memory_id"],
            "changed": bool(changed),
        }

    def _source_redact(self, principal: Principal,
                       body: Any) -> Tuple[int, Dict[str, Any]]:
        validated = validate_source_payload(body, self.config)
        return 200, self.pool.redact_source(
            principal.tenant, validated["source_id"])

    def _source_delete(self, principal: Principal,
                       body: Any) -> Tuple[int, Dict[str, Any]]:
        validated = validate_source_payload(body, self.config)
        return 200, self.pool.delete_source(
            principal.tenant, validated["source_id"])

    def _checkpoint(self, principal: Principal,
                    body: Any) -> Tuple[int, Dict[str, Any]]:
        validate_checkpoint_payload(body, self.config)
        return 200, self.pool.checkpoint(principal.tenant)


class MinemosHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(self, address, app: ServiceApp,
                 bind_and_activate: bool = True):
        self.app = app
        super().__init__(
            address, MinemosRequestHandler,
            bind_and_activate=bind_and_activate)


class MinemosRequestHandler(BaseHTTPRequestHandler):
    server_version = "MINEMOS"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    # -- plumbing --------------------------------------------------------- #
    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.server.app.config.request_timeout)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Structured request logging happens in _dispatch; suppress the
        # default access log, which writes unstructured text to stderr.
        return

    @property
    def app(self) -> ServiceApp:
        return self.server.app

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch("POST")

    def do_PUT(self) -> None:  # noqa: N802 - stdlib naming
        self._method_not_allowed()

    def do_PATCH(self) -> None:  # noqa: N802 - stdlib naming
        self._method_not_allowed()

    def do_DELETE(self) -> None:  # noqa: N802 - stdlib naming
        self._method_not_allowed()

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib naming
        self._method_not_allowed()

    def _method_not_allowed(self) -> None:
        request_id = request_id_from(self.headers.get("X-Request-ID"))
        self._send_error(405, "method_not_allowed",
                         "method not allowed", request_id,
                         {"Allow": "GET, POST"})

    # -- dispatch --------------------------------------------------------- #
    def _dispatch(self, method: str) -> None:
        request_id = request_id_from(self.headers.get("X-Request-ID"))
        path = urlsplit(self.path).path
        started = time.perf_counter()
        status = 500
        principal: Optional[Principal] = None
        try:
            route = self.app.resolve(method, path)
            if not route.public:
                principal = self.app.authenticate(self.headers)
                self.app.enforce_rate(principal)
                if route.role and not principal.can(route.role):
                    raise RequestError(
                        403, "forbidden",
                        f"this credential lacks the {route.role} role")
            body = (self._read_json_body(allow_empty=True)
                    if method == "POST" else None)
            status, payload = route.handler(principal, body)
            self._send_json(status, payload, request_id)
        except RequestError as exc:
            status = exc.status
            self._send_error(
                exc.status, exc.code, exc.message, request_id, exc.headers)
        except (BrokenPipeError, ConnectionResetError):
            status = 499
        except Exception:
            status = 500
            self.app.logger.exception(
                "unhandled request failure",
                extra={"request_id": request_id, "method": method, "path": path})
            self._send_error(500, "internal_error",
                             "internal server error", request_id)
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
            self.app.logger.info(
                "request",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status": status,
                    "duration_ms": duration_ms,
                    "tenant": principal.tenant if principal else None,
                    "key_id": principal.key_id if principal else None,
                })

    # -- body handling ---------------------------------------------------- #
    def _read_json_body(self, allow_empty: bool) -> Any:
        transfer_encoding = self.headers.get("Transfer-Encoding")
        if transfer_encoding:
            raise RequestError(
                400, "invalid_request",
                "chunked request bodies are not supported")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise RequestError(
                411, "length_required", "Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise RequestError(
                400, "invalid_request", "invalid Content-Length") from exc
        if length < 0:
            raise RequestError(
                400, "invalid_request", "invalid Content-Length")
        if length > self.app.config.max_body_bytes:
            self.close_connection = True
            raise RequestError(
                413, "payload_too_large",
                f"request body exceeds {self.app.config.max_body_bytes} bytes")
        if length == 0:
            if allow_empty:
                return None
            raise RequestError(
                400, "invalid_request", "request body is required")
        content_type = (self.headers.get("Content-Type") or "").split(";")[0]
        if content_type.strip().lower() != "application/json":
            raise RequestError(
                415, "unsupported_media_type",
                "Content-Type must be application/json")
        try:
            raw = self.rfile.read(length)
        except (socket.timeout, TimeoutError) as exc:
            raise RequestError(
                408, "request_timeout", "request body timed out") from exc
        if len(raw) != length:
            raise RequestError(
                400, "invalid_request", "incomplete request body")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RequestError(
                400, "invalid_json", "request body is not valid JSON") from exc

    # -- responses -------------------------------------------------------- #
    def _send_json(self, status: int, payload: Mapping[str, Any],
                   request_id: str,
                   extra_headers: Optional[Mapping[str, str]] = None) -> None:
        body = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"),
            default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Request-ID", request_id)
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_error(self, status: int, code: str, message: str,
                    request_id: str,
                    headers: Optional[Mapping[str, str]] = None) -> None:
        self._send_json(
            status,
            {
                "error": {
                    "code": code,
                    "message": safe_message(message),
                },
                "request_id": request_id,
            },
            request_id,
            headers,
        )


def _public_record(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Drops the process-local ``slot_id`` from API responses."""
    return {key: value for key, value in row.items() if key != "slot_id"}


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def create_server(config: ServiceConfig, app: ServiceApp,
                  bind_and_activate: bool = True) -> MinemosHTTPServer:
    """Creates the threaded HTTP server, wrapping the socket in TLS if set."""
    family = socket.AF_INET6 if ":" in config.host else socket.AF_INET
    server_cls = type(
        "MinemosHTTPServerV6" if family == socket.AF_INET6
        else "MinemosHTTPServerV4",
        (MinemosHTTPServer,),
        {"address_family": family},
    )
    server = server_cls(
        (config.host, config.port), app,
        bind_and_activate=bind_and_activate)
    if config.tls_enabled:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(
            str(config.tls_cert_file), str(config.tls_key_file))
        server.socket = context.wrap_socket(server.socket, server_side=True)
    return server
