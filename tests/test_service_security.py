"""Security-boundary tests for the MINEMOS service.

These tests cover configuration safety, authentication, authorization,
tenant-derived isolation, request validation, rate limiting, body limits,
safe error responses and secret-free logging.  They are verified locally
against the stdlib HTTP server with deterministic stub encoders.
"""

from __future__ import annotations

import json
import logging
import ssl
import subprocess
from http.client import HTTPConnection, HTTPSConnection
from io import StringIO

import pytest

from minemos_service.config import (
    ApiKeySpec,
    ServiceConfigurationError,
    ServiceConfig,
)
from minemos_service.security import (
    ApiKeyRegistry,
    JsonLogFormatter,
    RateLimiter,
    RequestError,
    find_sensitive,
    validate_ingest_payload,
    validate_query_payload,
    redact_sensitive,
)
from tests._service_helpers import (
    ADMIN_KEY,
    OTHER_KEY,
    READER_KEY,
    WRITER_KEY,
    default_keys,
    make_config,
    make_pool,
    running_server,
)


# ── helpers ────────────────────────────────────────────────────────────────


def request(base: str, method: str, path: str, body=None,
            headers=None, raw: bytes | None = None):
    host, port = base.split("//", 1)[1].rsplit(":", 1)
    connection = HTTPConnection(host.strip("[]"), int(port), timeout=10)
    payload = raw
    if payload is None and body is not None:
        payload = json.dumps(body).encode("utf-8")
    merged = {"Content-Type": "application/json"}
    for name, value in (headers or {}).items():
        if value is None:
            merged.pop(name, None)
        else:
            merged[name] = value
    if payload is not None and "Content-Length" not in merged:
        merged["Content-Length"] = str(len(payload))
    try:
        connection.request(method, path, body=payload, headers=merged)
        response = connection.getresponse()
        data = response.read()
        parsed = json.loads(data) if data else None
        return response.status, dict(response.getheaders()), parsed
    finally:
        connection.close()


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── configuration ──────────────────────────────────────────────────────────


def test_config_requires_credentials(tmp_path):
    with pytest.raises(ServiceConfigurationError):
        ServiceConfig.load(env={}, overrides={
            "data_dir": str(tmp_path / "data"),
        })


def test_dev_no_auth_must_be_explicit(tmp_path):
    config = make_config(tmp_path, keys=(), allow_unauthenticated_dev=True)
    assert config.api_keys == ()
    assert config.allow_unauthenticated_dev is True


def test_production_rejects_unauthenticated_dev(tmp_path):
    with pytest.raises(ServiceConfigurationError):
        make_config(tmp_path, keys=(), allow_unauthenticated_dev=True,
                    environment="production")


def test_production_rejects_short_keys_relative_paths_and_legacy_mode(
        tmp_path):
    short = (ApiKeySpec("short", "too-short", "tenant-a", ("admin",)),)
    with pytest.raises(ServiceConfigurationError, match="shorter than 32"):
        make_config(tmp_path, keys=short, environment="production")
    with pytest.raises(ServiceConfigurationError, match="absolute"):
        make_config(tmp_path, environment="production",
                    data_dir="relative/path",
                    retrieval_mode="production",
                    qdrant_url="http://127.0.0.1:6333")
    with pytest.raises(ServiceConfigurationError, match="requires"):
        make_config(tmp_path, environment="production",
                    retrieval_mode="legacy")


def test_production_rejects_embedded_qdrant(tmp_path):
    with pytest.raises(ServiceConfigurationError, match="embedded Qdrant"):
        make_config(tmp_path, environment="production",
                    retrieval_mode="production",
                    qdrant_path=str(tmp_path / "qdrant"))


def test_non_loopback_requires_tls_and_key(tmp_path):
    with pytest.raises(ServiceConfigurationError, match="without TLS"):
        make_config(tmp_path, environment="production", host="0.0.0.0",
                    retrieval_mode="production",
                    qdrant_url="https://qdrant.example:6333")
    with pytest.raises(ServiceConfigurationError, match="API key"):
        make_config(tmp_path, environment="production", host="0.0.0.0",
                    allow_insecure_http=True, retrieval_mode="production",
                    qdrant_url="http://qdrant.example:6333")


def test_canonical_mode_cannot_attach_qdrant(tmp_path):
    with pytest.raises(ServiceConfigurationError, match="canonical"):
        make_config(tmp_path, retrieval_mode="canonical",
                    qdrant_url="http://127.0.0.1:6333")


def test_describe_and_repr_never_leak_secrets(tmp_path):
    secret = "sk-live-" + "z" * 40
    keys = (ApiKeySpec("k", secret, "tenant-a", ("admin",)),)
    config = make_config(
        tmp_path, keys=keys, qdrant_api_key="qdrant-secret-value")
    described = json.dumps(config.describe())
    combined = described + repr(config)
    assert secret not in combined
    assert "qdrant-secret-value" not in combined
    assert "k" in described
    assert config.qdrant_api_key == "qdrant-secret-value"


def test_safe_url_strips_userinfo(tmp_path):
    config = make_config(
        tmp_path, qdrant_url="https://user:password@127.0.0.1:6333")
    assert "user" not in config.describe()["qdrant_url"]
    assert "password" not in config.describe()["qdrant_url"]


# ── authentication and rate limiting ───────────────────────────────────────


def test_registry_authenticates_all_configured_keys():
    registry = ApiKeyRegistry(default_keys())
    principal = registry.authenticate(f"Bearer {WRITER_KEY}")
    assert principal is not None
    assert principal.tenant == "tenant-a"
    assert principal.roles == ("reader", "writer")
    assert principal.can("writer") and principal.can("reader")
    assert not principal.can("admin")


def test_registry_rejects_malformed_and_unknown_credentials():
    registry = ApiKeyRegistry(default_keys())
    for value in (None, "", "Basic abc", "Bearer", "Bearer wrong",
                  "Bearer " + "x" * 5000):
        assert registry.authenticate(value) is None


def test_registry_does_not_short_circuit(monkeypatch):
    import hmac as hmac_module
    calls = {"count": 0}
    real = hmac_module.compare_digest

    def counting(left, right):
        calls["count"] += 1
        return real(left, right)

    monkeypatch.setattr(hmac_module, "compare_digest", counting)
    keys = (
        ApiKeySpec("first", "a" * 40, "tenant-a", ("reader",)),
        ApiKeySpec("second", "b" * 40, "tenant-a", ("reader",)),
    )
    registry = ApiKeyRegistry(keys)
    assert registry.authenticate(f"Bearer {'a' * 40}") is not None
    assert calls["count"] == 2


def test_admin_role_authorizes_everything():
    from minemos_service.security import Principal
    principal = Principal("tenant-a", "admin", ("admin",))
    assert all(principal.can(role) for role in ("reader", "writer", "admin"))


def test_rate_limiter_refills_and_reports_wait():
    now = {"value": 0.0}
    limiter = RateLimiter(60, 2, clock=lambda: now["value"])
    assert limiter.acquire("k") == 0.0
    assert limiter.acquire("k") == 0.0
    wait = limiter.acquire("k")
    assert wait >= 1
    now["value"] = 1.0
    assert limiter.acquire("k") == 0.0
    assert limiter.snapshot()["bucket_count"] == 1


# ── request validation ─────────────────────────────────────────────────────


def _ingest_body(**observation) -> dict:
    source = observation.pop("source", {
        "source_id": "s1", "source_type": "conversation"})
    return {"source": source,
            "observations": [{"subject": "alice", "predicate": "city",
                              "object": "Berlin", **observation}]}


def test_validation_rejects_unknown_fields(tmp_path):
    config = make_config(tmp_path)
    body = _ingest_body()
    body["unexpected"] = True
    with pytest.raises(RequestError) as error:
        validate_ingest_payload(body, config)
    assert error.value.status == 400


def test_validation_rejects_control_characters_and_bounds(tmp_path):
    config = make_config(tmp_path, max_object_chars=10)
    with pytest.raises(RequestError):
        validate_ingest_payload(
            _ingest_body(object="bad\x00value"), config)
    with pytest.raises(RequestError):
        validate_ingest_payload(
            _ingest_body(object="x" * 11), config)
    with pytest.raises(RequestError):
        validate_ingest_payload(
            _ingest_body(timestamp=float("nan")), config)
    with pytest.raises(RequestError):
        validate_ingest_payload(_ingest_body(confidence=True), config)
    with pytest.raises(RequestError):
        validate_ingest_payload(_ingest_body(confidence=1.5), config)


def test_validation_rejects_too_many_observations(tmp_path):
    config = make_config(tmp_path, max_observations_per_request=2)
    body = {"source": {"source_id": "s1"},
            "observations": [
                {"subject": "a", "predicate": "p", "object": str(i)}
                for i in range(3)]}
    with pytest.raises(RequestError) as error:
        validate_ingest_payload(body, config)
    assert error.value.status == 413


def test_credentials_are_rejected_or_redacted(tmp_path):
    secret = "AKIA" + "A" * 16
    reject = make_config(tmp_path)
    with pytest.raises(RequestError) as error:
        validate_ingest_payload(_ingest_body(object=secret), reject)
    assert error.value.status == 422

    redact = make_config(tmp_path, credential_policy="redact")
    validated = validate_ingest_payload(
        _ingest_body(object=f"key {secret}"), redact)
    observation = validated["observations"][0]
    assert secret not in observation["object"]
    assert "REDACTED" in observation["object"]
    assert observation["redacted_kinds"]


def test_credentials_in_source_identifiers_are_always_rejected(tmp_path):
    config = make_config(tmp_path, credential_policy="redact")
    body = _ingest_body(source={
        "source_id": "AKIA" + "A" * 16, "source_type": "document"})
    with pytest.raises(RequestError) as error:
        validate_ingest_payload(body, config)
    assert error.value.status == 422


def test_query_validation_bounds(tmp_path):
    config = make_config(tmp_path)
    with pytest.raises(RequestError):
        validate_query_payload({"query": "x", "top_k": True}, config)
    with pytest.raises(RequestError):
        validate_query_payload({"query": "x", "top_k": 101}, config)
    with pytest.raises(RequestError):
        validate_query_payload({"query": "x", "as_of": float("inf")}, config)
    with pytest.raises(RequestError):
        validate_query_payload({"query": "x", "extra": 1}, config)


def test_sensitive_detection_reports_kinds_not_values():
    secret = "ghp_" + "a" * 30
    kinds = find_sensitive(f"token {secret}")
    assert "github_token" in kinds
    assert secret not in json.dumps(kinds)
    assert secret not in redact_sensitive(secret)


def test_json_log_formatter_scrubs_secrets():
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonLogFormatter())
    logger = logging.getLogger("minemos_service_test_logs")
    logger.handlers[:] = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    secret = "AKIA" + "B" * 16
    logger.info("failed with %s", secret,
                extra={"request_id": "r-1", "note": f"token={secret}"})
    line = stream.getvalue().strip()
    record = json.loads(line)
    assert record["request_id"] == "r-1"
    assert secret not in line
    assert secret not in record["note"]


# ── HTTP boundary ──────────────────────────────────────────────────────────


def test_health_and_ready_are_public(tmp_path):
    config, pool = make_pool(tmp_path)
    with running_server(config, pool) as (base, _):
        status, headers, body = request(base, "GET", "/healthz")
        assert status == 200 and body["status"] == "ok"
        status, _, body = request(base, "GET", "/readyz")
        assert status == 200 and body["ready"] is True
        assert headers["Cache-Control"] == "no-store"
        assert headers["X-Content-Type-Options"] == "nosniff"


def test_protected_routes_require_bearer_token(tmp_path):
    config, pool = make_pool(tmp_path)
    with running_server(config, pool) as (base, _):
        status, headers, body = request(base, "GET", "/v1/status")
        assert status == 401
        assert headers["WWW-Authenticate"] == "Bearer"
        assert body["error"]["code"] == "unauthorized"
        assert "request_id" in body


def test_roles_are_enforced(tmp_path):
    config, pool = make_pool(tmp_path)
    with running_server(config, pool) as (base, _):
        status, _, _ = request(base, "POST", "/v1/ingest",
                               _ingest_body(), bearer(READER_KEY))
        assert status == 403
        status, _, _ = request(base, "POST", "/v1/admin/checkpoint", {},
                               bearer(WRITER_KEY))
        assert status == 403
        status, _, body = request(base, "POST", "/v1/admin/checkpoint", {},
                                  bearer(ADMIN_KEY))
        assert status == 200 and body["bytes"] > 0


def test_ingest_response_hides_process_local_slot_id(tmp_path):
    config, pool = make_pool(tmp_path)
    with running_server(config, pool) as (base, _):
        status, _, body = request(base, "POST", "/v1/ingest",
                                  _ingest_body(), bearer(WRITER_KEY))
        assert status == 200
        assert body["ingested"] == 1 and body["failed"] == 0
        result = body["results"][0]
        assert "memory_id" in result and "slot_id" not in result
        status, _, body = request(base, "POST", "/v1/query",
                                  {"query": "Berlin"},
                                  bearer(READER_KEY))
        assert status == 200 and body["count"] >= 1
        assert all("slot_id" not in row for row in body["rows"])


def test_tenant_isolation_is_derived_from_credential(tmp_path):
    config, pool = make_pool(tmp_path)
    with running_server(config, pool) as (base, _):
        request(base, "POST", "/v1/ingest", _ingest_body(),
                bearer(WRITER_KEY))
        status, _, body = request(base, "POST", "/v1/query",
                                  {"query": "Berlin"}, bearer(READER_KEY))
        assert status == 200 and body["count"] >= 1
        status, _, body = request(base, "POST", "/v1/query",
                                  {"query": "Berlin"}, bearer(OTHER_KEY))
        assert status == 200 and body["count"] == 0
        status, _, body = request(base, "GET", "/v1/status",
                                  headers=bearer(OTHER_KEY))
        assert body["tenant"] == "tenant-b"


def test_tenant_header_cannot_override_authenticated_tenant(tmp_path):
    config, pool = make_pool(tmp_path)
    with running_server(config, pool) as (base, _):
        headers = bearer(WRITER_KEY)
        headers["X-MINEMOS-TENANT"] = "tenant-b"
        status, _, body = request(base, "POST", "/v1/ingest", _ingest_body(),
                                  headers)
        assert status == 200 and body["tenant"] == "tenant-a"
        status, _, body = request(base, "POST", "/v1/query",
                                  {"query": "Berlin"}, headers=bearer(READER_KEY))
        assert body["count"] >= 1
        status, _, body = request(base, "POST", "/v1/query",
                                  {"query": "Berlin"}, bearer(OTHER_KEY))
        assert body["count"] == 0


def test_redact_and_delete_by_stable_memory_id(tmp_path):
    config, pool = make_pool(tmp_path)
    with running_server(config, pool) as (base, _):
        _, _, body = request(base, "POST", "/v1/ingest", _ingest_body(),
                             bearer(WRITER_KEY))
        memory_id = body["results"][0]["memory_id"]
        status, _, body = request(base, "POST", "/v1/query",
                                  {"query": "Berlin"}, bearer(READER_KEY))
        assert body["count"] == 1
        status, _, body = request(base, "POST", "/v1/redact",
                                  {"memory_id": memory_id},
                                  bearer(WRITER_KEY))
        assert status == 200 and body["changed"] is True
        status, _, body = request(base, "POST", "/v1/query",
                                  {"query": "Berlin"}, bearer(READER_KEY))
        assert body["count"] == 0
        # A second redact is a no-op; deleting the tombstone is a real change.
        status, _, body = request(base, "POST", "/v1/redact",
                                  {"memory_id": memory_id},
                                  bearer(WRITER_KEY))
        assert status == 200 and body["changed"] is False
        status, _, body = request(base, "POST", "/v1/delete",
                                  {"memory_id": memory_id},
                                  bearer(WRITER_KEY))
        assert status == 200 and body["changed"] is True
        status, _, body = request(base, "POST", "/v1/delete",
                                  {"memory_id": memory_id},
                                  bearer(WRITER_KEY))
        assert status == 200 and body["changed"] is False


def test_malformed_requests_get_safe_errors(tmp_path):
    config, pool = make_pool(tmp_path)
    with running_server(config, pool) as (base, _):
        host, port = base.split("//", 1)[1].rsplit(":", 1)
        connection = HTTPConnection(host, int(port), timeout=10)
        try:
            connection.request("POST", "/v1/ingest", body=b"{not-json",
                               headers={"Authorization": f"Bearer {WRITER_KEY}",
                                        "Content-Type": "application/json",
                                        "Content-Length": "9"})
            response = connection.getresponse()
            body = json.loads(response.read())
            assert response.status == 400
            assert body["error"]["code"] == "invalid_json"
        finally:
            connection.close()

        status, _, body = request(
            base, "POST", "/v1/ingest", _ingest_body(),
            {"Authorization": f"Bearer {WRITER_KEY}",
             "Content-Type": "text/plain"})
        assert status == 415
        status, _, body = request(base, "POST", "/v1/ingest", None,
                                  bearer(WRITER_KEY))
        assert status == 400
        assert body["error"]["code"] == "invalid_request"
        status, _, body = request(base, "PUT", "/v1/ingest", {})
        assert status == 405
        assert body["error"]["code"] == "method_not_allowed"
        status, _, body = request(base, "GET", "/nope")
        assert status == 404
        status, _, body = request(base, "POST", "/v1/query",
                                  {"query": "x", "top_k": 101},
                                  bearer(READER_KEY))
        assert status == 400
        assert "internal" not in json.dumps(body).lower()


def test_body_limit_is_enforced(tmp_path):
    config, pool = make_pool(tmp_path, max_body_bytes=1024)
    with running_server(config, pool) as (base, _):
        status, _, body = request(
            base, "POST", "/v1/ingest",
            {"padding": "x" * 2000},
            bearer(WRITER_KEY))
        assert status == 413
        assert body["error"]["code"] == "payload_too_large"


def test_rate_limit_returns_retry_after(tmp_path):
    config, pool = make_pool(tmp_path)
    limiter = RateLimiter(1, 1, clock=lambda: 0.0)
    with running_server(config, pool, limiter=limiter) as (base, _):
        status, _, _ = request(base, "GET", "/v1/status",
                               headers=bearer(READER_KEY))
        assert status == 200
        status, headers, body = request(base, "GET", "/v1/status",
                                        headers=bearer(READER_KEY))
        assert status == 429
        assert headers["Retry-After"] == "60"
        assert body["error"]["code"] == "rate_limited"


def test_dev_no_auth_requires_tenant_header(tmp_path):
    config, pool = make_pool(
        tmp_path, keys=(), allow_unauthenticated_dev=True)
    pool.initialize(["tenant-a"])
    with running_server(config, pool) as (base, _):
        status, _, _ = request(base, "GET", "/v1/status")
        assert status == 401
        status, _, body = request(
            base, "GET", "/v1/status",
            headers={"X-MINEMOS-TENANT": "tenant-a"})
        assert status == 200 and body["tenant"] == "tenant-a"
        status, _, _ = request(
            base, "GET", "/v1/status",
            headers={"X-MINEMOS-TENANT": "../bad"})
        assert status == 401


def test_tls_listener_serves_health(tmp_path):
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    try:
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
             "-keyout", str(key), "-out", str(cert), "-days", "1",
             "-subj", "/CN=127.0.0.1"],
            check=True, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"openssl unavailable: {exc}")
    config, pool = make_pool(
        tmp_path, tls_cert_file=str(cert), tls_key_file=str(key))
    with running_server(config, pool) as (base, _):
        host, port = base.split("//", 1)[1].rsplit(":", 1)
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        connection = HTTPSConnection(host, int(port), timeout=10,
                                     context=context)
        try:
            connection.request("GET", "/healthz")
            response = connection.getresponse()
            body = json.loads(response.read())
            assert response.status == 200 and body["status"] == "ok"
        finally:
            connection.close()
