"""Shared deterministic fixtures for the service-layer tests."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from minemos_service.config import ApiKeySpec, ServiceConfig
from minemos_service.engine_pool import EnginePool, ModelSet
from minemos_service.security import ApiKeyRegistry, RateLimiter
from minemos_service.server import ServiceApp, create_server
from tests._retrieval_helpers import FixedEncoder, RuleVerifier, make_memory

READER_KEY = "reader-" + "r" * 40
WRITER_KEY = "writer-" + "w" * 40
ADMIN_KEY = "admin-" + "a" * 40
OTHER_KEY = "other-" + "o" * 40


def make_models() -> ModelSet:
    return ModelSet(dense_encoder=FixedEncoder(), nli_gate=None,
                    lex_encoder=None)


def make_factory():
    def factory(config, tenant_id, models):
        return make_memory(
            capacity=config.engine_capacity,
            verifier=RuleVerifier(),
            encoder=models.dense_encoder,
        )
    return factory


def default_keys() -> Tuple[ApiKeySpec, ...]:
    return (
        ApiKeySpec("reader", READER_KEY, "tenant-a", ("reader",)),
        ApiKeySpec("writer", WRITER_KEY, "tenant-a", ("reader", "writer")),
        ApiKeySpec("admin", ADMIN_KEY, "tenant-a", ("admin",)),
        ApiKeySpec("other", OTHER_KEY, "tenant-b", ("admin",)),
    )


def make_config(tmp_path: Path, keys: Optional[Tuple[ApiKeySpec, ...]] = None,
                **overrides: Any) -> ServiceConfig:
    values: Dict[str, Any] = {
        "data_dir": str(Path(tmp_path) / "data"),
        "environment": "development",
        "api_keys": default_keys() if keys is None else keys,
        "engine_capacity": 64,
        "engine_dim": 4096,
        "persist_after_mutation": True,
        "rate_limit_per_minute": 60_000,
        "rate_limit_burst": 10_000,
        "log_level": "CRITICAL",
    }
    values.update(overrides)
    return ServiceConfig.load(env={}, overrides=values)


def make_pool(tmp_path: Path, keys: Optional[Tuple[ApiKeySpec, ...]] = None,
              **overrides: Any) -> Tuple[ServiceConfig, EnginePool]:
    config = make_config(tmp_path, keys=keys, **overrides)
    pool = EnginePool(config, engine_factory=make_factory(),
                      models=make_models())
    pool.initialize([spec.tenant for spec in config.api_keys])
    return config, pool


@contextmanager
def running_server(config: ServiceConfig, pool: EnginePool,
                   limiter: Optional[RateLimiter] = None
                   ) -> Iterator[Tuple[str, ServiceApp]]:
    app = ServiceApp(
        config, ApiKeyRegistry.from_config(config), pool, limiter=limiter)
    server = create_server(config, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    if ":" in str(host):
        host = f"[{host}]"
    base = f"http://{host}:{port}"
    try:
        yield base, app
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        pool.shutdown()
