"""``python -m minemos_service`` entry point.

Startup is deliberately fail-closed: an invalid production configuration, an
unreadable checkpoint, or an untraceable persistence path stops the process
instead of serving under an unsafe assumption.  Shutdown saves every tenant
checkpoint before closing engines.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from urllib.parse import urlsplit, urlunsplit

from .config import ServiceConfig, ServiceConfigurationError
from .engine_pool import (
    EngineInitializationError,
    EnginePool,
    EnginePoolError,
)
from .security import ApiKeyRegistry, configure_logging
from .server import ServiceApp, create_server


def _safe_url(url: str | None) -> str | None:
    if not url:
        return url
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query,
                       parsed.fragment))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="minemos_service",
        description="MINEMOS controlled-deployment HTTP service")
    parser.add_argument(
        "--check", action="store_true",
        help="validate configuration and initialize every configured tenant, "
             "then exit")
    parser.add_argument(
        "--check-config", action="store_true",
        help="validate configuration without loading models or checkpoints")
    parser.add_argument(
        "--print-config", action="store_true",
        help="print the secret-free effective configuration and exit")
    args = parser.parse_args(argv)

    try:
        config = ServiceConfig.load()
    except ServiceConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if args.print_config:
        import json
        described = config.describe()
        described["qdrant_url"] = _safe_url(described.get("qdrant_url"))
        print(json.dumps(described, indent=2, sort_keys=True))
        return 0
    if args.check_config:
        print("configuration OK")
        return 0

    logger = configure_logging(config.log_level)
    registry = ApiKeyRegistry.from_config(config)
    pool = EnginePool(config)
    try:
        initialized = pool.initialize(registry.tenants())
    except (EngineInitializationError, EnginePoolError) as exc:
        logger.error(
            "startup failed", extra={"error": type(exc).__name__})
        print(f"startup error: {exc}", file=sys.stderr)
        pool.shutdown()
        return 1
    if args.check:
        readiness = pool.readiness()
        print("initialization OK")
        print(f"tenants: {', '.join(initialized['tenants']) or '(none)'}")
        print(f"ready: {readiness['ready']} degraded: {readiness['degraded']}")
        pool.shutdown()
        return 0

    app = ServiceApp(config, registry, pool, logger=logger)
    try:
        server = create_server(config, app)
    except OSError as exc:
        logger.error("bind failed", extra={"error": type(exc).__name__})
        print(f"bind error: {exc}", file=sys.stderr)
        pool.shutdown()
        return 1

    stop = threading.Event()

    def _request_stop(signum, frame):  # noqa: ARG001 - signal signature
        if not stop.is_set():
            stop.set()
            logger.info("shutdown requested", extra={"signal": int(signum)})
            threading.Thread(target=server.shutdown, daemon=True).start()

    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, _request_stop)
            except (ValueError, OSError):
                pass

    logger.info(
        "service starting",
        extra={
            "host": config.host,
            "port": int(config.port),
            "environment": config.environment,
            "tls": config.tls_enabled,
            "retrieval_mode": config.retrieval_mode,
            "tenants": len(initialized["tenants"]),
            "data_dir": str(config.resolved_data_dir),
            "qdrant_url": _safe_url(config.qdrant_url),
        })
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        result = pool.shutdown()
        logger.info(
            "service stopped",
            extra={
                "checkpoints_saved": len(result.get("saved", [])),
                "shutdown_errors": len(result.get("errors", [])),
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
