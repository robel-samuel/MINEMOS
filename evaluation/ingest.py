"""Command-line ingestion and retention operations for the evaluation store.

Examples::

    python -m evaluation.ingest init --root evaluation-data
    python -m evaluation.ingest ingest --root evaluation-data \
        --source conversations.jsonl --source-type conversation \
        --mode verbatim --source-id conv-2026-01
    python -m evaluation.ingest delete-source --root evaluation-data \
        --source-id conv-2026-01

The CLI never resets state automatically and never accepts ground-truth data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .formats import (
    SourceFormatError,
    load_observation_mapping,
    observation_segments,
    parse_source_file,
)
from .store import (
    DEFAULT_TENANT,
    EvaluationEnvironment,
    EvaluationStore,
    EvaluationStoreError,
)


def _dump(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluation.ingest",
        description="MINEMOS real-world evaluation store")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_root(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", required=True,
                       help="dedicated evaluation directory")
        p.add_argument("--tenant", default=DEFAULT_TENANT)

    init = sub.add_parser("init", help="initialize an empty evaluation store")
    add_root(init)
    init.add_argument("--description", default="")

    ingest = sub.add_parser("ingest", help="ingest a real source file")
    add_root(ingest)
    ingest.add_argument("--source", required=True)
    ingest.add_argument("--source-type", required=True,
                        help="conversation, document, notes, journal, "
                             "technical_doc, research_doc, project_history, "
                             "observation or other")
    ingest.add_argument("--source-id", default=None)
    ingest.add_argument("--mode", choices=("verbatim", "structured"),
                        default="verbatim")
    ingest.add_argument(
        "--observations", default=None,
        help="authorized upstream extraction aligned by segment_key "
             "(required for structured mode on prose/conversation sources)")
    ingest.add_argument("--credential-policy", choices=("reject", "redact"),
                        default="reject")
    ingest.add_argument("--retrieval-mode",
                        choices=("canonical", "legacy", "production"),
                        default="canonical")
    ingest.add_argument("--qdrant-url", default=None)
    ingest.add_argument("--qdrant-path", default=None, help=argparse.SUPPRESS)
    ingest.add_argument("--qdrant-api-key", default=None,
                        help=argparse.SUPPRESS)
    ingest.add_argument("--dry-run", action="store_true")

    listing = sub.add_parser("list-sources")
    add_root(listing)

    delete = sub.add_parser("delete-source")
    add_root(delete)
    delete.add_argument("--source-id", required=True)
    delete.add_argument("--retrieval-mode", choices=("canonical", "legacy",
                                                     "production"),
                        default="canonical")

    redact = sub.add_parser("redact-source")
    add_root(redact)
    redact.add_argument("--source-id", required=True)
    redact.add_argument("--retrieval-mode", choices=("canonical", "legacy",
                                                     "production"),
                        default="canonical")

    for name in ("delete-memory", "redact-memory"):
        memory = sub.add_parser(name)
        add_root(memory)
        memory.add_argument("--memory-id", required=True)
        memory.add_argument("--retrieval-mode", choices=(
            "canonical", "legacy", "production"), default="canonical")

    verify = sub.add_parser("verify")
    add_root(verify)
    verify.add_argument("--retrieval-mode", choices=(
        "canonical", "legacy", "production"), default="canonical")

    status = sub.add_parser("status")
    add_root(status)
    status.add_argument("--retrieval-mode", choices=(
        "canonical", "legacy", "production"), default="canonical")
    return parser


def _open_environment(args) -> EvaluationEnvironment:
    return EvaluationEnvironment.open(
        args.root,
        tenant=args.tenant,
        retrieval_mode=args.retrieval_mode,
        qdrant_url=getattr(args, "qdrant_url", None),
        qdrant_path=getattr(args, "qdrant_path", None),
        qdrant_api_key=getattr(args, "qdrant_api_key", None),
    )


def _load_segments(args, store: Optional[EvaluationStore] = None):
    source_path = Path(args.source).expanduser()
    if store is None:
        store = EvaluationStore(args.root, tenant=args.tenant)
    segments = parse_source_file(
        source_path, args.source_type, source_id=args.source_id)
    if args.observations:
        if args.mode != "structured":
            raise EvaluationStoreError(
                "--observations requires --mode structured")
        mapping = load_observation_mapping(args.observations)
        segments = observation_segments(segments, mapping)
    if args.mode == "structured" and not any(
            segment.observations for segment in segments):
        raise EvaluationStoreError(
            "structured mode needs observations: pass --observations or use "
            "--source-type observation with subject/predicate/object lines")
    return store, segments


def _cmd_init(args) -> int:
    store = EvaluationStore.init(
        args.root, tenant=args.tenant, description=args.description)
    _dump({"root": str(store.root), "tenant": store.tenant,
           "format_version": store.manifest["format_version"]})
    return 0


def _cmd_ingest(args) -> int:
    store, segments = _load_segments(args)
    if not segments:
        raise EvaluationStoreError("source contains no ingestible segments")
    source_ids = {segment.source_id for segment in segments}
    if len(source_ids) != 1:
        raise EvaluationStoreError("a source file must contain one source_id")
    source_id = next(iter(source_ids))
    tombstone = store.source_tombstoned(source_id)
    if tombstone:
        raise EvaluationStoreError(
            f"source {source_id!r} was previously {tombstone}; use a new "
            "source_id instead of reviving deleted content")
    if args.dry_run:
        observations = 0
        for segment in segments:
            if args.mode == "verbatim":
                observations += 1
            else:
                observations += len(segment.observations)
        _dump({
            "dry_run": True,
            "source_id": source_id,
            "segments": len(segments),
            "observations": observations,
            "timestamped_segments": sum(
                1 for segment in segments
                if segment.original_timestamp is not None),
            "timestamp_inferred_segments": sum(
                1 for segment in segments
                if segment.original_timestamp is None),
        })
        return 0
    environment = _open_environment(args)
    try:
        result = environment.ingest_segments(
            segments,
            mode=args.mode,
            credential_policy=args.credential_policy,
        )
    finally:
        environment.close()
    result["tenant"] = args.tenant
    _dump(result)
    return 0


def _cmd_list_sources(args) -> int:
    store = EvaluationStore(args.root, tenant=args.tenant)
    _dump({"sources": store.sources()})
    return 0


def _cmd_delete_source(args) -> int:
    environment = _open_environment(args)
    try:
        result = environment.delete_source(args.source_id)
    finally:
        environment.close()
    _dump(result)
    return 0


def _cmd_redact_source(args) -> int:
    environment = _open_environment(args)
    try:
        result = environment.redact_source(args.source_id)
    finally:
        environment.close()
    _dump(result)
    return 0


def _cmd_memory(args, action: str) -> int:
    environment = _open_environment(args)
    try:
        if action == "redact":
            result = environment.redact_memory(args.memory_id)
        else:
            result = environment.delete_memory(args.memory_id)
    finally:
        environment.close()
    _dump(result)
    return 0


def _cmd_verify(args) -> int:
    store = EvaluationStore(args.root, tenant=args.tenant)
    environment = _open_environment(args)
    try:
        provenance = environment.pool.verify_provenance(args.tenant)
        status = environment.pool.status(args.tenant)
    finally:
        environment.close()
    store_result = store.verify()
    ok = store_result["ok"] and provenance.get("ok", False)
    _dump({
        "ok": ok,
        "store": store_result,
        "provenance": provenance,
        "reconciliation": status.get("reconciliation"),
        "checkpoint": status.get("checkpoint"),
    })
    return 0 if ok else 1


def _cmd_status(args) -> int:
    environment = _open_environment(args)
    try:
        status = environment.pool.status(args.tenant)
        readiness = environment.pool.readiness()
    finally:
        environment.close()
    _dump({
        "tenant": args.tenant,
        "readiness": readiness,
        "status": status,
    })
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            return _cmd_init(args)
        if args.command == "ingest":
            return _cmd_ingest(args)
        if args.command == "list-sources":
            return _cmd_list_sources(args)
        if args.command == "delete-source":
            return _cmd_delete_source(args)
        if args.command == "redact-source":
            return _cmd_redact_source(args)
        if args.command == "delete-memory":
            return _cmd_memory(args, "delete")
        if args.command == "redact-memory":
            return _cmd_memory(args, "redact")
        if args.command == "verify":
            return _cmd_verify(args)
        if args.command == "status":
            return _cmd_status(args)
    except (EvaluationStoreError, SourceFormatError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
