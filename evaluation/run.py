"""Repeatable, versioned evaluation queries over the dedicated store.

Expected answers live only in the query file supplied on the command line.
They are used exclusively for scoring, never passed to the engine, and never
written into the engine's checkpoint or provenance ledger.  Results are
immutable, versioned directories under ``results/``; failures are recorded
explicitly rather than dropped.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from minemos_service.engine_pool import EnginePoolError

from .formats import normalize_text
from .store import (
    DEFAULT_TENANT,
    EvaluationEnvironment,
    EvaluationStore,
    EvaluationStoreError,
    _atomic_write_json,
    _atomic_write_jsonl,
)

QUERY_TYPES = (
    "retrieval", "temporal", "contradiction", "history", "context",
    "duplicate",
)


class QueryFormatError(ValueError):
    """A query file is malformed."""


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="milliseconds")


def _read_queries(path: Path) -> Tuple[List[Dict[str, Any]], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise QueryFormatError(f"cannot read query file {path}: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QueryFormatError("query file is not UTF-8") from exc
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            entries = json.loads(text)
        except json.JSONDecodeError as exc:
            raise QueryFormatError(
                f"query file is not valid JSON: {exc}") from exc
        if not isinstance(entries, list):
            raise QueryFormatError("query file must contain a JSON array")
    else:
        entries = []
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QueryFormatError(
                    f"line {number} is not valid JSON: {exc}") from exc
            if not isinstance(entry, dict):
                raise QueryFormatError(f"line {number} must be an object")
            entries.append(entry)
    if not entries:
        raise QueryFormatError("query file contains no queries")
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        query_id = entry.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise QueryFormatError(
                f"query {index} needs a non-empty query_id")
        if query_id in seen:
            raise QueryFormatError(f"duplicate query_id {query_id!r}")
        seen.add(query_id)
        query_type = entry.get("type", "retrieval")
        if query_type not in QUERY_TYPES:
            raise QueryFormatError(
                f"query {query_id!r} has unknown type {query_type!r}")
        entry.setdefault("type", query_type)
        if query_type in ("retrieval", "temporal", "contradiction",
                          "context"):
            if not isinstance(entry.get("query"), str) or not entry[
                    "query"].strip():
                raise QueryFormatError(
                    f"query {query_id!r} needs a non-empty query string")
        if query_type == "history":
            if not isinstance(entry.get("subject"), str) or not entry[
                    "subject"].strip():
                raise QueryFormatError(
                    f"query {query_id!r} needs a subject")
        if query_type == "duplicate":
            if not isinstance(entry.get("source_id"), str):
                raise QueryFormatError(
                    f"query {query_id!r} needs a source_id")
            count = entry.get("expected_count")
            if isinstance(count, bool) or not isinstance(count, int):
                raise QueryFormatError(
                    f"query {query_id!r} needs an integer expected_count")
        if query_type == "temporal" and entry.get("as_of") is None:
            raise QueryFormatError(
                f"query {query_id!r} is temporal and needs as_of")
        for key in ("expected", "expected_inactive"):
            if key in entry:
                value = entry[key]
                if not isinstance(value, list):
                    raise QueryFormatError(
                        f"query {query_id!r} {key} must be a list")
                entry[key] = [_fact(item) for item in value]
        if "expected_sequence" in entry:
            sequence = entry["expected_sequence"]
            if not isinstance(sequence, list) or not sequence:
                raise QueryFormatError(
                    f"query {query_id!r} needs a non-empty "
                    "expected_sequence")
            if not all(isinstance(item, str) and item.strip()
                       for item in sequence):
                raise QueryFormatError(
                    f"query {query_id!r} expected_sequence must contain "
                    "non-empty strings")
        if (query_type in ("retrieval", "temporal", "contradiction")
                and not entry.get("expected")
                and not entry.get("expected_inactive")):
            raise QueryFormatError(
                f"query {query_id!r} needs expected or expected_inactive "
                "facts")
        as_of = entry.get("as_of")
        if as_of is not None:
            if isinstance(as_of, bool) or not isinstance(as_of, (int, float)):
                raise QueryFormatError(
                    f"query {query_id!r} as_of must be a number")
            if not math.isfinite(float(as_of)):
                raise QueryFormatError(
                    f"query {query_id!r} as_of must be finite")
        top_k = entry.get("top_k", 5)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not (
                1 <= top_k <= 100):
            raise QueryFormatError(
                f"query {query_id!r} top_k must be in 1..100")
    return entries, raw


def _fact(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        raise QueryFormatError("expected facts must be objects")
    subject = value.get("subject")
    predicate = value.get("predicate")
    obj = value.get("object")
    if not isinstance(subject, str) or not isinstance(predicate, str):
        raise QueryFormatError(
            "expected facts need string subject and predicate")
    if obj is not None and not isinstance(obj, str):
        raise QueryFormatError("expected fact object must be a string")
    return {"subject": subject, "predicate": predicate,
            "object": obj if obj is not None else ""}


def _render_fact(subject: str, predicate: str, obj: str) -> str:
    return normalize_text(f"{subject} {predicate} {obj}")


def _row_fact(row: Dict[str, Any]) -> str:
    subject = str(row.get("subject") or "")
    predicate = str(row.get("predicate") or "")
    text = str(row.get("text") or "")
    prefix = f"{subject} {predicate} "
    obj = text[len(prefix):] if text.startswith(prefix) else text
    return _render_fact(subject, predicate, obj)


def _match(row: Dict[str, Any], fact: Dict[str, str]) -> bool:
    if fact.get("object"):
        return _row_fact(row) == _render_fact(
            fact["subject"], fact["predicate"], fact["object"])
    return (normalize_text(str(row.get("subject") or ""))
            == normalize_text(fact["subject"])
            and normalize_text(str(row.get("predicate") or ""))
            == normalize_text(fact["predicate"]))


def _matches(rows: Sequence[Dict[str, Any]],
             facts: Sequence[Dict[str, str]]) -> List[int]:
    """Returns the first rank (1-based) of each expected fact, 0 if absent."""
    ranks: List[int] = []
    for fact in facts:
        rank = 0
        for index, row in enumerate(rows, start=1):
            if _match(row, fact):
                rank = index
                break
        ranks.append(rank)
    return ranks


def recall_at_k(ranks: Sequence[int], k: int) -> float:
    if not ranks:
        return 1.0
    return sum(1 for rank in ranks if 0 < rank <= k) / len(ranks)


def reciprocal_rank(ranks: Sequence[int]) -> float:
    present = [rank for rank in ranks if rank > 0]
    return 1.0 / min(present) if present else 0.0


def ndcg_at_k(ranks: Sequence[int], k: int) -> float:
    if not ranks:
        return 1.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in ranks if 0 < rank <= k)
    ideal = sum(
        1.0 / math.log2(index + 1)
        for index in range(1, min(len(ranks), k) + 1))
    return dcg / ideal if ideal else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _rss_bytes() -> Optional[int]:
    try:
        if os.name == "nt":
            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            counters = _Counters()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            psapi = ctypes.windll.psapi
            psapi.GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong]
            psapi.GetProcessMemoryInfo.restype = ctypes.c_bool
            if psapi.GetProcessMemoryInfo(
                    process, ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
            return None
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value) if sys.platform == "darwin" else int(value) * 1024
    except Exception:
        return None


def _retrieved(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep = ("memory_id", "subject", "predicate", "text", "similarity",
            "confidence", "update_count", "is_active", "valid_from",
            "valid_until")
    return [{key: row.get(key) for key in keep if key in row}
            for row in rows]


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for value in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            if value == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _duplicate_count(store: EvaluationStore, source_id: str) -> int:
    if not store.ingestions_path.exists():
        return 0
    count = 0
    for line in store.ingestions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("source_id") == source_id and entry.get("duplicate_of"):
            count += 1
    return count


def evaluate_query(environment: EvaluationEnvironment,
                   query: Dict[str, Any]) -> Dict[str, Any]:
    query_type = query["type"]
    top_k = int(query.get("top_k", 5))
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}
    failure: Optional[str] = None
    passed = False
    try:
        if query_type == "history":
            rows = environment.pool.history(
                environment.tenant, query["subject"],
                predicate=query.get("predicate"))
            expected = [normalize_text(str(item))
                        for item in query.get("expected_sequence", [])]
            actual = []
            for row in rows:
                text = str(row.get("text") or "")
                subject = str(row.get("subject") or "")
                predicate = str(row.get("predicate") or "")
                prefix = f"{subject} {predicate} "
                actual.append(normalize_text(
                    text[len(prefix):] if text.startswith(prefix) else text))
            lcs = _lcs_length(expected, actual)
            coverage = (
                sum(1 for item in expected if item in actual) / len(expected)
                if expected else 1.0)
            metrics = {
                "sequence_accuracy": lcs / len(expected) if expected else 1.0,
                "coverage": coverage,
                "exact_order": actual == expected,
                "expected_length": len(expected),
                "actual_length": len(actual),
            }
            passed = bool(expected) and coverage == 1.0 and lcs == len(expected)
            if not passed:
                failure = "history_sequence_mismatch"
        elif query_type == "duplicate":
            expected = int(query["expected_count"])
            actual = _duplicate_count(environment.store, query["source_id"])
            metrics = {"expected_count": expected, "actual_count": actual}
            passed = actual == expected
            if not passed:
                failure = "duplicate_count_mismatch"
        else:
            as_of = query.get("as_of")
            rows = environment.pool.query(
                environment.tenant, query["query"], top_k=top_k,
                as_of=as_of)
            expected = [_fact(item) for item in query.get("expected", [])]
            if (not expected and not query.get("expected_sequence")
                    and not query.get("expected_inactive")):
                raise QueryFormatError(
                    f"query {query['query_id']} has no expected facts")
            if query_type == "context":
                sequence = [normalize_text(str(item))
                            for item in query["expected_sequence"]]
                ranks: List[int] = []
                for item in sequence:
                    rank = 0
                    for index, row in enumerate(rows, start=1):
                        if item in normalize_text(str(row.get("text") or "")):
                            rank = index
                            break
                    ranks.append(rank)
                coverage = sum(1 for rank in ranks if rank > 0) / len(
                    sequence) if sequence else 1.0
                found_ranks = [rank for rank in ranks if rank > 0]
                order_ok = found_ranks == sorted(found_ranks)
                metrics = {
                    "coverage": coverage,
                    "order_ok": order_ok,
                    "ranks": ranks,
                }
                passed = coverage == 1.0 and order_ok
                if not passed:
                    failure = "context_reconstruction_mismatch"
            else:
                ranks = _matches(rows, expected)
                metrics = {
                    "recall_at_k": recall_at_k(ranks, top_k),
                    "mrr": reciprocal_rank(ranks),
                    "ndcg_at_k": ndcg_at_k(ranks, top_k),
                    "expected_count": len(expected),
                    "matched_count": sum(1 for rank in ranks if rank > 0),
                    "ranks": ranks,
                }
                passed = all(0 < rank <= top_k for rank in ranks)
                if not passed:
                    failure = "expected_fact_not_retrieved"
                if query_type in ("temporal", "contradiction"):
                    inactive = [_fact(item) for item in
                                query.get("expected_inactive", [])]
                    leaked = [index for index, row in enumerate(rows, start=1)
                              if any(_match(row, fact) for fact in inactive)]
                    metrics["inactive_leaked"] = leaked
                    if leaked:
                        passed = False
                        failure = "superseded_fact_leaked"
                    if query_type == "contradiction" and not query.get(
                            "expected_inactive"):
                        # A contradiction check is only meaningful when the
                        # superseded value is named explicitly.
                        metrics["warning"] = (
                            "expected_inactive is empty; only active-state "
                            "retrieval was checked")
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        passed = False
    duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
    result = {
        "query_id": query["query_id"],
        "type": query_type,
        "passed": passed,
        "metrics": metrics,
        "retrieved": _retrieved(rows),
        "duration_ms": duration_ms,
    }
    if failure:
        result["failure"] = failure
    return result


def run(environment: EvaluationEnvironment, queries: Sequence[Dict[str, Any]],
        query_path: Path, query_bytes: bytes,
        baseline: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    digest = hashlib.sha256(query_bytes).hexdigest()
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_id = f"{stamp}-{digest[:8]}"
    run_dir = environment.store.new_run_dir(run_id)
    started_at = _now_iso()
    started = time.perf_counter()
    rss_before = _rss_bytes()
    results = [evaluate_query(environment, query) for query in queries]
    duration = time.perf_counter() - started
    rss_after = _rss_bytes()
    durations = [result["duration_ms"] for result in results]
    by_type: Dict[str, Dict[str, Any]] = {}
    for result in results:
        bucket = by_type.setdefault(result["type"], {
            "count": 0, "passed": 0, "metrics": {}, "failures": 0})
        bucket["count"] += 1
        bucket["passed"] += 1 if result["passed"] else 0
        bucket["failures"] += 0 if result["passed"] else 1
        for key, value in result["metrics"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                bucket["metrics"].setdefault(key, []).append(float(value))
    for bucket in by_type.values():
        bucket["pass_rate"] = (
            bucket["passed"] / bucket["count"] if bucket["count"] else 0.0)
        bucket["metrics"] = {
            key: round(statistics.fmean(values), 4)
            for key, values in bucket["metrics"].items() if values
        }
    passed = sum(1 for result in results if result["passed"])
    status = environment.pool.status(environment.tenant)
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": _now_iso(),
        "query_file": str(query_path),
        "query_sha256": digest,
        "query_count": len(queries),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "by_type": by_type,
        "failure_records": [
            {"query_id": result["query_id"], "type": result["type"],
             "failure": result.get("failure")}
            for result in results if not result["passed"]],
        "latency_ms": {
            "p50": round(_percentile(durations, 0.50), 3),
            "p95": round(_percentile(durations, 0.95), 3),
            "max": round(max(durations), 3) if durations else 0.0,
        },
        "resource": {
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "engine_footprint_bytes": status["engine"].get("footprint_bytes"),
            "duration_seconds": round(duration, 3),
        },
        "engine": status["engine"],
        "retrieval": status["retrieval"],
        "provenance": status["provenance"],
        "checkpoint": status["checkpoint"],
    }
    if baseline is not None:
        summary["baseline_comparison"] = _compare(summary, baseline)
    metadata = {
        "run_id": run_id,
        "tenant": environment.tenant,
        "store": str(environment.store.root),
        "config": environment.config.describe(),
        "python": sys.version.split()[0],
        "started_at": started_at,
        "finished_at": summary["finished_at"],
        "query_file_sha256": digest,
    }
    _atomic_write_json(run_dir / "summary.json", summary)
    _atomic_write_json(run_dir / "metadata.json", metadata)
    _atomic_write_jsonl(run_dir / "per_query.jsonl", results)
    _atomic_write_jsonl(run_dir / "failures.jsonl",
                        ({
                            "query_id": result["query_id"],
                            "type": result["type"],
                            "failure": result.get("failure"),
                            "metrics": result["metrics"],
                        } for result in results if not result["passed"]))
    summary["result_dir"] = str(run_dir)
    return summary


def _compare(current: Dict[str, Any],
             baseline: Dict[str, Any]) -> Dict[str, Any]:
    comparison: Dict[str, Any] = {
        "baseline_run_id": baseline.get("run_id"),
        "pass_rate_delta": round(
            current["pass_rate"] - float(baseline.get("pass_rate", 0.0)), 4),
        "by_type": {},
    }
    base_types = baseline.get("by_type", {})
    for name, bucket in current["by_type"].items():
        base = base_types.get(name, {})
        comparison["by_type"][name] = {
            "pass_rate_delta": round(
                bucket["pass_rate"] - float(base.get("pass_rate", 0.0)), 4),
            "count": bucket["count"],
            "baseline_count": base.get("count", 0),
        }
    return comparison


def load_baseline(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / "summary.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QueryFormatError(f"cannot read baseline {path}: {exc}") from exc


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evaluation.run",
        description="Run versioned evaluation queries against the dedicated "
                    "store")
    parser.add_argument("--root", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--tenant", default=DEFAULT_TENANT)
    parser.add_argument("--retrieval-mode",
                        choices=("canonical", "legacy", "production"),
                        default="canonical")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--qdrant-api-key", default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        queries, raw = _read_queries(Path(args.queries))
        baseline = load_baseline(args.baseline) if args.baseline else None
        environment = EvaluationEnvironment.open(
            args.root,
            tenant=args.tenant,
            retrieval_mode=args.retrieval_mode,
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key,
        )
        try:
            summary = run(environment, queries, Path(args.queries), raw,
                          baseline=baseline)
        finally:
            environment.close()
    except (EvaluationStoreError, QueryFormatError, EnginePoolError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
