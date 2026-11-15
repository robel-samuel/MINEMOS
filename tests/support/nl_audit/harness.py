"""Controlled A-E natural-language robustness audit harness.

The harness uses the existing MINEMOS ingestion and retrieval paths unchanged.
Expected values are read only after the engine has answered; they are never
passed into ``absorb`` or ``query``.

Reported audit numbers always use the real MiniLM dense encoder and the real
NLI cross-encoder.  ``stub=True`` exists for fast mechanics tests only.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import platform
import statistics
import sys
import time
import unicodedata
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tests.support.nl_audit.dataset import (
    DEFAULT_SEED,
    generate_cases,
    generate_lifecycle,
    load_jsonl,
    sha256_file,
    validate_cases,
    validate_lifecycle,
)
from minemos_engine.black_hole_memory import BlackHoleMemory
from minemos_engine.structured_candidate import Candidate

AUDIT_DIR = Path(__file__).resolve().parent
DATA_DIR = AUDIT_DIR / "data"
CASES_PATH = DATA_DIR / "cases.jsonl"
LIFECYCLE_PATH = DATA_DIR / "lifecycle.jsonl"

TOP_K = 10
QDRANT_CANDIDATE_LIMIT = 128
RETRIEVAL_METHODS = ("minemos_dense", "bm25", "qdrant_dense", "hybrid")
ELIGIBILITY_METHOD = "minemos_dense+eligibility"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def normalize(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join(value.split())


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(float(statistics.fmean(values)), 6) if values else None


def _percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return round(float(ordered[lower]), 4)
    weight = position - lower
    return round(float(ordered[lower] * (1 - weight)
                       + ordered[upper] * weight), 4)


def rss_mb() -> Optional[float]:
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
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb):
            return round(counters.WorkingSetSize / 1048576.0, 2)
        return None
    try:
        import resource

        return round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)
    except Exception:  # noqa: BLE001 - optional observability
        return None


def _rank_of(rows: Sequence[Dict[str, Any]], memory_id: Any) -> int:
    if memory_id is None:
        return 0
    target = str(memory_id)
    for rank, row in enumerate(rows, start=1):
        if str(row.get("memory_id")) == target:
            return rank
    return 0


def _row_text(row: Dict[str, Any]) -> str:
    return str(row.get("text") or "")


def _fact_in_text(text: str, subject: str, obj: str) -> bool:
    haystack = normalize(text)
    return normalize(subject) in haystack and normalize(obj) in haystack


def _slot_matches_fact(mem: BlackHoleMemory, slot: Any,
                       fact: Dict[str, Any]) -> bool:
    text = mem.slot_texts.get(id(slot), slot.text or "")
    return _fact_in_text(text, fact["subject"], fact["object"])


def _slot_by_id(mem: BlackHoleMemory, slot_id: Any) -> Optional[Any]:
    if slot_id is None:
        return None
    for slot in mem.memory.slots:
        if id(slot) == slot_id:
            return slot
    return None


# ---------------------------------------------------------------------------
# Components and memory construction
# ---------------------------------------------------------------------------


def build_components(stub: bool = False) -> Dict[str, Any]:
    if stub:
        from tests.support.nl_audit.stubs import (
            EqualityVerifier,
            HashEncoder,
            StubGate,
        )
        encoder = HashEncoder()
        return {
            "label": "deterministic-stub",
            "encoder": encoder,
            "gate": StubGate(),
            "verifier": EqualityVerifier(),
            "dense_model": "sha1-token-hash-384",
            "nli_model": "normalized-equality-stub",
        }

    from minemos_engine.intelligence.nli import NliSemanticGate
    from minemos_engine.intelligence.dense import DenseSemanticEncoder
    from minemos_engine.intelligence.verification import CalibratedSemanticVerifier

    encoder = DenseSemanticEncoder()
    gate = NliSemanticGate()
    return {
        "label": "real",
        "encoder": encoder,
        "gate": gate,
        "verifier": CalibratedSemanticVerifier(gate),
        "dense_model": getattr(encoder, "model_name", type(encoder).__name__),
        "nli_model": getattr(gate, "model_name", type(gate).__name__),
    }


def _new_memory(components: Dict[str, Any], capacity: int,
                qdrant_config: Optional[Any] = None,
                strict_entity_identity: bool = False) -> BlackHoleMemory:
    return BlackHoleMemory(
        capacity=capacity,
        dim=4096,
        dense_encoder=components["encoder"],
        nli_gate=components["gate"],
        verifier=components["verifier"],
        retrieval_backend="canonical",
        qdrant_config=qdrant_config,
        strict_entity_identity=strict_entity_identity,
    )


def _qdrant_config(collection: str, path: str) -> Any:
    from minemos_engine.retrieval import QdrantConfig

    return QdrantConfig(
        path=path,
        collection=collection,
        tenant_id="natural-language-audit",
        candidate_limit=QDRANT_CANDIDATE_LIMIT,
    )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_datasets(seed: int = DEFAULT_SEED, regenerate: bool = False,
                  limit: Optional[int] = None,
                  lifecycle_limit: Optional[int] = None
                  ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]],
                             Dict[str, Any]]:
    if regenerate or not CASES_PATH.exists() or not LIFECYCLE_PATH.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cases = generate_cases(seed)
        lifecycle = generate_lifecycle(seed)
        CASES_PATH.write_text(
            "".join(json.dumps(case, ensure_ascii=False, sort_keys=True)
                    + "\n" for case in cases),
            encoding="utf-8")
        LIFECYCLE_PATH.write_text(
            "".join(json.dumps(scenario, ensure_ascii=False, sort_keys=True)
                    + "\n" for scenario in lifecycle),
            encoding="utf-8")
    else:
        cases = load_jsonl(CASES_PATH)
        lifecycle = load_jsonl(LIFECYCLE_PATH)

    cases_validation = validate_cases(cases)
    lifecycle_validation = validate_lifecycle(lifecycle)
    if limit is not None:
        kept = []
        per_level: Dict[str, int] = defaultdict(int)
        for case in cases:
            if per_level[case["level"]] < limit:
                kept.append(case)
                per_level[case["level"]] += 1
        cases = kept
    if lifecycle_limit is not None:
        lifecycle = lifecycle[:lifecycle_limit]
    metadata = {
        "seed": seed,
        "cases_path": str(CASES_PATH),
        "lifecycle_path": str(LIFECYCLE_PATH),
        "cases_sha256": sha256_file(CASES_PATH),
        "lifecycle_sha256": sha256_file(LIFECYCLE_PATH),
        "cases_validation": cases_validation,
        "lifecycle_validation": lifecycle_validation,
        "cases_used": len(cases),
        "lifecycle_used": len(lifecycle),
    }
    return cases, lifecycle, metadata


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _bm25_rows(mem: BlackHoleMemory, query_text: str,
               top_k: int) -> List[Dict[str, Any]]:
    with mem._lock:
        slots = list(mem.memory.slots)
        id_to_slot_index = {id(slot): idx for idx, slot in enumerate(slots)}
        hits = mem.bm25_index.query(
            query_text=query_text,
            top_k=max(top_k * 4, top_k),
            id_to_slot_index=id_to_slot_index,
        )
        rows: List[Dict[str, Any]] = []
        for idx, score in hits:
            if not (0 <= idx < len(slots)):
                continue
            slot = slots[idx]
            if not slot.is_active:
                continue
            rows.append({
                "slot_index": idx,
                "slot_id": id(slot),
                "memory_id": slot.memory_id,
                "similarity": round(float(score), 4),
                "subject": slot.debug_subject,
                "predicate": slot.debug_predicate,
                "text": mem.slot_texts.get(id(slot), slot.text),
                "confidence": round(float(slot.confidence), 4),
                "update_count": slot.update_count,
                "is_active": slot.is_active,
                "valid_from": slot.valid_from,
                "valid_until": slot.valid_until,
            })
            if len(rows) >= top_k:
                break
        return rows


def _query_rows(mem: BlackHoleMemory, method: str, query_text: str,
                top_k: int) -> List[Dict[str, Any]]:
    if method == "bm25":
        return _bm25_rows(mem, query_text, top_k)
    return list(mem.query(
        query_text,
        top_k=top_k,
        hybrid=(method == "hybrid"),
    ))


def _target_memory_id(case: Dict[str, Any],
                      ingested: Dict[str, Dict[str, Dict[str, Any]]]) -> Any:
    target = case.get("target_observation")
    if target is None:
        return None
    entry = ingested.get(case["case_id"], {}).get(target)
    return entry.get("memory_id") if entry else None


def _observation_values(case: Dict[str, Any], noisy: bool
                        ) -> Iterable[Tuple[Dict[str, Any],
                                            str, str, str]]:
    for observation in case["observations"]:
        if noisy:
            yield (observation, observation["noisy_subject"],
                   observation["noisy_predicate"], observation["noisy_object"])
        else:
            yield (observation, observation["subject"],
                   observation["predicate"], observation["object"])


def _ingest_case(mem: BlackHoleMemory, case: Dict[str, Any],
                 noisy: bool) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for observation, subject, predicate, obj in _observation_values(
            case, noisy):
        result = mem.absorb(
            Candidate(subject, predicate, obj,
                      timestamp=float(observation["timestamp"])),
            timestamp=float(observation["timestamp"]),
        )
        results[observation["id"]] = {
            "memory_id": result.get("memory_id"),
            "slot_id": result.get("slot_id"),
            "action": result.get("action"),
            "update_count": result.get("update_count"),
        }
    return results


def _build_corpus(components: Dict[str, Any], cases: Sequence[Dict[str, Any]],
                  noisy: bool, capacity: int,
                  qdrant_config: Optional[Any],
                  strict_entity_identity: bool = False) -> Tuple[
                      BlackHoleMemory, Dict[str, Dict[str, Dict[str, Any]]],
                      Dict[str, Any]]:
    mem = _new_memory(components, capacity, qdrant_config,
                      strict_entity_identity=strict_entity_identity)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    rss_before = rss_mb()
    ingested: Dict[str, Dict[str, Dict[str, Any]]] = {}
    actions: Counter = Counter()
    for case in cases:
        results = _ingest_case(mem, case, noisy)
        ingested[case["case_id"]] = results
        for entry in results.values():
            actions[str(entry.get("action"))] += 1
    stats = {
        "facts_ingested": sum(len(v) for v in ingested.values()),
        "final_slots": len(mem.memory.slots),
        "active_slots": sum(1 for slot in mem.memory.slots if slot.is_active),
        "actions": dict(actions),
        "ingest_ms": round((time.perf_counter() - started_wall) * 1000.0, 1),
        "ingest_cpu_ms": round(
            (time.process_time() - started_cpu) * 1000.0, 1),
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_mb(),
    }
    return mem, ingested, stats


def _retrieval_record(case: Dict[str, Any], condition: str, method: str,
                      query_kind: str, ingest_kind: str,
                      target_memory_id: Any, rows: Sequence[Dict[str, Any]],
                      latency_ms: float) -> Dict[str, Any]:
    rank = _rank_of(rows, target_memory_id)
    expected = case.get("expected") or {}
    top1 = rows[0] if rows else {}
    return {
        "record_type": "retrieval",
        "case_id": case["case_id"],
        "level": case["level"],
        "category": case["category"],
        "noise_family": case["noise_family"],
        "condition": condition,
        "method": method,
        "query_kind": query_kind,
        "ingest_kind": ingest_kind,
        "rank": rank,
        "recall@1": 1.0 if 1 <= rank <= 1 else 0.0,
        "recall@5": 1.0 if 1 <= rank <= 5 else 0.0,
        "recall@10": 1.0 if 1 <= rank <= 10 else 0.0,
        "mrr": round(1.0 / rank, 6) if rank else 0.0,
        "ndcg@5": (round(1.0 / math.log2(rank + 1), 6)
                   if 1 <= rank <= 5 else 0.0),
        "fact_top1": (
            1.0 if top1 and _fact_in_text(
                _row_text(top1),
                expected.get("subject", ""),
                expected.get("object", ""))
            else 0.0),
        "top1_similarity": top1.get("similarity"),
        "top1_memory_id": top1.get("memory_id"),
        "top1_text": _row_text(top1),
        "target_memory_id": target_memory_id,
        "target_text": (case.get("observations") and
                        case["observations"][-1].get("clean_text")),
        "latency_ms": round(latency_ms, 4),
        "top5": [
            {"memory_id": row.get("memory_id"),
             "text": _row_text(row),
             "similarity": row.get("similarity")}
            for row in rows[:5]
        ],
    }


def _unsupported_record(case: Dict[str, Any], condition: str, method: str,
                        rows: Sequence[Dict[str, Any]], latency_ms: float,
                        ingest_kind: str,
                        eligibility: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
    top1 = rows[0] if rows else {}
    query_subject = case.get("query_subject") or case.get("subject")
    expected = case.get("expected") or {}
    status = eligibility.get("status") if eligibility else None
    return {
        "eligibility_status": status,
        "eligibility_reason": (
            eligibility.get("reason") if eligibility else None),
        "eligibility_focus": (
            eligibility.get("focus") if eligibility else None),
        "eligibility_attributes": (
            eligibility.get("query_attributes") if eligibility else None),
        "answered": (None if status is None
                     else int(status == "answered")),
        "abstained": (None if status is None
                      else int(status == "abstained")),
        "unverified": (None if status is None
                       else int(status == "unverified")),
        "record_type": "unsupported",
        "case_id": case["case_id"],
        "level": case["level"],
        "category": case["category"],
        "noise_family": case["noise_family"],
        "condition": condition,
        "method": method,
        "ingest_kind": ingest_kind,
        "subcategory": case.get("subcategory"),
        "non_empty": 1 if rows else 0,
        "top1_similarity": top1.get("similarity"),
        "top1_same_subject": (
            1 if top1 and normalize(top1.get("subject"))
            == normalize(query_subject) else 0),
        "top1_same_stored_subject": (
            1 if top1 and normalize(top1.get("subject"))
            == normalize(case.get("subject")) else 0),
        "accepted_at_0_5": (
            1 if rows and float(top1.get("similarity") or 0.0) >= 0.5 else 0),
        "accepted_at_0_7": (
            1 if rows and float(top1.get("similarity") or 0.0) >= 0.7 else 0),
        "query": case.get("clean_query"),
        "top1_text": _row_text(top1),
        "latency_ms": round(latency_ms, 4),
        "top5": [
            {"memory_id": row.get("memory_id"),
             "text": _row_text(row),
             "similarity": row.get("similarity")}
            for row in rows[:5]
        ],
    }


def _run_memory_queries(mem: BlackHoleMemory,
                        cases: Sequence[Dict[str, Any]],
                        ingested: Dict[str, Dict[str, Dict[str, Any]]],
                        ingest_kind: str,
                        methods: Sequence[str],
                        skip_qdrant: bool,
                        rebuild_info: Optional[Dict[str, Any]] = None,
                        eligibility: bool = False
                        ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]],
                                   Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    unsupported: List[Dict[str, Any]] = []
    resources: Dict[str, Any] = {"rebuild": rebuild_info}

    if "qdrant_dense" in methods or "hybrid" in methods:
        if skip_qdrant:
            methods = [method for method in methods
                       if method not in ("qdrant_dense", "hybrid")]
        else:
            mem.set_retrieval_backend("qdrant")
            retriever = mem._get_retrieval_adapter().ensure_retriever()
            started = time.perf_counter()
            rebuilt = retriever.rebuild()
            resources["rebuild"] = {
                "points": rebuilt.get("points"),
                "ms": round((time.perf_counter() - started) * 1000.0, 1),
            }

    for method in methods:
        if method in ("qdrant_dense", "hybrid"):
            mem.set_retrieval_backend("qdrant")
        else:
            mem.set_retrieval_backend("canonical")
        for query_kind in ("clean", "noisy"):
            query_key = f"{query_kind}_query"
            condition = f"{ingest_kind}_{query_kind}_query"
            for case in cases:
                query_text = case[query_key]
                eligibility_info: Optional[Dict[str, Any]] = None
                started = time.perf_counter()
                try:
                    if method == ELIGIBILITY_METHOD:
                        outcome = mem.eligible_query(query_text, top_k=TOP_K)
                        rows = outcome["rows"]
                        eligibility_info = outcome
                    else:
                        rows = _query_rows(mem, method, query_text, TOP_K)
                    error = None
                except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                    rows = []
                    error = f"{type(exc).__name__}: {exc}"
                latency_ms = (time.perf_counter() - started) * 1000.0
                # Abstained and unverified outcomes are not answers.  Score
                # them as empty so the gate's recall reflects what it returned
                # to a caller, not the ungated rows it inspected.
                scored_rows = (
                    rows if eligibility_info is None
                    or eligibility_info["status"] == "answered" else [])
                if case["kind"] == "unsupported":
                    if method in ("minemos_dense", ELIGIBILITY_METHOD):
                        record = _unsupported_record(
                            case, condition, method, scored_rows, latency_ms,
                            ingest_kind, eligibility_info)
                        record["error"] = error
                        if eligibility_info is not None:
                            record["eligibility_inspected_rows"] = len(rows)
                        unsupported.append(record)
                    continue
                record = _retrieval_record(
                    case, condition, method, query_kind, ingest_kind,
                    _target_memory_id(case, ingested), scored_rows, latency_ms)
                record["error"] = error
                if eligibility_info is not None:
                    record["eligibility_status"] = eligibility_info["status"]
                    record["eligibility_reason"] = eligibility_info["reason"]
                    record["eligibility_focus"] = eligibility_info["focus"]
                    record["eligibility_attributes"] = (
                        eligibility_info["query_attributes"])
                    record["eligibility_inspected_rows"] = len(rows)
                records.append(record)
        _log(f"    {ingest_kind}: {method} complete")
    return records, unsupported, resources


def _metric_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    latencies = [float(r["latency_ms"]) for r in records]
    output = {
        "n": len(records),
        "recall@1": _mean([r["recall@1"] for r in records]),
        "recall@5": _mean([r["recall@5"] for r in records]),
        "recall@10": _mean([r["recall@10"] for r in records]),
        "mrr": _mean([r["mrr"] for r in records]),
        "ndcg@5": _mean([r["ndcg@5"] for r in records]),
        "fact_top1": _mean([r["fact_top1"] for r in records]),
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "latency_max_ms": round(max(latencies), 4) if latencies else None,
        "errors": sum(1 for r in records if r.get("error")),
    }
    gated = [r for r in records if r.get("eligibility_status") is not None]
    if gated:
        output["eligibility_answered_rate"] = _mean(
            [1.0 if r["eligibility_status"] == "answered" else 0.0
             for r in gated])
        output["eligibility_abstained_rate"] = _mean(
            [1.0 if r["eligibility_status"] == "abstained" else 0.0
             for r in gated])
        output["eligibility_unverified_rate"] = _mean(
            [1.0 if r["eligibility_status"] == "unverified" else 0.0
             for r in gated])
    return output


def _group_summary(records: Sequence[Dict[str, Any]],
                   keys: Sequence[str]) -> Dict[str, Any]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in keys)].append(record)
    result: Dict[str, Any] = {}
    for key, group in grouped.items():
        if len(keys) == 1:
            label = str(key[0])
        else:
            label = "|".join(str(part) for part in key)
        result[label] = _metric_summary(group)
    return dict(sorted(result.items()))


def _degradation(by_condition: Dict[str, Any]) -> Dict[str, Any]:
    baseline = by_condition.get("clean_ingest_clean_query")
    if not baseline:
        return {}
    output: Dict[str, Any] = {}
    for condition, metrics in by_condition.items():
        if condition == "clean_ingest_clean_query":
            continue
        entry: Dict[str, Any] = {}
        for metric in ("recall@1", "recall@5", "recall@10", "mrr",
                       "ndcg@5", "fact_top1"):
            base = baseline.get(metric)
            value = metrics.get(metric)
            if base is None or value is None:
                entry[metric] = None
            else:
                entry[metric] = round(value - base, 6)
        output[condition] = entry
    return output


def run_retrieval(cases: Sequence[Dict[str, Any]],
                  components: Dict[str, Any],
                  skip_qdrant: bool = False,
                  include_hybrid: bool = True,
                  eligibility: bool = False,
                  strict_entity_identity: bool = False) -> Dict[str, Any]:
    methods = list(RETRIEVAL_METHODS)
    if not include_hybrid:
        methods = [method for method in methods if method != "hybrid"]
    if skip_qdrant:
        methods = [method for method in methods
                   if method not in ("qdrant_dense", "hybrid")]
    if eligibility:
        methods.append(ELIGIBILITY_METHOD)
    capacity = max(64, 2 * len(cases) + 64)
    qdrant_root = os.path.join(
        os.environ.get("TEMP", "/tmp"),
        f"minemos_nl_audit_{uuid.uuid4().hex[:10]}")
    qdrant_configs: Dict[str, Any] = {}
    if not skip_qdrant:
        qdrant_configs = {
            "clean": _qdrant_config(
                f"nl_clean_{uuid.uuid4().hex[:10]}", qdrant_root),
            "noisy": _qdrant_config(
                f"nl_noisy_{uuid.uuid4().hex[:10]}", qdrant_root),
        }

    results: Dict[str, Any] = {
        "methods": methods,
        "corpora": {},
        "records": [],
        "unsupported": [],
    }

    for ingest_kind, noisy in (("clean_ingest", False),
                               ("noisy_ingest", True)):
        _log(f"  ingesting {ingest_kind} corpus ({len(cases)} cases)")
        mem, ingested, stats = _build_corpus(
            components, cases, noisy, capacity,
            qdrant_configs.get(ingest_kind.split("_")[0]),
            strict_entity_identity=strict_entity_identity)
        try:
            records, unsupported, resources = _run_memory_queries(
                mem, cases, ingested, ingest_kind, methods, skip_qdrant,
                eligibility=eligibility)
        finally:
            close = getattr(mem, "close", None)
            if close is not None:
                close()
        stats.update(resources)
        results["corpora"][ingest_kind] = stats
        results["records"].extend(records)
        results["unsupported"].extend(unsupported)

    retrieval_records = results.pop("records")
    by_condition = _group_summary(
        retrieval_records, ("method", "condition"))
    by_condition_method: Dict[str, Any] = {}
    for method in methods:
        method_records = [r for r in retrieval_records
                          if r["method"] == method]
        conditions = _group_summary(method_records, ("condition",))
        by_condition_method[method] = {
            "by_condition": conditions,
            "degradation": _degradation(conditions),
        }

    results["summary"] = {
        "by_method_condition": by_condition,
        "by_method": by_condition_method,
        "by_level": _group_summary(
            retrieval_records, ("method", "condition", "level")),
        "by_category": _group_summary(
            retrieval_records, ("method", "condition", "category")),
        "overall": _metric_summary(retrieval_records),
    }
    results["records"] = retrieval_records
    results["unsupported_summary"] = _unsupported_summary(
        results["unsupported"])
    return results


def _unsupported_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["method"], record["condition"],
                 record["subcategory"])].append(record)
    output: Dict[str, Any] = {}
    for (method, condition, subcategory), group in sorted(
            grouped.items()):
        sims = [float(r["top1_similarity"] or 0.0) for r in group]
        entry = {
            "n": len(group),
            "non_empty_rate": _mean([r["non_empty"] for r in group]),
            "top1_same_subject_rate": _mean(
                [r["top1_same_subject"] for r in group]),
            "accepted_at_0_5": _mean(
                [r["accepted_at_0_5"] for r in group]),
            "accepted_at_0_7": _mean(
                [r["accepted_at_0_7"] for r in group]),
            "top1_similarity_mean": round(float(statistics.fmean(sims)), 6),
            "top1_similarity_p95": _percentile(sims, 0.95),
        }
        gated = [r for r in group
                 if r.get("eligibility_status") is not None]
        if gated:
            entry["eligibility_answered_rate"] = _mean(
                [r["answered"] for r in gated])
            entry["eligibility_abstained_rate"] = _mean(
                [r["abstained"] for r in gated])
            entry["eligibility_unverified_rate"] = _mean(
                [r["unverified"] for r in gated])
            reasons = Counter(
                str(r.get("eligibility_reason")) for r in gated)
            entry["eligibility_reason_counts"] = dict(sorted(reasons.items()))
        output[f"{method}|{condition}|{subcategory}"] = entry
    return output


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def _scenario_values(scenario: Dict[str, Any], noisy: bool
                     ) -> Iterable[Tuple[Dict[str, Any],
                                         str, str, str]]:
    for observation in scenario["observations"]:
        if noisy:
            yield (observation, observation.get("noisy_subject",
                                                observation["subject"]),
                   observation.get("noisy_predicate", observation["predicate"]),
                   observation.get("noisy_object", observation["object"]))
        else:
            yield (observation, observation["subject"],
                   observation["predicate"], observation["object"])


def _run_scenario(scenario: Dict[str, Any], noisy: bool,
                  components: Dict[str, Any],
                  strict_entity_identity: bool = False) -> Dict[str, Any]:
    mem = _new_memory(components, capacity=20,
                      strict_entity_identity=strict_entity_identity)
    results: List[Dict[str, Any]] = []
    for observation, subject, predicate, obj in _scenario_values(
            scenario, noisy):
        result = mem.absorb(
            Candidate(subject, predicate, obj,
                      timestamp=float(observation["timestamp"])),
            timestamp=float(observation["timestamp"]),
        )
        result["subject"] = subject
        result["predicate"] = predicate
        result["object"] = obj
        results.append(result)

    slots = list(mem.memory.slots)
    active = [slot for slot in slots if slot.is_active]
    current_result = results[-1]
    if scenario["category"] in ("historical_statement", "near_miss_entity"):
        current_result = results[0]

    query = scenario["noisy_query"] if noisy else scenario["query"]
    current_rows = mem.query(query, top_k=5)
    current_top1 = current_rows[0] if current_rows else {}
    expected_object = scenario.get("expected_current_object")
    expected_subject = scenario.get("subject")
    multi_fact = scenario["category"] in (
        "cumulative_distinct", "similar_wording_distinct")
    distinct_facts = scenario.get("distinct_facts") or []

    both_facts_present = all(
        any(_slot_matches_fact(mem, slot, fact) for slot in active)
        for fact in distinct_facts) if distinct_facts else None

    current_target_ok = (
        str(current_top1.get("memory_id"))
        == str(current_result.get("memory_id"))
        if current_result.get("memory_id") is not None else False)
    current_object_ok = (
        expected_object is not None
        and normalize(expected_object) in normalize(_row_text(current_top1)))
    current_subject_ok = (
        normalize(current_top1.get("subject")) == normalize(expected_subject))

    historical_ok = None
    historical_rows: List[Dict[str, Any]] = []
    if scenario.get("as_of") is not None:
        historical_rows = mem.query(
            query, top_k=5, as_of=float(scenario["as_of"]))
        old_memory_id = results[0].get("memory_id")
        historical_ok = _rank_of(historical_rows, old_memory_id) > 0
    elif scenario.get("expected_historical_object") is not None:
        historical_ok = any(
            normalize(scenario["expected_historical_object"])
            in normalize(mem.slot_texts.get(id(slot), slot.text or ""))
            and not slot.is_active
            for slot in slots)

    supersession_ok = None
    if scenario.get("expected_final_action") == "temporal_update":
        old_slot = _slot_by_id(mem, results[0].get("slot_id"))
        new_slot = _slot_by_id(mem, results[-1].get("slot_id"))
        if old_slot is not None and new_slot is not None and old_slot is not new_slot:
            supersession_ok = (
                not old_slot.is_active
                and old_slot.superseded_by == id(new_slot)
                and new_slot.supersedes == id(old_slot)
                and old_slot.valid_until is not None
                and float(old_slot.valid_until) == float(new_slot.valid_from)
            )
        else:
            supersession_ok = False

    expected_update_count = scenario.get("expected_update_count")
    update_count_ok = None
    if expected_update_count is not None:
        update_count_ok = (
            results[-1].get("update_count") == expected_update_count)

    expected_action = scenario.get("expected_final_action")
    action_ok = (
        results[-1].get("action") == expected_action
        if expected_action is not None else None)

    allowed_subjects = {
        normalize(observation["subject"])
        for observation in scenario["observations"]}
    allowed_subjects.update(
        normalize(observation.get("noisy_subject",
                                  observation["subject"]))
        for observation in scenario["observations"])
    for fact in distinct_facts:
        allowed_subjects.add(normalize(fact["subject"]))
    contamination = any(
        normalize(slot.debug_subject) not in allowed_subjects
        for slot in slots)

    record = {
        "record_type": "lifecycle",
        "scenario_id": scenario["scenario_id"],
        "category": scenario["category"],
        "subcategory": scenario.get("subcategory"),
        "ingest_kind": "noisy_ingest" if noisy else "clean_ingest",
        "actions": [result.get("action") for result in results],
        "expected_final_action": expected_action,
        "action_ok": action_ok,
        "active_count": len(active),
        "expected_active_count": scenario["expected_active_count"],
        "active_count_ok": (
            len(active) == scenario["expected_active_count"]),
        "total_count": len(slots),
        "expected_total_count": scenario["expected_total_count"],
        "total_count_ok": (
            len(slots) == scenario["expected_total_count"]),
        "duplicate": (
            scenario["expected_active_count"] == 1 and len(active) > 1),
        "duplicate_total": (
            scenario["expected_total_count"] == 1 and len(slots) > 1),
        "false_merge": (
            scenario["expected_active_count"] > 1
            and len(active) < scenario["expected_active_count"]),
        "both_facts_present": both_facts_present,
        "current_target_ok": current_target_ok,
        "current_object_ok": current_object_ok,
        "current_subject_ok": current_subject_ok,
        "current_check_applicable": not multi_fact,
        "historical_ok": historical_ok,
        "historical_rows": [
            {"memory_id": row.get("memory_id"),
             "text": _row_text(row)}
            for row in historical_rows[:5]
        ],
        "supersession_ok": supersession_ok,
        "update_count_ok": update_count_ok,
        "contamination": contamination,
        "strict_entity_identity": mem.strict_entity_identity,
        "suppressed_value_merges": mem.total_suppressed_value_merges,
        "near_miss_separations": mem.total_near_miss_separations,
        "query": query,
        "current_top1": {
            "memory_id": current_top1.get("memory_id"),
            "text": _row_text(current_top1),
            "similarity": current_top1.get("similarity"),
        },
        "slots": [
            {
                "memory_id": slot.memory_id,
                "subject": slot.debug_subject,
                "text": mem.slot_texts.get(id(slot), slot.text),
                "is_active": slot.is_active,
                "valid_from": slot.valid_from,
                "valid_until": slot.valid_until,
                "supersedes": slot.supersedes,
                "superseded_by": slot.superseded_by,
                "update_count": slot.update_count,
            }
            for slot in slots
        ],
    }
    close = getattr(mem, "close", None)
    if close is not None:
        close()
    return record


def _rate(records: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    values = [record[key] for record in records
              if record.get(key) is not None]
    if not values:
        return None
    return round(sum(1 for value in values if value) / len(values), 6)


def run_lifecycle(scenarios: Sequence[Dict[str, Any]],
                  components: Dict[str, Any],
                  strict_entity_identity: bool = False) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for index, scenario in enumerate(scenarios, start=1):
        for noisy in (False, True):
            records.append(_run_scenario(
                scenario, noisy, components,
                strict_entity_identity=strict_entity_identity))
        if index % 20 == 0:
            _log(f"  lifecycle {index}/{len(scenarios)}")

    def summary(selected: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "n": len(selected),
            "action_accuracy": _rate(selected, "action_ok"),
            "active_count_accuracy": _rate(selected, "active_count_ok"),
            "total_count_accuracy": _rate(selected, "total_count_ok"),
            "duplicate_rate": _rate(selected, "duplicate"),
            "duplicate_total_rate": _rate(selected, "duplicate_total"),
            "false_merge_rate": _rate(selected, "false_merge"),
            "both_facts_present_rate": _rate(
                selected, "both_facts_present"),
            "current_target_rate": _rate(selected, "current_target_ok"),
            "current_object_rate": _rate(selected, "current_object_ok"),
            "current_subject_rate": _rate(selected, "current_subject_ok"),
            "historical_rate": _rate(selected, "historical_ok"),
            "supersession_rate": _rate(selected, "supersession_ok"),
            "update_count_rate": _rate(selected, "update_count_ok"),
            "contamination_rate": _rate(selected, "contamination"),
            "suppressed_value_merges": sum(
                int(r.get("suppressed_value_merges") or 0)
                for r in selected),
            "near_miss_separations": sum(
                int(r.get("near_miss_separations") or 0)
                for r in selected),
        }

    by_category: Dict[str, Any] = {}
    for category in sorted({r["category"] for r in records}):
        by_category[category] = {}
        for ingest_kind in ("clean_ingest", "noisy_ingest"):
            selected = [r for r in records
                        if r["category"] == category
                        and r["ingest_kind"] == ingest_kind]
            by_category[category][ingest_kind] = summary(selected)
    return {
        "strict_entity_identity": bool(strict_entity_identity),
        "by_category": by_category,
        "overall": {
            "clean_ingest": summary(
                [r for r in records if r["ingest_kind"] == "clean_ingest"]),
            "noisy_ingest": summary(
                [r for r in records if r["ingest_kind"] == "noisy_ingest"]),
        },
        "records": records,
    }


# ---------------------------------------------------------------------------
# Raw-sentence ingestion probe
# ---------------------------------------------------------------------------


def _raw_probe(components: Dict[str, Any], sentence: str,
               timestamp: float, query: str, intended_subject: str,
               intended_predicate: str, intended_object: str
               ) -> Dict[str, Any]:
    mem = _new_memory(components, capacity=5)
    result = mem.absorb(sentence, timestamp=timestamp)
    slot = _slot_by_id(mem, result.get("slot_id"))
    parsed_subject = slot.debug_subject if slot else None
    parsed_predicate = slot.debug_predicate if slot else None
    parsed_text = (mem.slot_texts.get(id(slot), slot.text)
                   if slot else result.get("text"))
    rows = mem.query(query, top_k=5)
    rank = _rank_of(rows, result.get("memory_id"))
    close = getattr(mem, "close", None)
    if close is not None:
        close()
    return {
        "sentence": sentence,
        "parsed_subject": parsed_subject,
        "parsed_predicate": parsed_predicate,
        "parsed_text": parsed_text,
        "intended_subject": intended_subject,
        "intended_predicate": intended_predicate,
        "intended_object": intended_object,
        "subject_match": normalize(parsed_subject)
        == normalize(intended_subject),
        "predicate_match": normalize(parsed_predicate)
        == normalize(intended_predicate),
        "object_present": normalize(intended_object)
        in normalize(parsed_text),
        "rank": rank,
        "recall@1": 1.0 if rank == 1 else 0.0,
        "recall@5": 1.0 if 1 <= rank <= 5 else 0.0,
        "action": result.get("action"),
        "query": query,
    }


def run_raw_ingestion(cases: Sequence[Dict[str, Any]],
                      components: Dict[str, Any],
                      limit: int = 100) -> Dict[str, Any]:
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_category[case["category"]].append(case)
    selected: List[Dict[str, Any]] = []
    cursor = 0
    while len(selected) < limit:
        added = False
        for category in sorted(by_category):
            items = by_category[category]
            if cursor < len(items) and len(selected) < limit:
                selected.append(items[cursor])
                added = True
        if not added:
            break
        cursor += 1

    records: List[Dict[str, Any]] = []
    for case in selected:
        target_id = case.get("target_observation")
        observation = next(
            (obs for obs in case["observations"]
             if obs["id"] == target_id), case["observations"][-1])
        sentence = observation.get("noisy_text") or observation["clean_text"]
        clean_sentence = observation["clean_text"]
        intended_subject = observation.get("noisy_subject",
                                           observation["subject"])
        intended_predicate = observation.get("noisy_predicate",
                                             observation["predicate"])
        intended_object = observation.get("noisy_object",
                                          observation["object"])
        timestamp = float(observation["timestamp"])
        noisy_probe = _raw_probe(
            components, sentence, timestamp, case["noisy_query"],
            intended_subject, intended_predicate, intended_object)
        clean_probe = _raw_probe(
            components, clean_sentence, timestamp, case["noisy_query"],
            observation["subject"], observation["predicate"],
            observation["object"])
        record = {
            "record_type": "raw_ingestion",
            "case_id": case["case_id"],
            "level": case["level"],
            "category": case["category"],
            "sentence": sentence,
            "clean_sentence": clean_sentence,
            "parsed_subject": noisy_probe["parsed_subject"],
            "parsed_predicate": noisy_probe["parsed_predicate"],
            "parsed_text": noisy_probe["parsed_text"],
            "intended_subject": intended_subject,
            "intended_predicate": intended_predicate,
            "intended_object": intended_object,
            "subject_match": noisy_probe["subject_match"],
            "predicate_match": noisy_probe["predicate_match"],
            "object_present": noisy_probe["object_present"],
            "rank": noisy_probe["rank"],
            "recall@1": noisy_probe["recall@1"],
            "recall@5": noisy_probe["recall@5"],
            "action": noisy_probe["action"],
            "clean_parsed_subject": clean_probe["parsed_subject"],
            "clean_parsed_predicate": clean_probe["parsed_predicate"],
            "clean_parsed_text": clean_probe["parsed_text"],
            "clean_subject_match": clean_probe["subject_match"],
            "clean_predicate_match": clean_probe["predicate_match"],
            "clean_object_present": clean_probe["object_present"],
            "clean_recall@1": clean_probe["recall@1"],
            "clean_recall@5": clean_probe["recall@5"],
            "subject_is_multi_token": len(observation["subject"].split()) > 1,
            "predicate_is_multi_token": len(
                observation["predicate"].split()) > 1,
            "query": case["noisy_query"],
        }
        records.append(record)

    return {
        "n": len(records),
        "subject_match_rate": _mean(
            [1.0 if r["subject_match"] else 0.0 for r in records]),
        "predicate_match_rate": _mean(
            [1.0 if r["predicate_match"] else 0.0 for r in records]),
        "object_present_rate": _mean(
            [1.0 if r["object_present"] else 0.0 for r in records]),
        "recall@1": _mean([r["recall@1"] for r in records]),
        "recall@5": _mean([r["recall@5"] for r in records]),
        "clean_subject_match_rate": _mean(
            [1.0 if r["clean_subject_match"] else 0.0 for r in records]),
        "clean_predicate_match_rate": _mean(
            [1.0 if r["clean_predicate_match"] else 0.0 for r in records]),
        "clean_object_present_rate": _mean(
            [1.0 if r["clean_object_present"] else 0.0 for r in records]),
        "clean_recall@1": _mean([r["clean_recall@1"] for r in records]),
        "clean_recall@5": _mean([r["clean_recall@5"] for r in records]),
        "multi_token_subject_rate": _mean(
            [1.0 if r["subject_is_multi_token"] else 0.0 for r in records]),
        "multi_token_predicate_rate": _mean(
            [1.0 if r["predicate_is_multi_token"] else 0.0 for r in records]),
        "records": records,
    }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def _git_commit() -> Optional[str]:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None


def run_audit(seed: int = DEFAULT_SEED,
              regenerate: bool = False,
              limit: Optional[int] = None,
              lifecycle_limit: Optional[int] = None,
              stub: bool = False,
              skip_qdrant: bool = False,
              include_hybrid: bool = True,
              raw_limit: int = 100,
              run_lifecycle_probe: bool = True,
              run_raw_probe: bool = True,
              safety: bool = False) -> Dict[str, Any]:
    started = time.perf_counter()
    cases, lifecycle, dataset_meta = load_datasets(
        seed=seed, regenerate=regenerate, limit=limit,
        lifecycle_limit=lifecycle_limit)
    components = build_components(stub=stub)
    _log(
        f"  components={components['label']} cases={len(cases)} "
        f"lifecycle={len(lifecycle)} safety={safety}")

    retrieval = run_retrieval(
        cases, components, skip_qdrant=skip_qdrant,
        include_hybrid=include_hybrid, eligibility=safety,
        strict_entity_identity=safety)

    lifecycle_result: Optional[Dict[str, Any]] = None
    if run_lifecycle_probe:
        lifecycle_result = run_lifecycle(
            lifecycle, components, strict_entity_identity=safety)

    raw_result: Optional[Dict[str, Any]] = None
    if run_raw_probe:
        raw_result = run_raw_ingestion(cases, components, limit=raw_limit)

    payload: Dict[str, Any] = {
        "audit": "natural_language_robustness",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": seed,
        "dataset": dataset_meta,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "git_commit": _git_commit(),
            "components": components["label"],
            "dense_model": components["dense_model"],
            "nli_model": components["nli_model"],
            "stub_mechanics_only": bool(stub),
        },
        "config": {
            "top_k": TOP_K,
            "methods": retrieval["methods"],
            "qdrant_candidate_limit": QDRANT_CANDIDATE_LIMIT,
            "retrieval_conditions": [
                "clean_ingest_clean_query",
                "clean_ingest_noisy_query",
                "noisy_ingest_clean_query",
                "noisy_ingest_noisy_query",
            ],
            "raw_probe_limit": raw_limit,
            "run_lifecycle_probe": run_lifecycle_probe,
            "safety_profile": {
                "enabled": bool(safety),
                "strict_entity_identity": bool(safety),
                "retrieval_eligibility": bool(safety),
                "min_similarity": None,
            },
        },
        "retrieval": retrieval,
        "lifecycle": lifecycle_result,
        "raw_ingestion": raw_result,
        "resources": {
            "rss_start_mb": rss_mb(),
            "rss_end_mb": rss_mb(),
            "wall_seconds": round(time.perf_counter() - started, 2),
            "cpu_seconds": round(time.process_time(), 2),
        },
    }

    records: List[Dict[str, Any]] = list(payload["retrieval"].pop(
        "records", []))
    records.extend(payload["retrieval"].pop("unsupported", []))
    if payload["lifecycle"] is not None:
        records.extend(payload["lifecycle"].pop("records", []))
    if payload["raw_ingestion"] is not None:
        records.extend(payload["raw_ingestion"].pop("records", []))
    payload["record_counts"] = Counter(
        record["record_type"] for record in records)
    return {"summary": payload, "records": records}


def write_json(path: str, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)


def write_jsonl(path: str, records: Sequence[Dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(target)
