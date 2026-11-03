"""Retrieval benchmark for the integrated Qdrant backend.

This is a new benchmark for the Qdrant integration.  It does not modify or
reinterpret the historical Phase 27/31 results; those remain the canonical
record of the earlier storage evaluations.

It measures four retrieval methods on the same canonical memory:

* ``minemos_dense``  - MINEMOS canonical dense retrieval (reference path)
* ``qdrant_dense``   - Qdrant dense candidates, exactly re-ranked by MINEMOS
* ``bm25``           - the existing lexical BM25 index only
* ``hybrid``         - Qdrant dense + BM25 candidates merged, then re-ranked
                       exactly by MINEMOS

Reported per method: Recall@1/3/5, MRR, nDCG@5, p50/p95 latency, CPU time.
Reported per memory size: ingestion time (with live Qdrant indexing), full
rebuild time, resident set size, and process CPU time.

Usage::

    .venv/Scripts/python.exe -m benchmarks.qdrant_retrieval_benchmark
    .venv/Scripts/python.exe -m benchmarks.qdrant_retrieval_benchmark \
        --sizes 100 300 1000 2000 --queries 40
    MINEMOS_QDRANT_URL=http://127.0.0.1:6333 \
        .venv/Scripts/python.exe -m benchmarks.qdrant_retrieval_benchmark

Without ``MINEMOS_QDRANT_URL`` the benchmark uses Qdrant's persistent embedded
mode (a real local database, not a mock).  Server mode is the production
transport and uses the server's HNSW index.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import random
import statistics
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from benchmarks.phase18_experiment import DenseSemanticEncoder
from benchmarks.phase27_dataset import RetrievalQuery
from benchmarks.phase27_metrics import compute_retrieval_metrics
from phase2.black_hole_memory import BlackHoleMemory
from phase2.retrieval import QdrantConfig
from phase2.structured_candidate import Candidate
from tests._retrieval_helpers import RuleVerifier, StubNLI

TOPIC_POOL = [
    "marine biology", "quantum optics", "volcanic geology", "protein folding",
    "urban hydrology", "stellar evolution", "coral ecology", "robot swarms",
    "synthetic chemistry", "glacial dynamics", "neural circuits",
    "materials fatigue", "atmospheric chemistry", "plant genetics",
    "seismic tomography", "game theory", "cryptographic protocols",
    "linguistic typology", "archaeological dating", "fluid turbulence",
    "catalyst design", "immune signalling", "deep sea ecology",
    "asteroid dynamics", "soil microbiology", "power grid stability",
    "acoustic engineering", "paleoclimate modelling", "drug metabolism",
    "swarm intelligence", "network topology", "renewable storage",
    "surface physics", "molecular motors", "sleep physiology",
    "speech perception", "combustion chemistry", "crop resilience",
    "ocean acidification", "quantum cryptography", "fossil anatomy",
    "biomechanical prosthetics", "geothermal systems", "exoplanet atmospheres",
    "microplastic transport", "gene editing", "ant colony behaviour",
    "thermal insulation", "radio astronomy", "river restoration",
    "wildfire modelling", "crystal growth", "vaccine adjuvants",
    "tectonic stress", "animal navigation", "enzyme kinetics",
    "desert hydrology", "computational geometry", "neuromorphic hardware",
    "plant signalling", "polar meteorology", "architectural acoustics",
    "fossil fuel chemistry", "cell membrane transport", "quantum sensing",
    "volcanic hazard", "forest succession", "protein interaction",
    "battery chemistry", "stellar magnetism", "glacier retreat",
    "pollinator ecology", "robot perception", "signal processing",
    "catalyst recycling", "immune memory", "deep ocean currents",
    "meteorite composition", "soil carbon", "grid resilience",
    "vibration control", "climate attribution", "drug delivery",
    "collective decision making", "network congestion", "hydrogen storage",
    "surface chemistry", "molecular recognition", "circadian rhythm",
    "speech synthesis", "flame propagation", "seed dormancy",
    "ocean circulation", "post-quantum security", "fossil footprints",
    "soft robotics", "geothermal exploration", "planetary formation",
    "microplastics", "crop genetics", "ant foraging", "building physics",
]

FACET_POOL = [
    "field observations", "laboratory replication", "long-term monitoring",
    "numerical simulation", "comparative analysis", "high-resolution imaging",
    "large-scale surveys", "controlled experiments", "archival reconstruction",
    "multi-site sampling", "statistical modelling", "instrument calibration",
    "cross-disciplinary synthesis", "open data curation", "bench-scale testing",
    "pilot deployments", "longitudinal tracking", "spatial mapping",
    "signal decomposition", "thermodynamic modelling", "molecular assays",
    "network analysis", "risk assessment", "scenario planning",
    "high-throughput screening", "remote sensing", "isotope analysis",
    "time-series forecasting", "mechanistic modelling", "systematic review",
    "field trials", "sensor fusion", "genomic sequencing", "flow visualisation",
    "material characterisation", "reaction pathway analysis",
    "behavioural experiments", "acoustic monitoring", "climate downscaling",
    "pharmacokinetic modelling", "evolutionary reconstruction",
    "structural analysis", "energy balance modelling", "water quality sampling",
    "quantum state tomography", "fossil preparation", "biomechanical testing",
    "thermal profiling", "radio interferometry", "habitat restoration",
    "fire spread simulation", "crystallography", "immunological assays",
    "stress-field mapping", "animal tracking", "kinetic profiling",
    "hydrological modelling", "computational topology", "hardware benchmarking",
    "plant hormone assays", "polar ice coring", "room acoustics",
    "fuel combustion analysis", "membrane transport assays", "quantum metrology",
    "hazard mapping", "forest inventory", "protein docking", "battery cycling",
    "magnetohydrodynamics", "mass balance analysis", "pollinator surveys",
    "perception benchmarking", "filter design", "catalyst recovery",
    "immune profiling", "current profiling", "meteorite petrology",
    "soil carbon accounting", "grid contingency analysis",
    "vibration testing", "extreme event attribution", "nanoparticle delivery",
    "consensus modelling", "congestion control", "hydrogen embrittlement",
    "surface spectroscopy", "molecular dynamics", "sleep staging",
    "voice conversion", "flame imaging", "seed germination",
    "ocean reanalysis", "lattice cryptography", "trace fossil analysis",
    "soft actuator design", "geothermal reservoir modelling",
    "planet formation modelling", "microplastic tracing", "crop breeding",
    "ant trail analysis", "building energy simulation",
]

FIRST_NAMES = [
    "Avery", "Rowan", "Seraphina", "Thaddeus", "Valerie", "Zephyr",
    "Cassian", "Orion", "Lyra", "Eldridge", "Morrigan", "Persephone",
    "Barnaby", "Corinna", "Leopold", "Astrid", "Theron", "Ignatius",
    "Solene", "Amara", "Benedict", "Clara", "Darius", "Elowen",
    "Fenwick", "Greta", "Hollis", "Iris", "Jasper", "Keira",
    "Lucian", "Maren", "Nadia", "Oscar", "Priya", "Quentin",
    "Rosalind", "Silas", "Tamsin", "Ulric", "Vera", "Wren",
]

LAST_NAMES = [
    "Ashford", "Bellamy", "Carrow", "Dunmore", "Ellery", "Fairbanks",
    "Galloway", "Harlow", "Ingram", "Jarvis", "Kingsley", "Lockhart",
    "Merrivale", "Northcott", "Ormsby", "Pemberton", "Quill", "Ravenscroft",
    "Sutherland", "Thorne", "Underhill", "Vance", "Whitlock", "Yarrow",
    "Zeller", "Ainsley", "Blackwood", "Caldwell", "Davenport", "Everhart",
]

QUERY_TEMPLATES = [
    "Which researcher studies {topic} with a focus on {facet}?",
    "Who investigates {topic} using {facet}?",
    "Find the scientist whose work on {topic} relies on {facet}.",
    "Which person researches {topic} through {facet}?",
]


def _rss_mb() -> Optional[float]:
    """Current process resident set size in MiB (Windows or POSIX)."""
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

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(usage / 1024.0, 2)
    except Exception:  # noqa: BLE001 - optional observability
        return None


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
    return round(ordered[index], 3)


def _make_subjects(n: int) -> List[str]:
    subjects = []
    for i in range(n):
        first = FIRST_NAMES[i % len(FIRST_NAMES)]
        last = LAST_NAMES[(i // len(FIRST_NAMES)) % len(LAST_NAMES)]
        suffix = "" if i < len(FIRST_NAMES) * len(LAST_NAMES) else f" {i}"
        subjects.append(f"{first} {last}{suffix}")
    return subjects


def build_corpus(n: int, seed: int) -> Dict[str, Any]:
    """Deterministic corpus with one relevant fact per query.

    Each target fact is accompanied by a lexically overlapping distractor that
    shares its topic and facet vocabulary.  This keeps the benchmark from
    saturating on trivial keyword matching and separates dense, lexical and
    hybrid behaviour.
    """
    rng = random.Random(seed)
    subjects = _make_subjects(2 * n)
    facts = []
    for i, subject in enumerate(subjects[:n]):
        # Pair topics and facets so every (topic, facet) query identifies
        # exactly one target fact; cycling both pools with the same period
        # would make large-N queries ambiguous by construction.
        topic = TOPIC_POOL[i % len(TOPIC_POOL)]
        facet = FACET_POOL[(i // len(TOPIC_POOL)) % len(FACET_POOL)]
        obj = f"{topic} through {facet}"
        text = f"{subject} studies {topic} through {facet}."
        facts.append({
            "subject": subject,
            "predicate": "studies",
            "object": obj,
            "text": text,
            "topic": topic,
            "facet": facet,
            "timestamp": float(i + 1),
        })
        distractor = subjects[n + i]
        facts.append({
            "subject": distractor,
            "predicate": "reviewed",
            "object": f"earlier literature about {topic} and {facet}",
            "text": (f"{distractor} reviewed earlier literature about "
                     f"{topic} and {facet}."),
            "topic": topic,
            "facet": facet,
            "timestamp": float(1_000_000 + i),
        })

    query_count = min(40, n)
    chosen = rng.sample(range(n), query_count)
    queries = []
    for q_id, fact_idx in enumerate(chosen):
        fact = facts[2 * fact_idx]
        template = QUERY_TEMPLATES[q_id % len(QUERY_TEMPLATES)]
        queries.append(RetrievalQuery(
            query_id=f"q{q_id:03d}",
            query_text=template.format(
                topic=fact["topic"], facet=fact["facet"]),
            target_subject=fact["subject"],
            target_predicate=fact["predicate"],
            expected_answer=fact["object"],
            expected_fact_text=fact["text"],
            condition="qdrant_integration",
        ))
    return {"facts": facts, "target_facts": n, "queries": queries}


def _bm25_rows(mem: BlackHoleMemory, query_text: str,
               top_k: int) -> List[Dict[str, Any]]:
    with mem._lock:
        slots = list(mem.memory.slots)
        id_to_slot_index = {id(slot): idx for idx, slot in enumerate(slots)}
        hits = mem.bm25_index.query(
            query_text=query_text, top_k=top_k,
            id_to_slot_index=id_to_slot_index)
        rows = []
        for idx, score in hits:
            if not (0 <= idx < len(slots)):
                continue
            slot = slots[idx]
            rows.append({
                "slot_index": idx,
                "slot_id": id(slot),
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
        return rows


def run_method(mem: BlackHoleMemory, queries: List[RetrievalQuery],
               method: str, top_k: int) -> Dict[str, Any]:
    results: List[List[Dict[str, Any]]] = []
    latencies: List[float] = []
    cpu_start = time.process_time()
    for query in queries:
        started = time.perf_counter()
        if method == "bm25":
            rows = _bm25_rows(mem, query.query_text, top_k)
        elif method == "hybrid":
            rows = mem.query(query.query_text, top_k=top_k, hybrid=True)
        else:
            rows = mem.query(query.query_text, top_k=top_k)
        latencies.append((time.perf_counter() - started) * 1000.0)
        results.append(rows)
    cpu_ms = (time.process_time() - cpu_start) * 1000.0
    metrics = compute_retrieval_metrics(queries, results, latencies)
    metrics.update({
        "p50_ms": _percentile(latencies, 0.50),
        "p95_ms": _percentile(latencies, 0.95),
        "cpu_ms_total": round(cpu_ms, 2),
        "cpu_ms_per_query": round(cpu_ms / max(1, len(queries)), 3),
    })
    return metrics


def _qdrant_config(collection: str, path: Optional[str]) -> QdrantConfig:
    url = os.environ.get("MINEMOS_QDRANT_URL")
    if url:
        return QdrantConfig(
            url=url,
            api_key=os.environ.get("MINEMOS_QDRANT_API_KEY"),
            collection=collection,
            tenant_id="bench",
            timeout=30.0,
            candidate_limit=128,
        )
    return QdrantConfig(
        path=path or os.path.join(tempfile.mkdtemp(), "qdrant"),
        collection=collection,
        tenant_id="bench",
        candidate_limit=128,
    )


def benchmark_size(n: int, query_count: int, seed: int, top_k: int,
                   encoder: DenseSemanticEncoder,
                   qdrant_path: Optional[str]) -> Dict[str, Any]:
    corpus = build_corpus(n, seed)
    queries = corpus["queries"][:query_count]
    collection = f"minemos_bench_{uuid.uuid4().hex[:10]}"
    config = _qdrant_config(collection, qdrant_path)
    mode = "server" if config.url else "local"

    rss_start = _rss_mb()
    cpu_ingest_start = time.process_time()
    wall_ingest_start = time.perf_counter()
    mem = BlackHoleMemory(
        capacity=2 * n + 10,
        dense_encoder=encoder,
        nli_gate=StubNLI(),
        verifier=RuleVerifier(),
        retrieval_backend="qdrant",
        qdrant_config=config,
    )
    try:
        for fact in corpus["facts"]:
            mem.absorb(Candidate(
                fact["subject"], fact["predicate"], fact["object"],
                timestamp=fact["timestamp"]))
        ingest_ms = (time.perf_counter() - wall_ingest_start) * 1000.0
        ingest_cpu_ms = (time.process_time() - cpu_ingest_start) * 1000.0
        rss_after_ingest = _rss_mb()

        retriever = mem._get_retrieval_adapter().ensure_retriever()
        rebuild_start = time.perf_counter()
        rebuild = retriever.rebuild()
        rebuild_ms = (time.perf_counter() - rebuild_start) * 1000.0

        methods: Dict[str, Any] = {}
        for method in ("qdrant_dense", "hybrid"):
            methods[method] = run_method(mem, queries, method, top_k)

        qdrant_stats = mem.retrieval_stats().get("manager")
        mem.set_retrieval_backend("canonical")
        methods["minemos_dense"] = run_method(
            mem, queries, "minemos_dense", top_k)
        methods["bm25"] = run_method(mem, queries, "bm25", top_k)

        return {
            "facts": 2 * n,
            "target_facts": corpus["target_facts"],
            "queries": len(queries),
            "final_slots": len(mem.memory.slots),
            "indexed_points": rebuild["points"],
            "qdrant_mode": mode,
            "ingest_ms": round(ingest_ms, 1),
            "ingest_cpu_ms": round(ingest_cpu_ms, 1),
            "rebuild_ms": round(rebuild_ms, 1),
            "rss_start_mb": rss_start,
            "rss_after_ingest_mb": rss_after_ingest,
            "rss_end_mb": _rss_mb(),
            "methods": methods,
            "qdrant_stats": qdrant_stats,
            "qdrant_last_error": (
                None if not qdrant_stats else qdrant_stats.get("last_error")),
        }
    finally:
        mem.close()


def _write_payload(payload: Dict[str, Any], output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[100, 300, 1000, 2000])
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=os.path.join(
        "benchmarks", "results", "qdrant_integration_benchmark.json"))
    parser.add_argument("--keep-db", action="store_true",
                        help="keep the embedded Qdrant directory")
    args = parser.parse_args(argv)

    encoder = DenseSemanticEncoder()
    qdrant_path = None if os.environ.get("MINEMOS_QDRANT_URL") else (
        tempfile.mkdtemp(prefix="minemos_bench_qdrant_"))

    payload: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "mode": "server" if os.environ.get("MINEMOS_QDRANT_URL")
                    else "local-embedded",
            "qdrant_url": os.environ.get("MINEMOS_QDRANT_URL"),
        },
        "config": {
            "sizes": args.sizes,
            "queries": args.queries,
            "seed": args.seed,
            "top_k": args.top_k,
            "encoder": "sentence-transformers/all-MiniLM-L6-v2",
            "candidate_limit": 128,
        },
        "results": {},
        "notes": [
            "New benchmark for the Qdrant integration; historical Phase 27 "
            "and Phase 31 results are unchanged and remain separately "
            "authoritative.",
            "Local-embedded Qdrant uses brute-force search, not the server's "
            "HNSW index; run with MINEMOS_QDRANT_URL for production transport.",
            "Each method answers the same queries against the same canonical "
            "memory; Qdrant and hybrid results are exactly re-ranked by "
            "MINEMOS.",
        ],
    }

    try:
        for n in args.sizes:
            print(f"[benchmark] N={n}", flush=True)
            result = benchmark_size(
                n, args.queries, args.seed, args.top_k, encoder, qdrant_path)
            payload["results"][str(n)] = result
            # Persist completed sizes immediately so a long later size cannot
            # discard measurements that already finished.
            _write_payload(payload, args.output)
            for method, metrics in result["methods"].items():
                print(
                    f"  {method:14s} R@1={metrics['recall_at_1']:.3f} "
                    f"R@5={metrics['recall_at_5']:.3f} "
                    f"MRR={metrics['mrr']:.3f} "
                    f"nDCG@5={metrics['ndcg_at_5']:.3f} "
                    f"p50={metrics['p50_ms']:.2f}ms "
                    f"p95={metrics['p95_ms']:.2f}ms", flush=True)
    finally:
        if qdrant_path and not args.keep_db:
            import shutil

            shutil.rmtree(qdrant_path, ignore_errors=True)

    output = Path(args.output)
    _write_payload(payload, args.output)
    print(f"[benchmark] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())