"""Tests for the dedicated real-world evaluation pipeline.

The suite verifies parsing with source boundaries and timestamps, reproducible
persistent ingestion, independent ground truth, versioned metrics, privacy
tombstones, restart persistence, and the CLI/store refusal paths.  It uses
deterministic stub encoders; it does not claim a real 30-day run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.formats import (
    Observation,
    Segment,
    SourceFormatError,
    observation_segments,
    parse_source_file,
    split_text,
    verbatim_observations,
)
from evaluation.run import (
    QueryFormatError,
    _read_queries,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    run,
)
from evaluation.store import (
    EvaluationEnvironment,
    EvaluationStore,
    EvaluationStoreError,
)
from tests._service_helpers import make_factory, make_models
from tests._retrieval_helpers import RuleVerifier, make_memory

TENANT = "evaltenant"


# ── fixtures ───────────────────────────────────────────────────────────────


def conflict_factory(config, tenant_id, models):
    verifier = RuleVerifier([
        ("Berlin", "Munich", "CONTRADICTION"),
        ("Munich", "Berlin", "CONTRADICTION"),
    ])
    return make_memory(capacity=config.engine_capacity,
                       verifier=verifier, encoder=models.dense_encoder)


def open_env(root: Path, factory=None) -> EvaluationEnvironment:
    return EvaluationEnvironment.open(
        root, tenant=TENANT,
        engine_factory=factory or make_factory(),
        models=make_models())


def segment(index: int, text: str, *, source_id: str = "doc-1",
            document_id: str = "doc-1", timestamp=1_700_000_000.0,
            observations=()) -> Segment:
    return Segment(
        source_id=source_id,
        source_type="document",
        document_id=document_id,
        segment_index=index,
        text=text,
        original_timestamp=timestamp,
        observations=tuple(observations),
    )


@pytest.fixture()
def store_root(tmp_path):
    root = tmp_path / "evaluation"
    EvaluationStore.init(root, tenant=TENANT, description="test store")
    return root


# ── format parsing ─────────────────────────────────────────────────────────


def test_conversation_parsing_preserves_boundaries_and_timestamps(tmp_path):
    path = tmp_path / "conversation.jsonl"
    path.write_text("\n".join([
        json.dumps({
            "source_id": "conv-1", "conversation_id": "conv-1",
            "timestamp": 1.0,
            "messages": [
                {"role": "user", "content": "I live in Berlin."},
                {"role": "assistant", "content": "Noted."},
            ]}),
        json.dumps({
            "source_id": "conv-1", "conversation_id": "conv-1",
            "messages": [{"role": "user", "content": "I moved to Munich.",
                          "timestamp": 2.0}]}),
    ]), encoding="utf-8")
    segments = parse_source_file(path, "conversation")
    assert [s.segment_index for s in segments] == [0, 1, 2]
    assert all(s.conversation_id == "conv-1" for s in segments)
    assert segments[0].speaker == "user"
    assert segments[1].speaker == "assistant"
    assert segments[0].original_timestamp == 1.0
    assert segments[2].original_timestamp == 2.0
    assert segments[0].text == "I live in Berlin."
    assert segments[0].segment_key == "conv-1:conv-1:0"


def test_document_parsing_keeps_document_identity_and_offsets(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text(json.dumps({
        "source_id": "docs-1", "document_id": "design-1",
        "title": "Design",
        "timestamp": 10.0,
        "text": "First paragraph here.\n\nSecond paragraph follows.",
    }) + "\n", encoding="utf-8")
    segments = parse_source_file(path, "document")
    assert [s.document_id for s in segments] == ["design-1", "design-1"]
    assert [s.text for s in segments] == [
        "First paragraph here.", "Second paragraph follows."]
    assert segments[0].char_start < segments[1].char_start


def test_prose_split_offsets_round_trip(tmp_path):
    text = "Alpha paragraph.\n\nBeta paragraph with more words."
    path = tmp_path / "notes.md"
    path.write_text(text, encoding="utf-8")
    segments = parse_source_file(path, "notes", source_id="notes-1")
    assert len(segments) == 2
    for item in segments:
        assert text[item.char_start:item.char_end] == item.text


def test_untimed_sources_are_marked_inferred(tmp_path):
    path = tmp_path / "journal.md"
    path.write_text("Today was quiet.", encoding="utf-8")
    segments = parse_source_file(path, "journal", source_id="journal-1")
    assert segments[0].timestamp_inferred is True
    assert segments[0].original_timestamp is None


def test_observation_mapping_must_cover_every_segment():
    segments = [segment(0, "Alice lives in Berlin."),
                segment(1, "Alice works at Acme.")]
    mapping = {"doc-1:doc-1:0": [
        Observation("alice", "city", "Berlin", timestamp=1.0)]}
    with pytest.raises(SourceFormatError, match="does not cover"):
        observation_segments(segments, mapping)
    full = dict(mapping)
    full["doc-1:doc-1:1"] = [
        Observation("alice", "employer", "Acme", timestamp=2.0)]
    attached = observation_segments(segments, full)
    assert [len(item.observations) for item in attached] == [1, 1]


def test_verbatim_observations_do_not_invent_facts():
    item = segment(0, "I live in Berlin.", timestamp=5.0)
    observation = verbatim_observations([item])[0]
    assert observation.subject == "document:doc-1"
    assert observation.predicate == "segment_0"
    assert observation.object == "I live in Berlin."
    assert observation.timestamp == 5.0


def test_invalid_source_is_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"messages": []}), encoding="utf-8")
    with pytest.raises(SourceFormatError):
        parse_source_file(path, "nonsense")
    with pytest.raises(SourceFormatError):
        parse_source_file(path, "conversation")


def test_empty_text_has_no_segments():
    assert split_text("   \n\n  ") == []


# ── store invariants ───────────────────────────────────────────────────────


def test_store_init_refuses_non_empty_uninitialized_directory(tmp_path):
    root = tmp_path / "not-a-store"
    root.mkdir()
    (root / "file.txt").write_text("data", encoding="utf-8")
    with pytest.raises(EvaluationStoreError, match="not empty"):
        EvaluationStore.init(root)


def test_manifest_declares_canonical_authority(store_root):
    store = EvaluationStore(store_root, tenant=TENANT)
    manifest = store.manifest
    assert manifest["format_version"] == 1
    assert manifest["authoritative_state"] == "checkpoints/"
    assert "never authoritative" in manifest["qdrant_role"]


def test_ingestion_records_forbid_raw_content(store_root):
    store = EvaluationStore(store_root, tenant=TENANT)
    with pytest.raises(EvaluationStoreError, match="must not contain content"):
        store.record_ingestion({"ingestion_id": "i", "text": "secret"})


def test_result_versions_are_immutable(store_root):
    store = EvaluationStore(store_root, tenant=TENANT)
    first = store.new_run_dir("run-1")
    assert first.is_dir()
    with pytest.raises(EvaluationStoreError, match="immutable"):
        store.new_run_dir("run-1")


# ── environment ingestion ──────────────────────────────────────────────────


def test_verbatim_ingestion_is_reproducible_and_persistent(store_root):
    segments = [
        segment(0, "I live in Berlin.", timestamp=1_700_000_000.0),
        segment(1, "I moved to Munich.", timestamp=1_700_100_000.0),
    ]
    environment = open_env(store_root)
    try:
        dry = environment.ingest_segments(segments, dry_run=True)
        assert dry["observations"] == 2
        assert environment.store.segments("doc-1") == []
        first = environment.ingest_segments(segments)
        assert first["ingested"] == 2
        again = environment.ingest_segments(segments)
        assert again["new_segments"] == 0
        assert len(environment.store.segments("doc-1")) == 2
        assert len(environment.pool.history(
            TENANT, "document:doc-1")) == 2
        assert environment.store.verify()["ok"] is True
    finally:
        environment.close()

    reopened = open_env(store_root)
    try:
        again = reopened.ingest_segments(segments)
        assert again["new_segments"] == 0
        assert len(reopened.pool.history(TENANT, "document:doc-1")) == 2
        assert reopened.store.verify()["ok"] is True
    finally:
        reopened.close()


def test_ingest_requires_one_source_per_call(store_root):
    environment = open_env(store_root)
    try:
        with pytest.raises(EvaluationStoreError, match="exactly one source_id"):
            environment.ingest_segments([
                segment(0, "a", source_id="doc-1", document_id="doc-1"),
                segment(0, "b", source_id="doc-2", document_id="doc-2"),
            ])
    finally:
        environment.close()


def test_credentials_in_content_and_identifiers(store_root):
    environment = open_env(store_root)
    secret = "AKIA" + "A" * 16
    try:
        with pytest.raises(EvaluationStoreError, match="credentials"):
            environment.ingest_segments(
                [segment(0, f"key {secret}")], credential_policy="reject")
        with pytest.raises(EvaluationStoreError, match="identifiers"):
            environment.ingest_segments(
                [segment(0, "text", source_id=secret, document_id=secret)],
                credential_policy="redact")
        result = environment.ingest_segments(
            [segment(0, f"key {secret}")], credential_policy="redact")
        assert result["ingested"] == 1
        stored = environment.store.segments("doc-1")[0]["text"]
        assert secret not in stored and "REDACTED" in stored
    finally:
        environment.close()


def test_duplicate_fact_is_recorded_without_new_memory(store_root):
    first = segment(0, "Alice works at Acme.", source_id="doc-1",
                    document_id="doc-1",
                    observations=(Observation("alice", "employer", "Acme",
                                              timestamp=1.0),))
    second = segment(0, "Alice works at Acme.", source_id="doc-2",
                     document_id="doc-2",
                     observations=(Observation("alice", "employer", "Acme",
                                               timestamp=2.0),))
    environment = open_env(store_root)
    try:
        one = environment.ingest_segments([first], mode="structured")
        two = environment.ingest_segments([second], mode="structured")
        assert one["ingested"] == 1
        assert two["ingested"] == 1
        ingestions = [
            json.loads(line) for line in
            environment.store.ingestions_path.read_text(
                encoding="utf-8").splitlines() if line.strip()]
        assert ingestions[1]["duplicate_of"] == ingestions[0]["memory_id"]
    finally:
        environment.close()


# ── deletion, redaction and source tombstones ─────────────────────────────


def test_delete_memory_purges_source_text_and_prevents_revival(store_root):
    segments = [segment(0, "content one"), segment(1, "content two")]
    environment = open_env(store_root)
    try:
        environment.ingest_segments(segments)
        rows = environment.pool.query(TENANT, "content one", top_k=5)
        target = next(row for row in rows
                      if "content one" in str(row.get("text")))
        memory_id = target["memory_id"]
        result = environment.delete_memory(memory_id)
        assert result["changed"] is True
        assert result["source_segments_removed"] == 1
        remaining = environment.store.segments("doc-1")
        assert all(item["text"] != "content one" for item in remaining)
        again = environment.ingest_segments(segments)
        assert again["new_segments"] == 0
        assert all(item["text"] != "content one"
                   for item in environment.store.segments("doc-1"))
        assert environment.store.verify()["ok"] is True
    finally:
        environment.close()


def test_source_deletion_removes_text_and_refuses_reingest(store_root):
    segments = [segment(0, "private note"), segment(1, "another note")]
    environment = open_env(store_root)
    try:
        environment.ingest_segments(segments)
        result = environment.delete_source("doc-1")
        assert result["changed_memories"] == 2
        assert environment.store.segments("doc-1") == []
        assert environment.store.sources()["doc-1"]["active"] is False
        with pytest.raises(EvaluationStoreError, match="previously delete"):
            environment.ingest_segments(segments)
        assert environment.store.verify()["ok"] is True
    finally:
        environment.close()


def test_source_redaction_is_tombstoned(store_root):
    environment = open_env(store_root)
    try:
        environment.ingest_segments([segment(0, "sensitive detail")])
        result = environment.redact_source("doc-1")
        assert result["changed_memories"] == 1
        assert environment.store.segments("doc-1") == []
        assert environment.store.source_tombstoned("doc-1") == "redact"
    finally:
        environment.close()


# ── metrics and versioned results ──────────────────────────────────────────


def _write_queries(path: Path, entries) -> None:
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n",
                    encoding="utf-8")


def test_metric_helpers_match_known_values():
    assert recall_at_k([1, 2, 0], 2) == 2 / 3
    assert reciprocal_rank([0, 4]) == 0.25
    assert reciprocal_rank([]) == 0.0
    assert ndcg_at_k([], 5) == 1.0
    assert 0.0 < ndcg_at_k([1], 5) <= 1.0


def test_query_validation_rejects_malformed_files(tmp_path):
    path = tmp_path / "q.jsonl"
    cases = [
        [{"query_id": "a", "type": "retrieval", "query": "x"}],
        [{"query_id": "a", "type": "nonsense", "query": "x"}],
        [{"query_id": "a", "type": "temporal", "query": "x"}],
        [{"query_id": "a", "type": "duplicate",
          "source_id": "s", "expected_count": True}],
        [{"query_id": "a", "type": "context", "query": "x",
          "expected_sequence": []}],
        [{"query_id": "a", "type": "retrieval", "query": "x",
          "top_k": True}],
    ]
    for entries in cases:
        _write_queries(path, entries)
        with pytest.raises(QueryFormatError):
            _read_queries(path)
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(QueryFormatError):
        _read_queries(path)


def test_ground_truth_is_separate_and_never_contaminates_memory(store_root):
    sentinel = "GROUND-TRUTH-SENTINEL-90317"
    segments = [
        segment(0, "Alice lives in Berlin.", timestamp=1_700_000_000.0,
                observations=(Observation("alice", "city", "Berlin",
                                          timestamp=1_700_000_000.0),)),
        segment(1, "Alice moved to Munich.", timestamp=1_700_100_000.0,
                observations=(Observation("alice", "city", "Munich",
                                          timestamp=1_700_100_000.0),)),
    ]
    queries_path = store_root / "ground_truth" / "queries.jsonl"
    _write_queries(queries_path, [
        {"query_id": "rt", "type": "retrieval", "query": "Munich",
         "top_k": 5,
         "expected": [{"subject": "alice", "predicate": "city",
                       "object": "Munich"}]},
        {"query_id": "tm", "type": "temporal", "query": "Berlin",
         "as_of": 1_700_050_000.0, "top_k": 5,
         "expected": [{"subject": "alice", "predicate": "city",
                       "object": "Berlin"}],
         "expected_inactive": [
             {"subject": "alice", "predicate": "city",
              "object": "Munich"}]},
        {"query_id": "ct", "type": "contradiction", "query": "city",
         "top_k": 5,
         "expected": [{"subject": "alice", "predicate": "city",
                       "object": "Munich"}],
         "expected_inactive": [
             {"subject": "alice", "predicate": "city",
              "object": "Berlin"}]},
        {"query_id": "hs", "type": "history", "subject": "alice",
         "expected_sequence": ["Berlin", "Munich"]},
        {"query_id": "dup", "type": "duplicate", "source_id": "doc-1",
         "expected_count": 0},
        {"query_id": "note", "type": "temporal", "query": "sentinel",
         "as_of": 1.0,
         "expected_inactive": [
             {"subject": "x", "predicate": "y", "object": sentinel}]},
    ])
    environment = open_env(store_root, factory=conflict_factory)
    try:
        environment.ingest_segments(segments, mode="structured")
        queries, raw = _read_queries(queries_path)
        summary = run(environment, queries, queries_path, raw)
        assert summary["query_count"] == 6
        assert summary["passed"] == 6, summary["failure_records"]
        assert summary["by_type"]["history"]["pass_rate"] == 1.0
        run_dir = Path(summary["result_dir"])
        assert (run_dir / "summary.json").is_file()
        assert (run_dir / "metadata.json").is_file()
        assert (run_dir / "per_query.jsonl").is_file()
        assert (run_dir / "failures.jsonl").is_file()
        assert summary["latency_ms"]["p50"] >= 0
        assert summary["resource"]["rss_after_bytes"] is not None

        checkpoint = (
            store_root / "checkpoints" / TENANT / "state.npz").read_bytes()
        ledger = (
            store_root / "provenance" / f"{TENANT}.jsonl").read_bytes()
        ingestions = store_root.joinpath("ingestions.jsonl").read_bytes()
        for blob in (checkpoint, ledger, ingestions):
            assert sentinel.encode() not in blob
            assert b"expected_inactive" not in blob
    finally:
        environment.close()


def test_failures_are_recorded_and_runs_are_versioned(store_root):
    environment = open_env(store_root)
    queries_path = store_root / "ground_truth" / "queries.jsonl"
    _write_queries(queries_path, [
        {"query_id": "ok", "type": "duplicate", "source_id": "none",
         "expected_count": 0},
        {"query_id": "missing", "type": "retrieval", "query": "anything",
         "expected": [{"subject": "nobody", "predicate": "nothing",
                       "object": "absent"}]},
    ])
    try:
        queries, raw = _read_queries(queries_path)
        first = run(environment, queries, queries_path, raw)
        assert first["passed"] == 1 and first["failed"] == 1
        failures = (Path(first["result_dir"]) /
                    "failures.jsonl").read_text(encoding="utf-8")
        assert "missing" in failures
        second = run(environment, queries, queries_path, raw)
        assert second["run_id"] != first["run_id"]
        assert second["result_dir"] != first["result_dir"]
    finally:
        environment.close()


def test_context_query_records_coverage(store_root):
    environment = open_env(store_root)
    try:
        environment.ingest_segments(
            [segment(0, "Alpha detail about the plan.")])
        queries_path = store_root / "ground_truth" / "context.jsonl"
        _write_queries(queries_path, [{
            "query_id": "ctx", "type": "context",
            "query": "Alpha detail about the plan",
            "expected_sequence": ["Alpha detail about the plan."],
            "top_k": 5,
        }])
        queries, raw = _read_queries(queries_path)
        summary = run(environment, queries, queries_path, raw)
        assert summary["passed"] == 1, summary["failure_records"]
    finally:
        environment.close()


def test_cli_init_and_dry_run_without_touching_engine(tmp_path, capsys):
    from evaluation import ingest as ingest_cli

    root = tmp_path / "cli-eval"
    assert ingest_cli.main(
        ["init", "--root", str(root), "--tenant", TENANT]) == 0
    capsys.readouterr()
    source = tmp_path / "notes.md"
    source.write_text("First note.\n\nSecond note.", encoding="utf-8")
    assert ingest_cli.main([
        "ingest", "--root", str(root), "--source", str(source),
        "--source-type", "notes", "--source-id", "cli-notes",
        "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["segments"] == 2
    assert payload["timestamp_inferred_segments"] == 2
    assert not (root / "checkpoints" / TENANT / "state.npz").exists()
    assert ingest_cli.main([
        "list-sources", "--root", str(root), "--tenant", TENANT]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["sources"] == {}


def test_cli_structured_dry_run_requires_observations(tmp_path, capsys):
    from evaluation import ingest as ingest_cli

    root = tmp_path / "cli-eval-2"
    ingest_cli.main(["init", "--root", str(root), "--tenant", TENANT])
    capsys.readouterr()
    source = tmp_path / "obs.jsonl"
    source.write_text(json.dumps({
        "source_id": "obs-1", "subject": "alice", "predicate": "city",
        "object": "Berlin", "timestamp": 5.0}) + "\n", encoding="utf-8")
    assert ingest_cli.main([
        "ingest", "--root", str(root), "--source", str(source),
        "--source-type", "observation", "--mode", "structured",
        "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["observations"] == 1
    assert payload["timestamped_segments"] == 1

    prose = tmp_path / "prose.md"
    prose.write_text("A sentence.", encoding="utf-8")
    assert ingest_cli.main([
        "ingest", "--root", str(root), "--source", str(prose),
        "--source-type", "notes", "--mode", "structured",
        "--dry-run"]) == 2
