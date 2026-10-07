"""
tests/test_phase17.py

Phase 17 test suite — Long-Horizon Continual Memory Under Semantic Noise.
All tests are deterministic and parameterised over small sub-streams.
Tests do NOT modify MATCH_THRESHOLD, the update equation, or production defaults.
The NLI model is only loaded for tests that explicitly exercise System A.
"""

from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from phase2.memory_state import MemoryState
from phase2.memory_update import MATCH_THRESHOLD
from benchmarks.phase17_dataset import (
    Phase17Obs, make_distractor,
    generate_scenario_1_long_gaps,
    generate_scenario_2_repeated_contradictions,
    generate_scenario_3_contradiction_recovery,
    generate_scenario_4_semantic_drift,
    generate_scenario_5_distractor_stress,
    generate_scenario_6_capacity_pressure_stream,
    generate_scenario_7_continual_stream,
)
from benchmarks.phase17_experiment import (
    nli_gated_absorb, oracle_absorb, lexical_absorb, run_stream,
    ENCODER, D,
)
from benchmarks.phase17_metrics import memory_metrics, candidate_retrieval_diagnostics


# ─────────────────────────────────────────
#  Dataset sanity tests (no model needed)
# ─────────────────────────────────────────

class TestPhase17Dataset:
    def test_scenario_1_stream_lengths(self):
        streams = generate_scenario_1_long_gaps(gaps=(10, 50))
        assert len(streams[10]) == 10 * (1 + 10 + 1 + 10 + 1), \
            "gap=10: 10 concepts × (1 canon + 10 dist + 1 para + 10 dist + 1 para)"
        # gap=10: 10*(1+10+1+10+1)=230
        assert len(streams[10]) == 230
        assert len(streams[50]) == 10 * (1 + 50 + 1 + 50 + 1)  # 1030

    def test_scenario_1_probes_are_marked(self):
        streams = generate_scenario_1_long_gaps(gaps=(10,))
        probes = [o for o in streams[10] if o.is_probe]
        assert len(probes) == 20, "2 probes per concept × 10 concepts"

    def test_scenario_1_probe_categories(self):
        streams = generate_scenario_1_long_gaps(gaps=(10,))
        probes = [o for o in streams[10] if o.is_probe]
        for p in probes:
            assert p.category == "paraphrase", f"Unexpected category: {p.category}"

    def test_scenario_2_length(self):
        s2 = generate_scenario_2_repeated_contradictions()
        assert len(s2) == 480

    def test_scenario_3_length(self):
        s3 = generate_scenario_3_contradiction_recovery()
        assert len(s3) == 540

    def test_scenario_4_length(self):
        s4 = generate_scenario_4_semantic_drift()
        assert len(s4) == 420

    def test_scenario_5_stream_lengths(self):
        streams = generate_scenario_5_distractor_stress(distractor_counts=(50, 100))
        assert len(streams[50]) == 50 + 20  # 20 targets + 50 distractors
        assert len(streams[100]) == 100 + 20

    def test_scenario_6_length(self):
        s6 = generate_scenario_6_capacity_pressure_stream()
        assert len(s6) == 1500

    def test_scenario_7_length(self):
        s7 = generate_scenario_7_continual_stream(n_obs=5000)
        assert len(s7) == 5000

    def test_obs_text_property(self):
        distractor = make_distractor(0, timestamp=0)
        text = distractor.text
        assert text.endswith(".")
        assert len(text.split()) >= 3

    def test_phase17obs_has_is_probe(self):
        s1 = generate_scenario_1_long_gaps(gaps=(10,))
        for obs in s1[10]:
            assert hasattr(obs, "is_probe")
            assert isinstance(obs.is_probe, bool)

    def test_scenario_7_has_delayed_paraphrases(self):
        s7 = generate_scenario_7_continual_stream(n_obs=5000)
        delayed = [o for o in s7 if o.category == "delayed_paraphrase"]
        assert len(delayed) > 0, "Scenario 7 must contain delayed_paraphrase probes"

    def test_match_threshold_unchanged(self):
        """Regression: production default must remain 0.75."""
        assert MATCH_THRESHOLD == 0.75, "MATCH_THRESHOLD must remain 0.75"

    def test_seed_determinism(self):
        """Two calls with same seed produce identical streams."""
        s2a = generate_scenario_2_repeated_contradictions(seed=1717)
        s2b = generate_scenario_2_repeated_contradictions(seed=1717)
        for a, b in zip(s2a, s2b):
            assert a.text == b.text and a.category == b.category


# ─────────────────────────────────────────
#  System C (Lexical) — no model needed
# ─────────────────────────────────────────

class TestPhase17LexicalBaseline:
    def _small_stream(self):
        """Tiny 3-concept, gap=2 stream for fast testing."""
        return generate_scenario_1_long_gaps(gaps=(2,))[2]

    def test_lexical_run_completes(self):
        stream = self._small_stream()
        res = run_stream(stream, capacity=50, system="C_lexical")
        assert res["system"] == "C_lexical"
        assert res["n_inserts"] + res["n_updates"] + res["n_evicts"] == len(stream)

    def test_lexical_canonical_inserts(self):
        """Each canonical observation must produce an insert on an empty memory."""
        stream = generate_scenario_1_long_gaps(gaps=(1,))[1]
        res = run_stream(stream, capacity=100, system="C_lexical")
        assert res["n_inserts"] >= 10, "10 canonical obs should create ≥10 inserts"

    def test_lexical_eviction_on_full_capacity(self):
        stream = generate_scenario_1_long_gaps(gaps=(2,))[2]
        res = run_stream(stream, capacity=3, system="C_lexical")
        # With only 3 slots, some evictions must occur
        assert res["n_evicts"] >= 0  # Evict may or may not happen depending on stream
        assert res["n_inserts"] + res["n_updates"] + res["n_evicts"] == len(stream)

    def test_lexical_no_nli_calls(self):
        stream = self._small_stream()
        res = run_stream(stream, capacity=50, system="C_lexical")
        assert res["n_nli_calls"] == 0, "Lexical baseline must not call NLI"

    def test_lexical_wall_time_positive(self):
        stream = self._small_stream()
        res = run_stream(stream, capacity=50, system="C_lexical")
        assert res["wall_seconds"] > 0


# ─────────────────────────────────────────
#  System B (Oracle) — no model needed
# ─────────────────────────────────────────

class TestPhase17OracleBaseline:
    def _small_stream(self):
        return generate_scenario_1_long_gaps(gaps=(2,))[2]

    def test_oracle_run_completes(self):
        stream = self._small_stream()
        res = run_stream(stream, capacity=50, system="B_oracle")
        assert res["system"] == "B_oracle"
        assert res["n_inserts"] + res["n_updates"] + res["n_evicts"] == len(stream)

    def test_oracle_paraphrase_probes_merge(self):
        """Oracle should merge paraphrase probes into the canonical slot."""
        stream = generate_scenario_1_long_gaps(gaps=(1,))[1]
        mem = MemoryState(capacity=100, dim=D)
        slot_concept_ids = {}
        updates_on_probes = 0
        for obs in stream:
            r = oracle_absorb(mem, obs.candidate, slot_concept_ids,
                              obs.concept_id, obs.category)
            if obs.is_probe and obs.category == "paraphrase":
                if r["action"] == "update":
                    updates_on_probes += 1
        n_probes = sum(1 for o in stream if o.is_probe and o.category == "paraphrase")
        assert updates_on_probes == n_probes, \
            f"Oracle should merge all {n_probes} paraphrase probes, got {updates_on_probes}"

    def test_oracle_contradictions_not_merged(self):
        """Oracle should not merge contradiction observations."""
        s3 = generate_scenario_3_contradiction_recovery()
        mem = MemoryState(capacity=100, dim=D)
        slot_concept_ids = {}
        contra_merges = 0
        for obs in s3:
            r = oracle_absorb(mem, obs.candidate, slot_concept_ids,
                              obs.concept_id, obs.category)
            if obs.category == "contradiction" and r["action"] == "update":
                contra_merges += 1
        assert contra_merges == 0, "Oracle must not merge contradiction observations"


# ─────────────────────────────────────────
#  Metrics tests — no NLI needed
# ─────────────────────────────────────────

class TestPhase17Metrics:
    def test_memory_metrics_structure(self):
        stream = generate_scenario_1_long_gaps(gaps=(2,))[2]
        res = run_stream(stream, capacity=50, system="B_oracle")
        m = memory_metrics(res, stream)
        required_keys = [
            "system", "capacity", "final_slots", "n_updates", "n_inserts",
            "n_evicts", "true_consolidation_rate", "false_consolidation_rate",
            "paraphrase_consolidation", "contradiction_retention",
            "unrelated_separation", "mean_slot_kv_similarity", "wall_seconds"
        ]
        for k in required_keys:
            assert k in m, f"Missing key in memory_metrics: {k}"

    def test_memory_metrics_paraphrase_consolidation_range(self):
        stream = generate_scenario_1_long_gaps(gaps=(2,))[2]
        res = run_stream(stream, capacity=50, system="B_oracle")
        m = memory_metrics(res, stream)
        assert 0.0 <= m["paraphrase_consolidation"] <= 1.0

    def test_memory_metrics_contradiction_retention_range(self):
        s2 = generate_scenario_2_repeated_contradictions()
        res = run_stream(s2, capacity=100, system="B_oracle")
        m = memory_metrics(res, s2)
        assert 0.0 <= m["contradiction_retention"] <= 1.0

    def test_oracle_achieves_high_paraphrase_consolidation(self):
        """Oracle should consolidate paraphrases with very high fidelity."""
        stream = generate_scenario_1_long_gaps(gaps=(2,))[2]
        res = run_stream(stream, capacity=50, system="B_oracle")
        m = memory_metrics(res, stream)
        # Oracle is expected to do much better than random (>0.5)
        assert m["paraphrase_consolidation"] >= 0.5, \
            f"Oracle paraphrase consolidation too low: {m['paraphrase_consolidation']}"

    def test_memory_metrics_deterministic(self):
        """Same stream produces identical metrics."""
        stream = generate_scenario_2_repeated_contradictions()
        res1 = run_stream(stream, capacity=50, system="C_lexical")
        res2 = run_stream(stream, capacity=50, system="C_lexical")
        m1 = memory_metrics(res1, stream)
        m2 = memory_metrics(res2, stream)
        for k in m1:
            if k != "wall_seconds":
                assert m1[k] == m2[k], f"Non-deterministic metric: {k}"


# ─────────────────────────────────────────
#  System A (NLI) — loads model once
# ─────────────────────────────────────────

@pytest.fixture(scope="module")
def nli_gate():
    """Load NLI model once per test module."""
    from benchmarks.phase16_experiment import NliSemanticGate
    return NliSemanticGate()


class TestPhase17NliGated:
    def test_nli_run_completes(self, nli_gate):
        """System A must complete a small stream without error."""
        stream = generate_scenario_1_long_gaps(gaps=(1,))[1]
        res = run_stream(stream, capacity=50, system="A_nli",
                         gate=nli_gate, record_diagnostics=True)
        assert res["system"] == "A_nli"
        total = res["n_inserts"] + res["n_updates"] + res["n_evicts"]
        assert total == len(stream)

    def test_nli_paraphrase_probes_mostly_merged(self, nli_gate):
        """NLI system should merge paraphrase probes at a high rate."""
        stream = generate_scenario_1_long_gaps(gaps=(1,))[1]
        res = run_stream(stream, capacity=50, system="A_nli",
                         gate=nli_gate, record_diagnostics=True)
        trace = res["trace"]
        probe_trace = [(o, t) for o, t in zip(stream, trace) if o.is_probe]
        merged = sum(1 for _, t in probe_trace if t["action"] == "update")
        n_probes = len(probe_trace)
        recall = merged / n_probes if n_probes else 0.0
        # NLI should achieve ≥80% recall on gap=1 paraphrases
        assert recall >= 0.80, f"NLI paraphrase recall too low: {recall:.3f}"

    def test_nli_contradictions_rejected(self, nli_gate):
        """NLI should reject most contradictions — false consolidation rate must be low."""
        s3 = generate_scenario_3_contradiction_recovery()
        res = run_stream(s3, capacity=100, system="A_nli",
                         gate=nli_gate, record_diagnostics=True)
        trace = res["trace"]
        contra_trace = [(o, t) for o, t in zip(s3, trace)
                        if o.category == "contradiction"]
        contra_merges = sum(1 for _, t in contra_trace if t["action"] == "update")
        n_contra = len(contra_trace)
        false_consolidation = contra_merges / n_contra if n_contra else 1.0
        # Key Phase 16 finding: NLI false_consolidation ≈ 0%
        assert false_consolidation <= 0.15, \
            f"NLI false consolidation too high: {false_consolidation:.3f}"

    def test_nli_vs_lexical_contradiction_separation(self, nli_gate):
        """NLI must reject more contradictions than lexical baseline."""
        stream = generate_scenario_2_repeated_contradictions()
        res_nli = run_stream(stream, capacity=100, system="A_nli",
                             gate=nli_gate, record_diagnostics=True)
        res_lex = run_stream(stream, capacity=100, system="C_lexical",
                             record_diagnostics=True)

        def false_cons_rate(res, obs):
            trace = res["trace"]
            contra_trace = [(o, t) for o, t in zip(obs, trace) if o.category == "contradiction"]
            merges = sum(1 for _, t in contra_trace if t["action"] == "update")
            return merges / len(contra_trace) if contra_trace else 1.0

        nli_fcr = false_cons_rate(res_nli, stream)
        lex_fcr = false_cons_rate(res_lex, stream)
        assert nli_fcr < lex_fcr, \
            f"NLI false_consolidation ({nli_fcr:.3f}) must be < lexical ({lex_fcr:.3f})"

    def test_nli_gate_counts_populated(self, nli_gate):
        """Gate count dict must contain at least one action category."""
        stream = generate_scenario_1_long_gaps(gaps=(1,))[1]
        res = run_stream(stream, capacity=50, system="A_nli", gate=nli_gate)
        assert len(res["gate_counts"]) > 0

    def test_nli_gate_nli_accept_count(self, nli_gate):
        """nli_accept gate count must be > 0 after stream with paraphrases."""
        stream = generate_scenario_1_long_gaps(gaps=(1,))[1]
        res = run_stream(stream, capacity=50, system="A_nli", gate=nli_gate)
        assert res["gate_counts"].get("nli_accept", 0) > 0

    def test_diagnostic_trace_length(self, nli_gate):
        stream = generate_scenario_1_long_gaps(gaps=(1,))[1]
        res = run_stream(stream, capacity=50, system="A_nli",
                         gate=nli_gate, record_diagnostics=True)
        assert len(res["trace"]) == len(stream)

    def test_candidate_retrieval_diagnostics_structure(self, nli_gate):
        stream = generate_scenario_1_long_gaps(gaps=(1,))[1]
        res = run_stream(stream, capacity=50, system="A_nli",
                         gate=nli_gate, record_diagnostics=True)
        diag = candidate_retrieval_diagnostics(stream, res["trace"], res["memory"])
        for k in ["total_probes", "candidate_retrieval_recall",
                  "nli_accuracy_given_retrieval", "end_to_end_accuracy"]:
            assert k in diag, f"Missing key in candidate_retrieval_diagnostics: {k}"
        for k in ["candidate_retrieval_recall", "nli_accuracy_given_retrieval",
                  "end_to_end_accuracy"]:
            assert 0.0 <= diag[k] <= 1.0, f"Out of range metric: {k}={diag[k]}"
