# MINEMOS Phase 29: Temporal Invariant Coverage and Verifier Telemetry

**Date:** October 29, 2026 (corrected October 30, 2026 after independent audit)
**Baseline commit:** `3afc2e480ba09d23c696d123fcc5c59fb5eb39a8`
**Status:** COMPLETE — scoped questions answered; one known false-negative class remains documented
**Artifacts:** `reports/PHASE_29_PLAN.md`, `benchmarks/phase29_integrity.py`, `runners/run_phase29.py`, `benchmarks/results/phase29_results.json`, `tests/test_phase29_temporal_integrity.py`

> **Correction notice (2026-10-30).** An independent audit found that this
> report described all 24 benchmark cases as pre-registered, while
> `PHASE_29_PLAN.md` enumerates 20. It also found that the invariant silently
> ignored structurally broken links, that a recurrence with no active version
> inserted an unlinked slot, and that the runner exited 0 while the known gap
> was open. Sections 4, 6, 7 and 10 record the corrected provenance, the fixed
> gaps, and the new verdict semantics. The original benchmark JSON is
> unchanged; fresh results were written to
> `scratch/phase29_audit/phase29_results_audit_20261030.json`.

---

## 1. Scope

Phase 29 follows the October 29 reliability repair (`3afc2e4`, R1–R3). It asks
whether `temporal_history_violations()` is sound and complete over the state the
engine can hold, and whether SAME-over-changed-value handling is safe when the
matched slot is historical. The plan's 20-case matrix was written before
execution in `reports/PHASE_29_PLAN.md`; four further cases were added before
the benchmark ran and are identified by provenance in §4. They are not
described as pre-registered.

Full-scale end-to-end accuracy (README roadmap item 2) is **not** part of this
phase; it measures retrieval/retention at scale and would confound attribution.

## 2. Changes

1. **Link validation in the invariant.** Each supersession link is checked
   before unioning. Different canonical subjects, or predicates in provably
   different known exclusive groups, produce `{"reason": "inconsistent_link",
   "subject_match": ..., "attribute_match": ...}` and the pair is not grouped.
   Paraphrases and negated restatements are not treated as attribute mismatches,
   because the engine legitimately links them.
2. **Closed-history SAME recurrence.** When a SAME verdict matches a closed
   historical slot and no candidate was selected, the engine now resolves the
   active chain head: identical value → consolidate; later timestamp → temporal
   update; earlier timestamp → historical splice. This removes the case where a
   recurrence was inserted as a second active version with no interval or link.
3. **Verifier telemetry.** `_values_are_equivalent()` increments
   `total_value_verifications` only when it actually calls the configured
   verifier; `status()` exposes the counter separately from `total_nli_calls`.
4. **Structural link anomalies (2026-10-30).** `temporal_history_violations()`
   now reports `dangling_link` (link target is not in memory), `self_link`
   (slot points at itself), and `non_reciprocal_link` (only one side confirms
   the relationship). Such pairs are never grouped.
5. **Recurrence without an active version (2026-10-30).** When every version of
   an attribute is closed, the recurrence is inserted as the new active version
   and linked to the nearest prior version, so the chain remains traversable.
6. **Explicit benchmark verdicts (2026-10-30).** Every case is classified as
   `expected_reproduced`, `unexpected_failure`, `known_limitation_confirmed`,
   or `known_limitation_resolved`. The runner exits 0 only for a clean pass,
   1 for unexpected failures, 2 while a documented limitation is open, and 3
   when a limitation changes state. `--output` allows an audit run without
   touching the canonical JSON.
7. New Phase 29 test suite, benchmark, runner, and result JSON.

No retention, checkpoint-schema, threshold, or exclusivity change was made.
The consolidation algorithm is unchanged; the recurrence branch now reuses it
for the active head and the insert branch attaches provenance only in the
closed-history case. `telemetry mode` remains non-exclusive.

## 3. Baseline probe (before the change)

Deterministic closed-history recurrence (`Maya city Rome` at 10, `Venice` at 20,
`Rome` again at 30; active head classified NEUTRAL):

```
actions:    ['insert', 'temporal_update', 'insert']
slots:      Rome [10,20) inactive, Venice [20,None) active, Rome [30,None) active
violations: []
```

Two active versions of one exclusive attribute existed with no link and no
diagnostic. That is the H2 gap; `tests/test_phase29_temporal_integrity.py`
now fixes and guards it.

After the change the same probe yields:

```
actions:    ['insert', 'temporal_update', 'temporal_update']
history:    [(10,20), (20,30), (30,None)]
violations: []
```

The probe script is `scratch/phase29_baseline_probe.py` (gitignored). The
pre-fix transcript was captured before the implementation edit and cannot be
re-executed in a read-only audit; its mechanism is the same unlinked path that
U1 still demonstrates (`test_unlinked_exclusive_conflict_remains_a_known_false_negative`).

## 4. Case matrix results

`benchmarks/results/phase29_results.json` — 24 executed cases: 23 passed, 1
known gap confirmed, 0 failed. Of these, **20 were pre-registered** in
`PHASE_29_PLAN.md`; four were not in the plan's case matrix:

| Case | Provenance | Evidence |
| --- | --- | --- |
| L7 paraphrase consolidation | added while building the benchmark | absent from plan §7; benchmark source predates the result JSON |
| L8 identical value reassertion | added while building the benchmark | absent from plan §7 |
| H4 closed recurrence, unchanged value | added while building the benchmark | absent from plan §7 (this is a case ID, distinct from plan hypothesis H4); an equivalent behaviour test was written before the implementation edit |
| NC04 pinned NLI boundary | implied by plan §8 ("real-NLI boundary check"), no case ID | plan §8 |

All 24 expectations were fixed in `benchmarks/phase29_integrity.py` before
`phase29_results.json` was generated (source mtime 15:11:37, result mtime
15:12:59 on 2026-10-29), but only 20 can be called pre-registered. The original
report's "24 pre-registered" wording was inaccurate. The source mtime was
observed during the audit; the 2026-10-30 docstring correction changed it, so
the ordering above is recorded here rather than re-derived from the current
file. Case logic was not altered by that correction.

| Class | Cases | Result |
| --- | ---: | --- |
| Legitimate coexistence | 8 | 8/8, **0 false positives** |
| Structural corruption | 6 | 6/6 detected |
| Historical SAME recurrence | 4 | 4/4 overlap-free |
| Verifier boundary | 3 | 3/3 expected action |
| Pinned real NLI (NC04) | 1 | passed |
| Telemetry | 1 | passed |
| Known gap (U1) | 1 | confirmed false negative, excluded from pass count |

| ID | Observed |
| --- | --- |
| L1–L6 | expected active counts; `violations: []` |
| L7 | consolidate; intervals `[[10, null]]` |
| L8 | consolidate; one active slot |
| C1 | `overlap` |
| C2 | `multiple_active`, `overlap` |
| C3 | `multiple_active`, `overlap` |
| C4 | `invalid_interval` |
| C5 | `inconsistent_link` (subject mismatch) |
| C6 | `inconsistent_link` (attribute mismatch) |
| H1/H2 | `temporal_update`; `[[10,20],[20,30],[30,null]]` |
| H3 | `insert`; two active memberships |
| H4 | `consolidate`; `[[10,20],[20,null]]` |
| V1 | `temporal_update`; 1 value check |
| V2 | `consolidate` |
| V3 | `insert`; two active values |
| NC04 | `insert`; two active values; prior state preserved |
| U1 | two active unlinked city values; `violations: []` (known gap) |
| T1 | `total_value_verifications=1`, `total_nli_calls=1` |

Corruption cases are scored by required-reason inclusion: C3 also reports
`overlap`, which is correct for two active intervals, and does not change the
detection verdict.

**Scoring provenance for C3 (corrected 2026-10-30).** The pre-registered C3
expectation listed only `multiple_active`. When C3 was executed it also
reported `overlap`, which is the correct consequence of two simultaneously
active intervals. The subset rule (every pre-registered reason must appear;
additional correct reasons are allowed) was introduced at that point, after
observing the case, and applied uniformly to all six corruption cases. C1, C2,
C4, C5 and C6 match exactly, so no other verdict depends on the rule. The
original report disclosed the rule but not that it was adopted after the C3
observation.

## 5. Regression checks

At phase completion (2026-10-29):

- New Phase 29 suite: **10 passed** (`tests/test_phase29_temporal_integrity.py`).
- Focused Phase 28/29/repair set: **94 passed in 157.13 s**
  (repair, both Phase 28 contradiction suites, Phase 28 adversarial, checkpoint
  integrity, Phase 29).
- The first link-validation draft flagged a legitimate negated-chain link
  (`holds a PhD` / `does not hold a PhD`) as an attribute mismatch. The rule was
  narrowed to provably different *known* exclusive groups; the test now passes.
- Full suite: **368 passed, 0 failed, 0 errors** across two file groups, both
  exit 0 — group A 191 passed in 723.31 s, group B 177 passed in 106.12 s. The
  split covered every collected test file exactly once with no deselections.

Post-audit re-validation (2026-10-30, repaired working tree):

- Phase 29 suite: **22 passed in 53.69 s**, exit 0
  (`.venv/Scripts/python.exe -m pytest -q --tb=short tests/test_phase29_temporal_integrity.py`;
  log `scratch/phase29_audit/phase29_suite_repair_20261030.log`).
- Focused set (the same six files as before): **106 passed in 114.80 s**,
  exit 0 (`scratch/phase29_audit/focused_repair_20261030.log`). The 106 are
  22 Phase 29 + 20 reliability + 17 checkpoint + 10 Phase 28 adversarial +
  33 Phase 28 contradiction adversarial + 4 Phase 28 regression.
- Full suite: **380 passed, 0 failed, 0 errors**, both groups exit 0; see §7.
- The four new structural/recurrence tests were executed against the pre-fix
  implementation first and failed there (no-head recurrence inserted an
  unlinked slot; dangling, self and non-reciprocal links were not reported).
  That output is session transcript only, not a repository artifact; the
  passing post-repair runs above are the reproducible evidence.

## 6. Carried risks — status

1. **Unlinked exclusive history (risk 1):** still not detected (U1). Without
   provenance, an imported conflict is indistinguishable from the conservative
   retention case that R2 requires to stay silent. Auto-grouping by predicate
   would re-create R2. Future work: persist slot provenance/`origin` and let the
   invariant group engine-produced state separately from imported state; the
   design is recorded in §10.1 and is not implemented.
2. **Link grouping validation (risk 2):** implemented. Subject mismatch is
   always reported; attribute mismatch is reported only when both predicates
   belong to known, different exclusive groups. Unknown/paraphrased attributes
   are accepted conservatively. Dangling (target absent), self-referential,
   and non-reciprocal links are reported as `dangling_link`, `self_link`, and
   `non_reciprocal_link`, and are never grouped (added 2026-10-30).
   Checkpoint loading independently rejects self-links, out-of-range indices,
   and non-reciprocal pairs before restoring any slot.
3. **Historical SAME (risk 3):** fixed and tested (H1–H4, plus the new gap,
   no-active-head and equal-timestamp tests). Multi-valued predicates still
   insert by design (H3).
4. **Verifier-dependent value equivalence (risk 4):** only a `SAME` decision
   counts as equivalent; `NEUTRAL`/`CONTRADICTION` count as a change. Confidence
   is not used as a threshold, so a low-confidence SAME still consolidates. The
   pinned model is checked in NC04, but results remain model/version-specific.
5. **`telemetry mode` (risk 5):** still not exclusive. NC04 preserves both
   values; whether the domain treats it as single-valued is unresolved, so no
   group was added.
6. **Extra verifier calls (risk 6):** counted separately in
   `total_value_verifications`. They are **not** added to `total_nli_calls`, are
   not persisted in checkpoints, and are not cached; each non-identical value
   comparison can add one verifier call. The counter is per instance and
   in-memory: a new instance starts at 0, `load_checkpoint()` into an existing
   instance leaves that instance's counters untouched, and no counter is read
   back from an archive.

## 7. Full-suite verification

The single-invocation suite exceeds the 10-minute foreground limit, so it was
executed as two sequential pytest invocations over disjoint file lists:

```bash
# Group A — exit 0
.venv/Scripts/python.exe -m pytest -q --tb=short \
  scratch/test_cross_process_lifecycle.py \
  tests/test_adversarial_verification.py tests/test_basic.py \
  tests/test_checkpoint.py tests/test_checkpoint_integrity.py \
  tests/test_concurrency_safety.py tests/test_contradiction_safety.py \
  tests/test_e2e_recovery.py tests/test_memory_state.py \
  tests/test_oracle_ablation.py tests/test_phase12_addressing.py \
  tests/test_phase13.py tests/test_phase14.py tests/test_phase15.py \
  tests/test_phase16.py tests/test_phase17.py tests/test_phase18.py \
  tests/test_phase19.py tests/test_phase20.py tests/test_phase21.py
# 191 passed in 723.31 s (0:12:03)

# Group B — exit 0
.venv/Scripts/python.exe -m pytest -q --tb=short \
  tests/test_phase22.py tests/test_phase23.py tests/test_phase24.py \
  tests/test_phase25.py tests/test_phase26.py tests/test_phase27.py \
  tests/test_phase28_adversarial.py \
  tests/test_phase28_contradiction_adversarial.py \
  tests/test_phase28_false_contradiction_regression.py \
  tests/test_phase29_temporal_integrity.py tests/test_reliability_repairs.py \
  tests/test_security_remediation.py tests/test_semantic_encoder.py
# 177 passed in 106.12 s (0:01:46)
```

Combined: **368 passed, 0 failed, 0 errors**, covering every file in
`pytest --collect-only` exactly once.

Post-audit rerun (2026-10-30) on the repaired working tree, same split:

```bash
# Group A — exit 0
.venv/Scripts/python.exe -m pytest -q --tb=short \
  scratch/test_cross_process_lifecycle.py \
  tests/test_adversarial_verification.py tests/test_basic.py \
  tests/test_checkpoint.py tests/test_checkpoint_integrity.py \
  tests/test_concurrency_safety.py tests/test_contradiction_safety.py \
  tests/test_e2e_recovery.py tests/test_memory_state.py \
  tests/test_oracle_ablation.py tests/test_phase12_addressing.py \
  tests/test_phase13.py tests/test_phase14.py tests/test_phase15.py \
  tests/test_phase16.py tests/test_phase17.py tests/test_phase18.py \
  tests/test_phase19.py tests/test_phase20.py tests/test_phase21.py
# 191 passed in 713.37 s (0:11:53)
# log: scratch/phase29_audit/group_a_repair_20261030.log

# Group B — exit 0
.venv/Scripts/python.exe -m pytest -q --tb=short \
  tests/test_phase22.py tests/test_phase23.py tests/test_phase24.py \
  tests/test_phase25.py tests/test_phase26.py tests/test_phase27.py \
  tests/test_phase28_adversarial.py \
  tests/test_phase28_contradiction_adversarial.py \
  tests/test_phase28_false_contradiction_regression.py \
  tests/test_phase29_temporal_integrity.py tests/test_reliability_repairs.py \
  tests/test_security_remediation.py tests/test_semantic_encoder.py
# 189 passed in 88.08 s (0:01:28)
# log: scratch/phase29_audit/group_b_repair_20261030.log
```

Collection after the additions reports **380 tests in 33 files** in 41.75 s
(`scratch/phase29_audit/collected_repair_20261030.txt`). A scripted comparison
confirms group A holds 20 files/191 tests, group B holds 13 files/189 tests,
the union is exactly the 33 collected files, the intersection is empty, and no
file appears twice: 191 + 189 = 380. Combined: **380 passed, 0 failed,
0 errors**, every collected test file covered exactly once.

## 8. Files

Modified: `phase2/black_hole_memory.py`.
Added: `benchmarks/phase29_integrity.py`, `runners/run_phase29.py`,
`benchmarks/results/phase29_results.json`, `reports/PHASE_29_PLAN.md`,
`reports/PHASE_29_REPORT.md`, `tests/test_phase29_temporal_integrity.py`.

No canonical benchmark result, historical report, license, or dependency file
was modified; `benchmarks/results/phase29_results.json` was read for comparison
but never written. Validation artifacts are gitignored under
`scratch/phase29_audit/`: the audit result JSON, the collection listing, and the
group A, group B, Phase 29 and focused run logs. The reviewed Phase 29 changes
are committed in this phase's commit.

## 9. Assessment

Within the pre-registered scope, link validation and closed-history recurrence
are fixed, legitimate coexistence remains violation-free, all injected structural
corruptions are reported, and the pinned NLI still refuses the NC04 destructive
merge. The invariant remains deliberately silent on unlinked imported conflicts;
that is a measured limitation, not a solved problem. This phase does not
establish production readiness, and the full-scale E2E experiment remains open.

## 10. Post-audit corrections (2026-10-30)

| Original claim | Corrected statement | Evidence |
| --- | --- | --- |
| "24 pre-registered cases" | 20 cases are in `PHASE_29_PLAN.md` §7. L7, L8 and H4 were added while building the benchmark (before execution, not pre-registered); NC04 is implied by plan §8 without a case ID. The runner now records provenance per case. | Plan §7/§8 text; benchmark source mtime 2026-10-29 15:11:37 (observed before the docstring correction) precedes the result JSON at 15:12:59 |
| "The invariant is sound over structural corruption" | Dangling, self-referential and non-reciprocal links were silently skipped by the union logic. They are now reported as `dangling_link`, `self_link` and `non_reciprocal_link`, and such pairs are never grouped. Checkpoint loading separately rejects self, out-of-range and non-reciprocal pairs. | The three structural tests failed when first executed on the pre-fix code and pass now |
| "Historical SAME recurrence is resolved" | The original branch only handled recurrences with an active version. With every version closed, the recurrence was inserted unlinked. It is now linked reciprocally to the nearest closed predecessor. | `test_recurrence_without_active_head_links_prior_version` failed when first executed on the pre-fix code and passes now |
| "Runner exit 0 means success" | Exit 0 now means every case reproduced and no known limitation is open. An open limitation exits 2, an unexpected failure exits 1, and a changed limitation state exits 3. | Audit run exit 2 with `status: known_limitations_open` |
| "C3 scoring used required-reason inclusion" | The subset rule was adopted after observing that C3 also reports `overlap`; it was then applied uniformly. No other verdict depends on it. | Disclosed in §4 |
| Pre-fix probe transcript | The transcript cannot be re-executed in a read-only audit. U1 reproduces the same unlinked path and is the standing evidence. | §3 |

The original `benchmarks/results/phase29_results.json` is byte-identical
(SHA-256 `93e9a75d…`, mtime 2026-10-29 15:12:59). The post-audit benchmark ran
against the repaired code and wrote
`scratch/phase29_audit/phase29_results_audit_20261030.json` with
`status: known_limitations_open`, 23 `expected_reproduced` and 1
`known_limitation_confirmed` (U1). Provenance counts: 20 `plan_registered`,
3 `added_before_execution`, 1 `plan_implied`. A case-level comparison shows the
canonical and audit runs agree on every record (`passed`, `observed`,
`expected`, `match`, `known_gap` for all 24); only `wall_seconds` differs
(2.39 s vs 3.74 s). The repairs therefore did not change any benchmark
outcome.

The Phase 29 test file grew from 10 to 22 tests. The new tests cover the
interval gap splice, the no-active-head link, equal-timestamp in-place
replacement, the three structural link anomalies, exclusivity independent of
value similarity, checkpoint link roundtrip, malformed-archive rejection, and
verdict classification. Each test asserts actions, active counts, intervals,
link reciprocity and violation reasons rather than encoding observed output.
For the four tests that target repaired behaviour, the expected state was
written before the implementation change and the test was run first on the
pre-fix code (it failed). The remaining new tests specify already-existing
behaviour (checkpoint rejection, verdict classification, exclusivity by group
membership) from the stated requirements rather than from a captured run.

### 10.1 Provenance design proposal (document only, not implemented)

U1 cannot be fixed by grouping every same-subject/same-predicate pair: that is
exactly the R2 false positive (legitimate multi-valued facts such as
`works on`). The following design is recorded for review and is **not** part of
this repair.

- **Field.** Add `origin` to `MemorySlot` with values `"engine"` (created by
  `absorb` through the contradiction pipeline) and `"imported"` (restored from
  an external archive or bulk loader that does not link slots).
- **Persistence.** Bump the checkpoint schema to v4 and store one byte per
  slot. v3 archives are loaded as `"unknown"`, which is treated like
  `"imported"` so old archives cannot produce new false positives.
- **Invariant.** For an unlinked pair sharing a subject and a known exclusive
  group: if both slots are `"engine"`, an unlinked conflict is an engine defect
  and may be reported as a violation; if either slot is `"imported"` or
  `"unknown"`, emit an advisory `unlinked_conflict` record. The advisory does
  not change retention or query behaviour, preserving R2's requirement that
  the conservative case stay silent. U1 is the imported/unknown case.
- **Testing.** Extend `test_checkpoint_integrity.py` for v3→v4 migration and the
  Phase 29 suite for engine/imported/mixed pairs, including a multi-valued
  predicate that must remain violation-free.
- **Why deferred.** It changes persisted state and the meaning of an existing
  diagnostic, so it needs its own pre-registered phase rather than an
  unreviewed schema change inside an audit repair.
