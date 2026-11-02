# MINEMOS — U1 Closure Report

**Date:** October 30, 2026
**Baseline commit:** `05fc8e4d9494795e02dacaed602daeda8e882cae`
**Status:** Implemented and locally validated; changes are uncommitted.
**Scope:** Close the Phase 29 U1 limitation (unlinked exclusive temporal
conflicts) and repair the noisy-language defects exposed while reproducing it.
Phase 30 was not started; no production-readiness claim is made.

> Scope note: the request's section 5D was truncated after the words
> "Extra spaces". The robustness matrix below covers every category named
> before the truncation: spelling mistakes, grammar errors, inconsistent
> capitalization, abbreviations, paraphrasing, and noisy input. No requirement
> beyond the visible text was invented.

---

## 1. Baseline and reproduction

The baseline was verified at `05fc8e4`, the commit that completed Phase 29.
Two independent probes were run before any code change; both are retained under
`scratch/u1_closure/` (git-ignored).

### 1.1 U1 itself

`scratch/u1_closure/reproduce_u1.py` absorbs two facts with the real pinned
encoders and a deterministic NEUTRAL verifier:

```
slot 0: text='Alice lives_in Addis Ababa.' active=True interval=(10.0,None) supersedes=None superseded_by=None
slot 1: text='Alice lives_in Nairobi.' active=True interval=(20.0,None) supersedes=None superseded_by=None
actions: ['insert', 'insert']
links present: False
violations: []
violation_count: 0
```

Two active, overlapping facts of one exclusive predicate, no supersession
link, and the structural invariant reports nothing. This is U1, reproduced
without manufacturing a failure.

### 1.2 Noisy-language probes

`scratch/u1_closure/nl_probe.py` ran 16 real-NLI scenarios against the same
baseline. The defects relevant to this mission were:

| Scenario | Baseline action | Baseline defect |
|---|---|---|
| `subject_typo_short` (`Alice` → `Alic`) | `insert, insert` | duplicate active slot for one fact |
| `value_typo` (`Nairobi` → `Narobi`) | `insert, temporal_update` | false state change; correct value closed |
| `historical_used_to` | `insert, consolidate` | current fact rewritten as a past statement |
| `historical_previous` | `insert, consolidate` | same |

The probe also confirmed baseline behavior that must be preserved: case/space
noise, grammar variants (`live in`), gerunds/present perfect, NYC, and
paraphrases already consolidated; distinct people/places stayed separate;
`AA` and `no longer` produced temporal updates.

The 42 existing Phase 29 and reliability tests passed at baseline
(`42 passed in 53.74 s`), so the repairs below are measured against a green
pre-change state.

---

## 2. Design

### 2.1 Predicate cardinality policy

Exclusivity is explicit, never inferred from predicate identity alone:

```
classify_predicate(predicate) -> EXCLUSIVE | MULTI_VALUED | UNKNOWN
```

Precedence:

1. per-instance `exclusive_predicates` override → `EXCLUSIVE`
2. per-instance `multi_valued_predicates` override → `MULTI_VALUED`
3. `CUMULATIVE_PREDICATES` → `MULTI_VALUED`
4. a marker in `EXCLUSIVE_SYNONYM_GROUPS` → `EXCLUSIVE`
5. otherwise → `UNKNOWN`

`UNKNOWN` predicates are never treated as exclusive. Overrides allow a caller
to reclassify a default group member (for example `status` as multi-valued)
without editing the engine. Overrides are deliberately not persisted in
checkpoints: an archive stores facts, not process configuration.

### 2.2 Conflict detector

New public method:

```python
BlackHoleMemory.exclusive_temporal_conflicts() -> List[dict]
```

For every pair of **active** slots `i < j` it reports one finding when all of
the following hold:

- the pair is not already connected by a supersession link in either direction
  (the structural invariant owns those pairs);
- the subjects are equivalent after normalization or a conservative one-edit
  spelling rule;
- both predicates resolve to `EXCLUSIVE`, and they are identical or share a
  configured exclusive synonym group;
- both values are known and are **not** lexically equivalent;
- the half-open intervals overlap.

Finding fields:

```
reason             "unlinked_exclusive_temporal_conflict"
slot_indices       [i, j]
slot_ids           [id, id]
subject            canonical (first slot) debug subject
predicates         [predicate_i, predicate_j]
values             [value_i, value_j]
valid_from         [t_i, t_j]
valid_until        [u_i, u_j]
overlap            True
predicate_policy   "exclusive"
detection_source   "active_unlinked_slots"
confidence         1.0  (deterministic policy determination, not semantic truth)
```

For three or more values, findings are pairwise and deterministic
(`[0,1], [0,2], [1,2]`).

The detector is read-only: it does not mutate slots, intervals, links, indexes,
or counters, and it **never calls a model**. It is therefore repeatable and
safe to run against imported or legacy state. Value equivalence in the detector
is lexical only; semantic paraphrase equivalence would require a verifier call
and is intentionally out of scope for a side-effect-free check.

`temporal_history_violations()` keeps its Phase 29 default behavior and gains an
opt-in parameter:

```python
temporal_history_violations(include_unlinked_conflicts=True)
```

The default remains `False` so the Phase 29 structural results stay exactly
reproducible. The U1 gap is closed by the new detector; the structural API was
not silently redefined.

### 2.3 Conservative lexical equivalence

Normalization: NFKC → casefold → whitespace collapse → edge-punctuation strip.
This handles capitalization, full-width characters, extra spaces, and trailing
punctuation without fuzzy matching.

Abbreviation aliases are a small, deliberately unambiguous table:
`nyc`, `uk`, `usa`, `uae`. `AA` is deliberately absent (Addis Ababa vs
American Airlines), so an ambiguous abbreviation is never merged; it remains a
visible conflict via the detector instead.

Typo equivalence is narrow:

- exactly one differing token;
- the token is alphabetic, at least 4 characters, same first character;
- the edit is an insertion, deletion, or adjacent transposition.

Single-letter substitutions are excluded: `Iran`/`Iraq` and `Rome`/`Roma` are
the ambiguous case where merging distinct names is worse than leaving a
misspelling unconflated. Numeric tokens are never typo-equivalent. Short names
are never typo-merged (`May`/`Maya`, `Alice`/`Alicia`).

These rules are used in `absorb()` as a deterministic fast path before a
semantic state change and in the detector. They are not applied blindly to all
text.

### 2.4 Temporal boundaries

- **Historical statements** (`used to`, `previously`, `formerly`, `in the
  past`, `at one time`) bypass retrieval/verification and are recorded as a
  closed zero-width observation `[t, t]`. They never consolidate into, or
  supersede, a current fact and are not linked into the interval chain.
  `no longer` is intentionally *not* historical: it asserts a current negation.
- **Negated restatements** are never lexically consolidated with an
  affirmative slot; the existing temporal-update path remains.
- **Surface variants** (subject/value typo or abbreviation) consolidate for
  counting and confidence, but the incumbent canonical wording is preserved in
  `text`, `latest_object`, and the indexes. Recall therefore does not regress
  to the noisy spelling.

### 2.5 Files changed

- `phase2/black_hole_memory.py` — policy, normalization, detector, absorb
  guards, surface-variant preservation.
- `tests/test_reliability_repairs.py` — the duplicate-active test now injects
  the duplicate directly, because identical restatements now consolidate. Its
  assertion (the invariant must report `multiple_active`) is unchanged.
- `tests/test_u1_closure.py` — new, 28 tests.
- `benchmarks/u1_closure.py` — new, 22 cases.
- `runners/run_u1_closure.py` — new runner.
- `benchmarks/results/u1_closure_results.json` — new result file.

No license, dependency, checkpoint-schema, persistence, or deployment code was
modified. The Phase 29 canonical result file was not overwritten.

---

## 3. Validation

### 3.1 Focused suites

```
pytest tests/test_u1_closure.py tests/test_phase29_temporal_integrity.py \
       tests/test_reliability_repairs.py \
       tests/test_phase28_contradiction_adversarial.py
```

Result: **103 passed in 50.23 s** (28 new U1 tests + 75 existing).

### 3.2 U1 closure benchmark

```
python runners/run_u1_closure.py
```

Result: **22/22 cases, status `all_expected_reproduced`, 1.30 s**.
The result file records the baseline commit and every case's expected/observed
pair. It is separate from `phase29_results.json`.

Coverage: unlinked detection, opt-in integration, multi-valued silence, unknown
silence, linked chains, closed history, distinct names, pairwise findings,
read-only/repeatability, explicit policy overrides, injected duplicate
corruption, subject/value typos, NYC/AA, case/width/space/punctuation,
historical statements, negation, numeric boundary, real-NLI grammar and
paraphrase, and the one-edit boundary.

### 3.3 Full suite

The suite collects **407 tests in the 33 files under `tests/`**; it was split
into two disjoint file groups because one foreground run can exceed the
10-minute tool limit. (A bare `pytest` also collects one git-ignored scratch
probe, `scratch/test_cross_process_lifecycle.py` -> 408 total; that probe is
not part of the reviewed change set and is excluded from this table.)

| Group | Tests | Result |
|---|---|---|
| B (checkpoint, concurrency, phases 12–15, 19–27, security) | 201 | 201 passed in 356.21 s (exit 0) |
| A (adversarial, phases 16–18, 28, 29, reliability, U1) | 206 | 206 passed in 156.45 s (exit 0) |

Both groups exited 0, and 201 + 206 = 407 matches the `--collect-only` count.
Every test is executed exactly once across the two groups. No test is claimed
as passing from an unexecuted run.

### 3.4 Phase 29 preservation

- Canonical `benchmarks/results/phase29_results.json` SHA-256:
  `93e9a75db5a68e5ef1ab156920971fc1d857982709de2f39434d5281a93ab137`
  — unchanged before and after this work.
- Replaying the Phase 29 benchmark on the new engine (to a scratch output)
  reproduces **23 of 24 cases**, confirms the structural U1 gap, and changes
  exactly one case: **C3**.

C3 injected its duplicate by relying on the old behavior of inserting a
duplicate when the verifier returned NEUTRAL for an identical restatement.
Identical restatements now consolidate, so C3 observes `active: 1` and no
`multiple_active`. This is the intended repair, not a regression, but it
supersedes C3's fixture. To keep the historical evidence intact:

- the Phase 29 benchmark and canonical result file were **not modified**;
- C3's corruption-detection intent is re-covered by `U1-12` (injected
  duplicate) and by the updated reliability regression test;
- the superseded expectation is recorded here rather than concealed.

---

## 4. Known limitations and residual risks

1. **Detector value equivalence is lexical.** Semantically equivalent
   paraphrases with different surface tokens (`Big Apple` vs `NYC`) can be
   reported as conflicts. This is deliberate: a read-only checker must not call
   a verifier or change `total_value_verifications`.
2. **Unknown predicates are not detected.** `commutes via`-style attributes
   remain silent unless explicitly configured as exclusive. This preserves the
   R2 guarantee against false positives.
3. **Substitution typos are not merged.** `Nairobj` is treated as a different
   value. This avoids `Iran`/`Iraq`-class false merges; it is a conservative
   loss of recall, not a safety failure.
4. **Historical records have no inferred window.** They are stored as
   `[t, t]` observations and never become active at any `as_of`. Their true
   start/end remain unknown; the engine does not invent them.
5. **Plain simple past is not a marker.** `used to`, `previously`, `formerly`,
   `in the past`, and `at one time` are handled; a bare `lived in` predicate is
   not classified historical. This can be extended if evidence requires it.
6. **Policy overrides are per-process.** They are not serialized. A restored
   checkpoint uses the loading process's policy.
7. **The structural invariant still defaults to silence on U1.** Callers must
   use `exclusive_temporal_conflicts()` or
   `temporal_history_violations(include_unlinked_conflicts=True)`. This is the
   compatibility choice that keeps Phase 29 reproducible.

---

## 5. Repository state

The changes are **uncommitted**. No commit or push was performed. No history
was rewritten. The only modified tracked file is
`phase2/black_hole_memory.py` plus the reliability-test fixture update; all
other work is new files. Scratch probes and logs remain under
`scratch/u1_closure/` and are not part of the reviewed change set.
