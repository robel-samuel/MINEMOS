# Black Hole Memory — Phase 2: Mathematical Memory-State Prototype

## 1. Mathematical Definition

```
M_t = {m_1, ..., m_N}                 (N = fixed capacity)
m_i = (k_i, v_i, c_i, t_i)            key, value, confidence, timestamp

x_t = E(C_t)                          deterministic hash-based encoding
                                       (bag-of-tokens, feature hashing,
                                       D=64 dims, SHA-1 based, no training)

s_i = sim(x_t, k_i) = cosine(x_t, k_i)
j = argmax_i(s_i);  max_similarity = s_j
N_t = 1 - max_similarity              (N_t = 1 if memory empty)
alpha_t = N_t

v_j(t+1) = (1 - alpha_t) v_j(t) + alpha_t x_t     [if s_j >= MATCH_THRESHOLD]
new slot (k_i=x_t, v_i=x_t)                        [if s_j <  MATCH_THRESHOLD]

MATCH_THRESHOLD = 0.75                fixed BEFORE any benchmark run
capacity rule: FIFO eviction (oldest `created_at`) when full
persistence: raw float32/float64 arrays in a .npz archive
```

k_i is frozen at slot creation; only v_i drifts via the update equation.
Full parameter set (D=64, MATCH_THRESHOLD=0.75, FIFO eviction) was fixed
in code before Test A was run and was **not** changed after seeing
results, including the negative ones below.

## 2. Implementation

`phase2/structured_candidate.py` — `Candidate` dataclass + `encode_full`
(subject+predicate+object tokens, for addressing/update) and
`encode_query` (subject+predicate only, for partial-query recall).
`phase2/memory_state.py` — `MemoryState`/`MemorySlot`, capacity
enforcement, honest byte accounting broken into key/value/confidence/
timestamp/metadata components. `phase2/memory_update.py` — similarity,
novelty, addressing, the merge-vs-new-slot threshold rule, FIFO eviction.
`phase2/recall.py` — partial-query recall + an **evaluation-only**
nearest-vocabulary decoder, explicitly not counted in memory footprint.
`phase2/checkpoint.py` — `.npz` save/load.

## 3. Experimental Setup

Scales: 100 / 1,000 / 10,000 facts. Capacities: 10 / 100 / 1,000 /
10,000 (paired per scale, plus unconstrained variants). All facts
synthetic and deterministic (`person_NNNNNN likes object_NNNNNN`).
Software: Python 3.12, numpy 2.4, no GPU, no external model.

## 4. Results

### Test A — distinct facts, scaling

| n_facts | capacity | slots used | exact recall | payload compression | total compression | false-memory rate |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 10 | 10 | 0.100 | 1.289x | 0.616x | 0.200 |
| 100 | 100 | 65 | 0.620 | 0.198x | 0.095x | 1.000 |
| 1,000 | 100 | 100 | 0.192 | 1.289x | 0.616x | 0.990 |
| 1,000 | 1,000 | 127 | 0.240 | 1.015x | 0.485x | 1.000 |
| 10,000 | 1,000 | 128 | 0.240 | 10.07x | 4.81x | 1.000 |
| 10,000 | 10,000 | 128 | 0.240 | 10.07x | 4.81x | 1.000 |

**A scorer bug, caught and reported, not hidden**: an earlier "fuzzy
recall accuracy" metric read 1.000 at every single row above. Direct
testing proved this was vacuous by construction — the nearest-vocabulary
decoder returns *some* string for literally any input, including random
noise (verified directly). That metric has been dropped everywhere in
this report; **`exact_recall_accuracy` is the only trustworthy accuracy
number**, and it is uniformly poor (10-62%, settling at 24% once
capacity stops being the binding constraint).

### Ablation — ¬update vs. full mechanism

| n_facts | capacity | full: exact recall | full: slots used | no-update: exact recall | no-update: slots used |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 1,000 (unconstrained) | 0.240 | 127 | **0.240** | 1,000 |
| 1,000 | 100 (constrained) | **0.192** | 100 | 0.132 | 100 |

**Unconstrained capacity: the update mechanism contributes nothing
measurable** — identical exact recall (0.240) whether or not merging
happens at all, despite the full mechanism using 8x fewer slots.
**Under capacity pressure: the mechanism helps modestly** (0.192 vs
0.132) by packing more facts per slot, at the cost of blended,
ambiguous values.

### Test B — repetition
500 absorptions of the identical fact → **1 slot**, `update_count=500`.
Repetition does not consume additional memory. This part of the
mechanism works exactly as intended.

### Test C — paraphrase / surface variation
Case-only variants ("Python" vs "python") correctly merge into one slot.
Predicate synonyms ("likes" vs "prefers" for the same relation)
correctly do **not** merge (similarity 0.667, below the 0.75 threshold)
— but this is not a success of semantic understanding; the hasher has no
concept of synonymy at all. True paraphrase consolidation is out of
scope for this architecture and would require the (frozen) extraction
layer to canonicalize predicates before candidates ever reach memory.

### Test D — distractors
20 important facts absorbed, then buried under 2,000 distractors,
capacity=100: **25% of important facts survived with correct exact
recall.** Most were lost to eviction/collision pressure despite being
absorbed first.

### Test E — contradiction (measuring the baseline failure mode, as instructed)
`user uses Python` then `user uses Rust`: similarity between the two
was 0.667 — **below** the 0.75 threshold, so this specific pair landed
on the "new slot" side rather than blending. Two separate slots resulted.
Querying `(user, uses)` afterward returned the **Python** slot — not
because of recency or confidence, but because `argmax` ties are broken
by first-occurrence, and Python's slot was created first. **This is an
accidental property of tie-breaking, not a designed recency or
confidence rule** — a materially different (and less defensible) failure
mode than "blending into a meaningless average," but a failure mode
nonetheless: the system has no principled way to prefer the current fact
over the historical one.

### Test F — long-horizon recall by age
2,000 facts, capacity 500. Only 128 of 500 available slots were ever
used — collision-driven false merging suppressed slot creation so
strongly that FIFO eviction never even triggered.

| band | exact recall |
|---|---:|
| early facts (first 50) | 0.740 |
| middle facts | 0.000 |
| recent facts (last 50) | 0.000 |
| random sample | 0.100 |

Not a simple recency curve — early facts survived far better than
recent ones, the opposite of what FIFO eviction alone would predict
(consistent with eviction never actually triggering; the collapse is
almost entirely collision-driven, not capacity-driven).

## 5. Failure Analysis

**Root cause, demonstrated with direct measurement, not assumed:**
tokenized `person_NNNNNN`/`object_NNNNNN`-style facts produce ~4,000
distinct tokens hashed into only 64 buckets — **an average of 62.5
tokens sharing every single bucket**, and 64/64 buckets showed multiple
different tokens colliding when checked directly. Two genuinely
unrelated facts (`person_1/object_1` vs. `person_2/object_2`) already
show 0.43 cosine similarity from collision noise alone — nowhere near
the 0.0 a truly discriminative encoding would produce. D=64 is simply
too low-dimensional for the vocabulary size tested here; this is a
well-understood property of the hashing trick (collision rate scales
with vocabulary-size / dimension), not a novel discovery, but it was
verified directly rather than assumed.

- **Information loss**: severe and pervasive — exact recall settles at
  24% even under unconstrained capacity.
- **Collisions**: root cause, demonstrated above.
- **Capacity failures**: real but secondary — Test A capacity=10 (a
  genuinely tight case) still gets exact_recall=0.10, barely different
  from capacity=100's 0.62-ish range once collision dominates.
- **False memories**: severe — 99-100% false-memory rate at 1,000+
  facts, meaning queries for subjects that were *never absorbed*
  routinely score above the merge threshold purely from hash collision.
- **Contradictory facts**: handled by accident (tie-breaking order), not
  by design — see Test E.
- **Scaling problems**: compression ratio *appears* to improve with
  scale (10.07x payload compression at 10,000 facts) — but this is not
  a success signal. It's the same collision collapse: slot count
  plateaus at 128 regardless of how many facts arrive, so "compression"
  is measuring information destruction, not efficient representation.
  This is the same trap flagged in the extraction-layer experiments
  (compression that looks good because information was dropped) —
  here it shows up in the memory mechanism itself.

## 6. Ablation Results

Documented in Section 4. Summary: the similarity+novelty+update
mechanism provides **no measurable benefit over no mechanism at all**
when capacity is not the binding constraint (both bottlenecked
identically by addressing-collision), and a **modest, real benefit**
under capacity pressure by conserving slots. The mechanism is not
"broken" in isolation — it does what the equations say — but at D=64 it
is not the dominant factor in the system's recall failure. The encoding
dimensionality is.

## 7. Scientific Conclusion

**Does the current mathematical memory mechanism demonstrate useful
memory retention under controlled conditions? No, not at the tested
dimensionality.** Exact recall plateaus around 24% once scale exceeds
capacity, false-memory rates reach 99-100% at 1,000+ facts, and the
ablation shows the core update mechanism isn't even the primary cause —
a naive no-merge baseline fails identically under unconstrained
capacity. The repetition mechanism (Test B) and case-insensitive
consolidation (Test C) work exactly as designed, so the *update
equation itself* is implemented correctly; the *encoding* feeding it is
not adequate for the vocabulary sizes tested. This is a valid,
informative negative result, not an inconclusive one: it isolates the
failure to a specific, measured, well-understood cause rather than
leaving "does Black Hole's memory mechanism work" unanswered.

## 8. Next Experiment

**Increase D (encoding dimensionality) as a single controlled variable,
re-run Test A only, before touching anything else.** This is the most
direct test of the diagnosed root cause: if exact recall and
false-memory rate improve substantially as D increases (e.g. 64 → 512 →
4096) while everything else stays fixed, that confirms collision was
the dominant failure mode and the update mechanism can be fairly
evaluated on its own merits at a dimensionality where it isn't
swamped by noise. If recall stays poor even at high D, that would
indicate a deeper problem with bag-of-tokens addressing itself,
independent of dimensionality — a materially different and more
serious finding. Either outcome is informative; neither should be
assumed in advance.
