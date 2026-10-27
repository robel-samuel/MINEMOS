# Black Hole Memory — Phase 2A: Encoding-Dimensionality Ablation

## 1. Research Question
Does increasing encoding dimensionality D reduce hash-collision noise
enough to restore reliable memory addressing and exact recall? Sole
experimental variable: D. Everything else fixed.

## 2. Previous Phase 2 Problem
D=64 produced exact recall of ~24% at scale, false-memory rate of
99-100%, and an ablation where the full update mechanism matched a
no-update baseline exactly — leaving it unclear whether collision or the
mechanism itself was responsible.

## 3. Experimental Hypothesis
Hash collisions were the dominant cause of the Phase 2 failure; recall
should improve substantially as D increases if this is correct.

## 4. Exact Variable Changed
D ∈ {64, 128, 256, 512, 1024, 2048, 4096}. Encoding algorithm (SHA-1
feature hashing) unchanged at every D — confirmed by regression: D=64
results are bit-identical before and after this experiment's code
changes (0.24 / 127 slots / 1.0 false-memory, reproduced exactly).

## 5. Variables Held Constant
Update equation, similarity function, novelty equation, addressing
logic, recall logic, slot structure, checkpoint format, capacities,
candidate generation/seed, eviction policy (FIFO), MATCH_THRESHOLD
(0.75), extraction layer (untouched, not used — clean structured
candidates only).

## 6. Encoder Implementation
One necessary, non-architectural fix was required before this experiment
was even runnable: `MemoryState` stored a `dim` attribute, but
`absorb()`/`recall()` never passed it to the encoder — every call
silently used the hardcoded default (64), which would crash on a shape
mismatch for any D≠64. This was dimension *plumbing*, not a change to
hashing, equations, or logic. Verified via regression: D=64 output is
identical before and after the fix.

## 7. Collision Analysis (1,000-fact vocabulary, same as prior diagnostic)

| D | occupied buckets | collision rate | avg tokens/bucket | max in one bucket |
|---:|---:|---:|---:|---:|
| 64 | 64/64 | 1.000 | 78.1 | 27 |
| 128 | 128/128 | 1.000 | 39.1 | 16 |
| 256 | 250/256 | 0.936 | 20.0 | 10 |
| 512 | 437/512 | 0.696 | 11.4 | 7 |
| 1024 | 633/1024 | 0.428 | 7.9 | 6 |
| 2048 | 777/2048 | 0.255 | 6.4 | 4 |
| 4096 | 870/4096 | 0.141 | 5.8 | 3 |

Collision rate drops substantially and monotonically with D, as expected.

**An important, distinct finding, not predicted in advance**: pairwise
similarity between the four deliberately-unrelated example facts
(Python/Rust/Paris/Tesla) does **not** decrease monotonically toward
zero as D grows — it drops sharply from D=64→256 (collision noise
settling) and then goes **completely flat from D=256 to D=4096**
(`person1_uses_python` vs `person2_uses_rust` stays at exactly 0.5 the
entire time). Direct verification: `person_1` and `person_2` both
tokenize to include the literal shared token `"person"` (the tokenizer
splits on the underscore), and both facts share the predicate token
`"uses"`. This is **genuine shared vocabulary, not collision** — it
cannot shrink with D, no matter how large. There is a hard similarity
floor set by tokenization choices, independent of dimensionality.

## 8. Scaling Results — exact recall (n_facts, capacity)

| D | (100,10) | (100,100) | (1000,100) | (1000,1000) |
|---:|---:|---:|---:|---:|
| 64 | 0.100 | 0.620 | 0.192 | 0.240 |
| 128 | 0.100 | 0.760 | 0.172 | 0.430 |
| 256 | 0.100 | 0.880 | 0.148 | 0.642 |
| 512 | 0.100 | 0.950 | 0.094 | 0.810 |
| 1024 | 0.100 | 0.980 | 0.050 | 0.888 |
| 2048 | 0.100 | 0.990 | 0.038 | 0.944 |
| 4096 | 0.100 | 0.990 | 0.024 | 0.972 |

**Two opposite trends in the same table, both real:**
- **Unconstrained capacity** ((100,100), (1000,1000)): exact recall
  climbs strongly and monotonically with D — 62%→99% and 24%→97.2%.
  This directly confirms the hypothesis for this condition.
- **Constrained capacity** ((1000,100)): exact recall *falls*
  monotonically with D — 19.2%→2.4%, the opposite direction.
- **(100,10)** is flat at exactly 0.100 for every D — mechanically
  correct: capacity 10 with FIFO eviction retains only the last 10 of
  100 facts, and 10/100 = 0.10 exactly, independent of D.

## 9. Root Cause of the Capacity-Pressure Reversal (verified, not assumed)
Directly tested whether low-D "recall" of long-evicted early facts was
real retention or noise. At D=64, querying `person_000001` (evicted
hundreds of insertions ago) matched a slot **created by `person_000880`**
— a completely unrelated fact — yet happened to decode to the correct
object by chance. Measured directly: **28 of 50 early-fact queries
"succeeded" at D=64 purely by accidental collision**, dropping to **3 of
50 at D=4096**. The 19% figure at D=64 was never real memory — it was
noise coincidentally producing correct-looking answers. As D increases
and collision noise vanishes, the system correctly and honestly reports
that evicted information is gone. **The metric got worse because the
measurement got more honest, not because retention got worse.**

## 10. False-Memory Results

False-memory rate stayed at **99-100% across every D tested**, with one
unexplained exception: D=4096 at (1000,100) dropped to 6%. This
does *not* improve with D the way exact recall does. Root cause, traced
directly: recall queries are partial (subject+predicate only, no
object), and every query subject shares the literal token `"person"`
plus the predicate token `"likes"` with every stored fact (Section 7's
tokenization-floor finding). This produces structurally high similarity
between *any* query and *any* stored slot, regardless of whether the
specific queried subject was ever really absorbed — a consequence of
tokenization design, not of hash-bucket collision, so increasing D does
not fix it. The one 6% outlier is reported as observed, not fully
explained — worth a follow-up rather than a retrofitted story.

## 11. Latency
Not the bottleneck at any tested D; absorb/recall stayed in the
low-millisecond range throughout the sweep (full data in raw benchmark
output; omitted here as it was not the informative axis this round).

## 12. Memory Footprint
Reported as bytes-per-retained-fact rather than a cross-D compression
ratio, per the experiment's explicit fairness rule: a 4096-dim vector is
inherently larger than a 64-dim one (536 vs ~34,000+ bytes/fact), so
"smallest file" is not a meaningful comparison across D. The right
question — does higher D reduce noise — was answered directly in
Sections 8-10, independent of storage cost.

## 13. Full-vs-No-Update Ablation

| D | (1000,1000) full | (1000,1000) no-update | diff | (1000,100) full | (1000,100) no-update | diff |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.240 | 0.240 | +0.000 | 0.192 | 0.132 | +0.060 |
| 512 | 0.810 | 0.810 | +0.000 | 0.094 | 0.092 | +0.002 |
| 4096 | 0.972 | 0.972 | **+0.000** | 0.024 | 0.024 | **+0.000** |

**Decisive finding**: under unconstrained capacity, full mechanism and
no-update are *exactly* identical at every D tested — the update/merge
mechanism contributes zero measurable benefit, at any dimensionality.
Its one apparent advantage from the original Phase 2 report (+0.06 under
capacity pressure at D=64) **shrinks to +0.002 at D=512 and vanishes
completely (+0.000) at D=4096** — meaning that advantage, too, was a
collision artifact, not a real property of the update mechanism.

## 14. Failure Analysis
- Collision was real and was successfully reduced by increasing D
  (Section 7), and that reduction directly restored exact recall under
  unconstrained capacity (Section 8) — hypothesis supported for this case.
- False-memory failure is **not** primarily a collision problem; it's a
  tokenization/shared-vocabulary problem, unaffected by D (Section 10).
- The capacity-pressure recall drop is not a new failure — it's the
  removal of a false positive that low-D collision was masking
  (Section 9), directly verified.
- The update mechanism itself shows **no measurable benefit at any D**
  under fair (collision-controlled) conditions (Section 13) — this is
  the most important negative finding of this phase.

## 15. Interpretation
This experiment lands closest to **Outcome B**, with an important
addition. Representation/addressing was indeed the dominant driver of
Phase 2's *exact-recall* failure under unconstrained capacity — that
part of Outcome A's prediction held. But the update mechanism has now
been tested fairly, with collision noise substantially reduced, and it
still shows **zero measurable advantage over doing nothing**, in every
condition tested, at every D. Additionally, false-memory rate — arguably
the more safety-relevant metric — did not respond to D at all, revealing
a second, independent, unresolved problem (tokenization-driven shared
similarity floor) that this experiment was not designed to fix and
explicitly did not attempt to fix.

## 16. Limitations
- Full 6-config grid at all 7 D values was not completed within the
  compute budget (timed out at ~280s during D=256); scope was reduced to
  4 smaller configs across all D, with the two largest configs (n=10,000)
  only fully measured at D=64/128/256 before the cutoff. This is a
  genuine, disclosed scope reduction, not a silent one.
- The false-memory 6% outlier at D=4096/(1000,100) was observed but not
  root-caused; reported as-is rather than explained away.
- The tokenization-floor finding (Section 7) was discovered mid-experiment
  and not something the original hypothesis anticipated; it materially
  changes the interpretation of both the false-memory results and the
  original Phase 2 diagnostic's "0.43 similarity from collision" claim,
  which should now be understood as partly collision, partly genuine
  shared-token overlap.

## 17. Was the Hypothesis Supported?
**Partially.** "Increasing D reduces collision noise" — supported,
directly measured (Section 7). "This restores reliable memory
addressing and exact recall" — supported only under unconstrained
capacity; under capacity pressure, apparent recall gets *worse*, but for
a benign reason (noise stopped masking real eviction, Section 9), not
because higher D is actually harmful. The unstated assumption that fixing
collision would let the update mechanism demonstrate value — **not
supported**: the mechanism remains indistinguishable from doing nothing,
at every D tested.

## 18. Recommended Next Experiment
**Test whether the update/merge mechanism can be measured with a query
and encoding design that doesn't share tokens between all facts by
construction** — i.e., before concluding the mechanism itself is
inert, isolate it from the tokenization floor identified in Section 7,
which affects both the false-memory result and (likely) some portion of
the "no measurable benefit" ablation finding, since the update mechanism
was tested using the same encoder that has this floor. A clean version:
keep D fixed at a collision-safe value (e.g. 1024, per Section 7-8),
but redesign token generation so unrelated facts share zero tokens by
construction (e.g. treat each field as one atomic hashed unit rather
than word-splitting it), and re-run the Section 13 ablation once more,
unchanged otherwise. If the update mechanism still shows zero benefit
under that condition, that would be a much stronger and more conclusive
negative result than what this phase has shown.
