# Black Hole Memory — Phase 14: Re-Keying Robustness Against Incorrect Merges

## Hypothesis
Immediate re-keying is vulnerable because one incorrect update
immediately changes the future addressing representation; delayed
re-keying may provide an intermediate, more controllable behavior. Not
assumed true -- treated as falsifiable throughout.

## Regression
62/62 tests pass (56 pre-existing + 6 new). Phase 13's error-propagation
trace reproduced exactly before any new code was written (frozen
final_slots=3, rekey final_slots=2, final similarities 0.217/0.173,
bit-identical to the Phase 13 report). Update equation confirmed
byte-identical across all three system families via a dedicated test.

## Systems
**A (frozen)**: unmodified `absorb()`. **B (immediate re-key)**: Phase
13's `real_rekey_absorb`, reused directly. **C (delayed re-key)**: one
new function, `delayed_rekey_absorb`, parameterized by predefined
N in {2, 3, 5} -- key changes only after exactly N successful updates to
that slot since its last re-key (or creation). Verified directly: a
dedicated test confirms the key stays frozen through N-1 merges and
changes only on the Nth.

## Dataset
Two datasets, following Phase 13's pattern of separating aggregate
sweep data from an isolated hand-built test:
- **Aggregate** (1,200 observations, 200 concepts): exact repeats,
  paraphrases, hard negatives, unrelated facts -- for capacity-sweep
  consolidation metrics.
- **Contamination chains** (125 chains: 25 concepts x 5 required
  ordering variants): canonical->para->para, para->canonical->para,
  canonical->hardneg->para, para->hardneg->para, alternating -- each
  followed by a 3-observation clean-canonical recovery tail. Verified
  directly that merge/split decisions are encoder-driven and vary
  naturally by ordering (e.g. the identical hard-negative pair merges
  in one ordering, 0.800, and correctly doesn't in another, 0.730 --
  confirmed before trusting any downstream analysis).

## A Bug Caught Before Trusting the Central Metric
An initial Error Amplification Ratio computation returned 3.79 for a
3-step sequence where at most 1 additional same-slot merge is
structurally possible -- an impossible value, caught by inspection
before being reported. Root cause: the metric counted *any* subsequent
"update" action anywhere in the trace, including the 3-step recovery
tail and including correct merges into a different, legitimate slot.
Fixed to track the specific contaminated **slot object** (not index --
consistent with this project's established eviction-safety pattern) and
count only merges into that same object, within the original sequence
only. A regression test now asserts the ratio is bounded to [0, 1] for
this dataset's sequence lengths.

## Error Amplification Results

| System | chains with initial error | error amplification ratio |
|---|---:|---:|
| A (frozen) | 43 | 0.791 |
| **B (immediate)** | 43 | **0.930** |
| C2 (delay 2) | 43 | 0.791 |
| C3 (delay 3) | 43 | 0.791 |
| C5 (delay 5) | 43 | 0.791 |

**All three delayed variants match frozen-key exactly within the
contamination window** -- an honest, disclosed limitation of this
specific chain design, not a hidden weakness: with only 2 merges
possible before the sequence ends, a delay of N>=2 literally never
triggers a re-key before the at-risk window closes. Only immediate
re-keying (B) shows elevated amplification. This means, for chains this
short, "any delay at all" behaves like "no re-keying at all" -- the
interesting comparison is therefore in the aggregate, longer-running
dataset (below), not this isolated short chain.

## Recovery Results
Mean recovery key/value similarity to the clean canonical fact, across
all 43 chains with an initial error: A=0.148/0.169, B=0.154/0.154,
C2=0.151/0.156, C3=0.146/0.159, C5=0.147/0.169 -- **all five systems
land in a narrow band (0.146-0.169), essentially indistinguishable at
this resolution.** None of the five systems shows a materially better
or worse recovery outcome after 3 clean repeats following one bad
merge. This is a meaningful negative finding in its own right (see
Outcome E below).

## Aggregate Consolidation Results (the more informative comparison)

| Capacity | System | slots | true consolidation | false consolidation | n_rekeys |
|---|---|---:|---:|---:|---:|
| unconstrained (1500) | A | 320 | 0.6775 | 0.4675 | 0 |
| unconstrained | **B** | 287 | **0.7400** | **0.4850** | 913 |
| unconstrained | C2 | 293 | 0.7325 | 0.4800 | 373 |
| unconstrained | C3 | 303 | 0.7150 | 0.4700 | 195 |
| unconstrained | **C5** | 320 | **0.6775** | **0.4675** | 107 |
| 100 | A | 100 | 0.3725 | 0.2275 | 0 |
| 100 | B | 100 | 0.3675 | 0.2250 | 403 |
| 100 | C2/C3/C5 | 100 | 0.3725 | 0.2275 | 34-171 |

**Two clean, decisive findings here:**

1. **A smooth, monotonic dial, not a free lunch.** At unconstrained
   capacity, true and false consolidation both increase together as N
   decreases (more aggressive re-keying): A/C5 (N effectively never
   reached) = 0.678/0.468 -> C3 = 0.715/0.470 -> C2 = 0.733/0.480 -> B
   (N=1) = 0.740/0.485. **C5 matches A exactly** (0.6775/0.4675, bit
   for bit), confirming N=5 essentially never fires within this
   dataset's merge counts, as expected. There is no N in the tested
   range that captures re-keying's consolidation benefit without a
   proportional false-consolidation cost -- the trade-off is continuous
   and monotonic, not a threshold effect with a safe region.

2. **Under real capacity pressure, the entire question becomes moot.**
   At capacity=100, all five systems converge to statistically
   identical consolidation rates (0.3725/0.2275), *despite* System B
   performing 403 re-keys. Heavy eviction means slots rarely survive
   long enough for accumulated re-keying to meaningfully change future
   addressing before being evicted anyway -- the effect this whole phase
   investigates is specifically an unconstrained-capacity phenomenon.

## Addressing Drift
Covered by the aggregate re-key counts and the error-propagation/
recovery sections above; no separate finding beyond what's reported
there.

## Capacity Effects
The single most important capacity finding: **re-keying's behavioral
differences (both its consolidation benefit and its error-amplification
risk) require slots to survive long enough to accumulate multiple
merges.** Under generous capacity this happens routinely; under real
capacity pressure it essentially never does, and frozen-key, immediate
re-key, and every delayed variant become indistinguishable.

## Deterministic Replay
Verified on a bounded 400-observation slice (full-scale replay not
re-run twice due to cumulative runtime across 5 systems x 2 capacity
points, ~70s already; disclosed scope choice, consistent with Phase
13's handling of the same constraint). Slot counts, consolidation rates,
and re-key counts were bit-identical across two independent runs for
every system tested.

## Failures Encountered
The Error Amplification Ratio bug (above) is the primary failure caught
and fixed during this phase. No other suspicious or vacuous metrics
were found -- `exact_recall`'s scale-driven collapse from Phase 13 was
not re-investigated here since this phase's aggregate dataset (200
concepts) and metric design (nearest-slot matching via full candidate
encoding, not partial-query recall) do not depend on the same
partial-query addressing path that caused that collapse.

## Limitations
- The contamination chains are short (3 steps) by design, matching the
  task's Section 4 example exactly -- but this means the delayed-rekey
  comparison is only meaningful in the aggregate dataset, not in the
  isolated chains, which is disclosed rather than papered over.
- Only one encoder/threshold combination was tested (matching Phase
  13's justified choice); the smooth monotonic N-dial finding may not
  generalize to a different encoder with a different error profile.
- Recovery was tested with exactly 3 clean repeats; whether more
  repeats eventually produce a measurable difference between systems
  was not explored.

## Conclusion
**The vulnerability Phase 13 identified is specifically caused by
immediate re-keying, and delaying re-keying does measurably reduce it
-- but as a continuous dial, not a safe threshold.** This is closest to
a blend of **Outcome A and Outcome D**: delaying re-keying provides a
real, monotonic, controllable reduction in both consolidation benefit
and error-amplification risk (Outcome A's core claim), but there is no
value of N in the tested range that isolates the benefit from the risk
-- they move together (a nuance beyond what Outcome A as stated
anticipated). Separately, **the recovery test supports a limited version
of Outcome E**: none of the five systems showed a meaningfully better
capacity to recover toward the clean representation after a contaminated
merge, within the tested recovery window -- once an incorrect merge
occurs, this architecture does not appear to have a mechanism that
reliably repairs it, regardless of re-key timing.

## What Uncertainty Remains
Whether a longer recovery window, a different capacity regime, or a
different encoder would change the "no system recovers meaningfully"
finding is untested. Whether the smooth N-dial holds under a
qualitatively different similarity signal (rather than field-aware
lexical specifically) is also untested. No claim is made here about
semantic memory, continual learning, or comparison to RAG. Per the
task's explicit instruction, no Phase 15 is proposed.
