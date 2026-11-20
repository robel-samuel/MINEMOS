# V4 exact fact-value extraction evaluation

A frozen, synthetic evaluation set for the fact values the semantic resolution
layer extracts from one message. It exists because the V3 real-conversation
labels showed the right subject and predicate attached to an **over-captured
value** in 6 of 13 gold facts, and because a value that was never written is a
wrong fact even when the rest of the tuple is right.

The set is public and contains no conversation text: every row was authored
for the V4 brief as a synthetic analogue of a recorded failure category.

## Frozen files

| File | Purpose |
| --- | --- |
| `cases.jsonl` | 56 cases. The ground truth is a careful reader's, not the resolver's output. |
| `manifest.json` | Case count and SHA-256 of `cases.jsonl`. The harness refuses to run if the two disagree. |
| `_build_cases.py` | Regenerates `cases.jsonl` and `manifest.json`. Editing a case means re-running this and accepting the new hash. |
| `measure_value_extraction.py` | The measurement harness. |
| `results/v4_value_extraction.json` | The last full run. |
| `results/v4_value_extraction_ablation.json` | The same set with the V4 trailing trims disabled (an ablation of the current source, not a V3 build). |

## Run

From the repository root:

```bash
.venv/Scripts/python.exe evaluation/resolution/v4_value_extraction/measure_value_extraction.py
.venv/Scripts/python.exe evaluation/resolution/v4_value_extraction/measure_value_extraction.py --ablate-v4-trims
```

The harness exits non-zero when any case fails.

## Case schema

```json
{
  "case_id": "V4-009",
  "category": "trailing_language_adverbial",
  "v3_shape": true,
  "text": "i have migrated to superbase, in english",
  "expect": [
    {"predicate": "has", "value": "migrated to superbase",
     "subject": "speaker", "polarity": "positive",
     "evidence": "migrated to superbase"}
  ]
}
```

* `expect: []` means the message asserts nothing; `no_fact_reason` records
  why (hypothetical, question, intention, opinion, not asserted).
* `subject` is `speaker`, `derived:<relation>` or `named:<label>`.
* `evidence` is the text that must appear in the original message span.
* `known_gap` marks a row whose ground truth the current implementation does
  not reach; it is still counted as a miss.

## Metrics, reported independently

| Metric | Question |
| --- | --- |
| recall | Of the facts a reader would record, how many were extracted with the right subject, predicate, polarity and value? |
| precision | Of the facts the resolver emitted, how many were right? |
| value_exactness | For matched facts, is the value string exactly the expected string? |
| evidence_exactness | For matched facts, is the evidence text exactly the expected source text? |
| evidence_grounded | Does the source slice at the recorded span equal the evidence text? |
| no_fact_compliance | Did messages that assert nothing emit zero facts? |

An over-captured value is counted twice: the emitted fact is a false positive
and the expected fact is a miss. That is deliberate.

## Current result (2026-11-17)

| Variant | Recall | Precision | Value exact | Evidence exact | No-fact compliance |
| --- | --- | --- | --- | --- | --- |
| V4 | 0.963 (52/54) | 1.0 (52/52) | 1.0 | 1.0 | 1.0 (8/8) |
| Ablation: V4 trailing trims disabled | 0.741 (40/54) | 0.769 (40/52) | 1.0 | 1.0 | 1.0 (8/8) |

Both misses are declared gaps in negation handling, not value errors:

* `V4-028` — `my brother doesn't live in oslo`: a negated third-party clause is
  not matched by the relation rules, so no fact is emitted. The safe outcome is
  abstention, but the negative fact is not recorded.
* `V4-029` — `i don't own a car`: negated possession has no speaker rule.

## What the set does not cover

* Non-English values. The real, synthetic and multilingual value sets are
  separate; this one is English only.
* Multi-message value correction across turns. This set resolves each case in
  its own conversation because it measures the value of a single message.
* Every organisation, place or role shape in the wild. It covers the failure
  categories recorded in V3, not a corpus.
