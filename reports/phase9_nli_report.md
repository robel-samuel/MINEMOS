# Black Hole Memory — Phase 9: NLI/Entailment-Aware Representation Experiment

## 1. Research Question
Can a representation explicitly trained for textual entailment/
contradiction distinguish paraphrases, contradictions, and unrelated
facts well enough for Black Hole's existing consolidation mechanism to
safely use that signal?

## 2. Hypothesis
Not assumed true. Not tested. See Section 20.

## 3. Baseline
Reproduced before any change: **36/36 tests pass.** No code in
`phase2/` or `benchmarks/` was modified in this phase -- see Section 20
for why the experiment stopped before reaching that stage.

## 4. Model Provenance -- STOP CONDITION TRIGGERED

Per the task's explicit instruction ("If no legitimate NLI model can be
obtained or executed, STOP and report that honestly... Do not pretend
that a lexical approximation is an NLI model"), **this experiment
stopped at the model-acquisition stage.** Four genuine, documented
attempts were made, in order:

**Attempt 1 -- `transformers` + standard MNLI checkpoint (huggingface.co).**
Installed the `transformers` library successfully (from pypi.org, an
allowed domain). Attempted to load `typeform/distilbert-base-uncased-mnli`,
a real, well-known pretrained NLI model. Result: direct connection
failure -- `OSError: We couldn't connect to 'https://huggingface.co'`.
huggingface.co is not on this environment's network allowlist. This is
the near-universal distribution channel for modern pretrained NLI
models (RoBERTa-MNLI, BART-MNLI, DeBERTa-NLI, etc. are all hosted
there); this single block rules out essentially the entire standard
ecosystem of trained NLI classifiers.

**Attempt 2 -- `flair` library (alternative NLP model hub).**
Installation failed for an unrelated reason (disk space exhausted by
`transformers`'s dependency chain, specifically `torch` and related
packages) before the network question could even be tested. Not pursued
further after cleanup, given Attempt 1 already demonstrated the
underlying network constraint would likely recur.

**Attempt 3 -- GitHub search (api.github.com, an allowed domain) for an
NLI model distributed outside the huggingface.co hub**, on the
reasoning that Phase 8's spaCy model was successfully obtained this way
(as a GitHub release asset). Search returned generic source-code
repositories about NLI/entailment research, not verified downloadable
trained model checkpoints. Unlike Phase 8, where spaCy's project
specifically and deliberately distributes model weights via GitHub
releases, there is no equivalent standard practice for NLI models --
they are essentially universally hosted on huggingface.co. No candidate
was found that could be confidently verified as a genuine,
legitimately-obtained trained NLI classifier within reasonable search
effort.

**Attempt 4 -- direct pip package name guesses** (`nli-models`,
`snli-model`, `textentail`, `entail`). Three did not exist. The fourth
(`entail`, v0.0.4) installed, but inspection revealed it depends on the
`openai` package and is designed to call an external LLM API for
entailment judgments -- not a locally-executable pretrained classifier.
This is explicitly forbidden by the task ("Do NOT substitute: ...
LLM-generated labels"), and the package's own internal import
(`entail.nli.NliModel`) was additionally broken/missing regardless.
Uninstalled; not used for anything.

**No legitimate pretrained NLI/entailment model could be obtained or
executed in this environment.** The blocking constraint is structural
(network allowlist does not include huggingface.co or an equivalent
model hub), not a matter of trying harder or searching longer.

## 5. Model Architecture
N/A -- no model was obtained.

## 6-19. Dataset, Ground Truth, Decision Rule, Results, Ablation, Recall,
False-Memory, Latency, State Size, Drift, Error Analysis
**Not performed.** Per the task's explicit stop condition, the
experiment did not proceed past model acquisition. No dataset was
scored, no memory-level test was run, and no code beyond the four
installation attempts (all now uninstalled/cleaned up) was written.
This is a deliberate compliance with the instruction to stop rather
than substitute an unauthorized approximation, not an oversight.

## 20. Bugs Discovered and Corrections
None -- no experimental code was written in this phase. The one
"bug"-adjacent finding is that the `entail` PyPI package is itself
broken (references a missing internal module) independent of the
network/LLM-dependency issue that ruled it out anyway; noted for the
record, not something this project needed to fix since the package was
never adopted.

## 21. Limitations
- This is a **negative result specific to this environment's network
  configuration**, not a claim that NLI-aware representations are
  fundamentally unobtainable or would fail if tested. A different
  environment with access to huggingface.co, or a locally pre-staged
  NLI model file, could plausibly complete this experiment.
- The search for alternative distribution channels (Attempt 3) was not
  exhaustive -- a more extensive search might locate a legitimate
  GitHub-hosted NLI checkpoint that this attempt missed. Time/effort
  was bounded, and the underlying structural constraint (no HF hub
  access) makes success unlikely regardless of search depth.
- Because no model was tested, this phase provides zero evidence either
  for or against the hypothesis that entailment-aware representations
  would solve the paraphrase/contradiction separation problem observed
  in Phases 6-8.

## 22. Scientific Interpretation
No interpretation of representation quality is possible, because no
representation was tested. The only interpretable finding is
environmental: this project's tooling constraints prevent testing the
one representation category (genuine NLI/entailment models) that Phase
8's own conclusion identified as the most promising remaining untested
option.

## 23. Whether the Black Hole Hypothesis Gained or Lost Evidence
**Neither.** No evidence was gathered in either direction in this phase.
The hypothesis remains exactly where Phase 8 left it: three
representation types tested and failed (hash, lexical, static
pretrained embedding), the update mechanism itself never yet shown
broken when given any signal, and the specific representation class most
likely to help (entailment-trained models) still untested -- now
confirmed untestable in this particular environment, rather than merely
unattempted.

## Recommended Next Experiment (exactly one)
**Do not attempt Phase 9 again in this environment without first
resolving the underlying access constraint** -- e.g., by running the NLI
model evaluation in an environment with legitimate access to a model
hub (huggingface.co or equivalent), or by obtaining a specific,
verifiable pretrained NLI model file through a channel this
environment's network allowlist actually supports, established and
confirmed working *before* any dataset or memory-level work is
attempted, so that a repeat of this phase's stop condition doesn't
consume further effort without resolving the actual blocker.
