"""
Independent evaluation set for uncertainty / hedged-language predicate
classification (Step 2 of the extraction-layer experiment).

Every sentence here is newly written for this evaluation -- none are
reused from tests/test_semantic_encoder.py, benchmarks/conversation_generator.py's
templates, or benchmarks/semantic_encoder.py's exemplar bank. This is
deliberate: reusing exemplar-bank sentences would let the classifier
"succeed" by memorization rather than generalization.

Each case records ground truth across five independent dimensions so a
failure can be localized (Step 3): subject, predicate, object, confidence
band, and negated flag. expected_confidence_band is one of "high" (no
hedge/negation), "low" (uncertainty), or "rejected" (negation).
"""

EVAL_CASES = [
    # -- A. direct statements --------------------------------------------
    dict(text="I use Python for backend development.",
         subject="user", predicate="backend_language", object="Python",
         confidence_band="high", negated=False),
    dict(text="I prefer FastAPI.",
         subject="user", predicate="framework", object="FastAPI",
         confidence_band="high", negated=False),
    dict(text="Our database is PostgreSQL.",
         subject="user", predicate="database", object="PostgreSQL",
         confidence_band="high", negated=False),
    dict(text="The service runs on AWS.",
         subject="user", predicate="deploy_platform", object="AWS",
         confidence_band="high", negated=False),

    # -- B. uncertain statements ------------------------------------------
    dict(text="I might switch to Rust.",
         subject="user", predicate="prefers_language", object="Rust",
         confidence_band="low", negated=False),
    dict(text="I'm considering using Rust.",
         subject="user", predicate="prefers_language", object="Rust",
         confidence_band="low", negated=False),
    dict(text="I may use Rust for my backend.",
         subject="user", predicate="backend_language", object="Rust",
         confidence_band="low", negated=False),
    dict(text="I'm not sure whether I'll move to Rust.",
         subject="user", predicate="prefers_language", object="Rust",
         confidence_band="low", negated=False),
    dict(text="I could potentially switch to Rust.",
         subject="user", predicate="prefers_language", object="Rust",
         confidence_band="low", negated=False),
    dict(text="We might end up using MongoDB.",
         subject="user", predicate="database", object="MongoDB",
         confidence_band="low", negated=False),
    dict(text="I'm leaning towards Kotlin, but haven't decided.",
         subject="user", predicate="prefers_language", object="Kotlin",
         confidence_band="low", negated=False),
    dict(text="Possibly we'll deploy on GCP instead.",
         subject="user", predicate="deploy_platform", object="GCP",
         confidence_band="low", negated=False),

    # -- C. negative statements --------------------------------------------
    dict(text="I don't use Rust anymore.",
         subject="user", predicate="rejected_language", object="Rust",
         confidence_band="rejected", negated=True),
    dict(text="I no longer prefer Python.",
         subject="user", predicate="rejected_language", object="Python",
         confidence_band="rejected", negated=True),
    dict(text="I decided not to use MongoDB after all.",
         subject="user", predicate="rejected_language", object="MongoDB",
         confidence_band="rejected", negated=True),
    dict(text="We chose not to use Azure.",
         subject="user", predicate="rejected_language", object="Azure",
         confidence_band="rejected", negated=True),

    # -- D. historical statements --------------------------------------------
    dict(text="I used Python last year.",
         subject="user", predicate="historical_language_use", object="Python",
         confidence_band="high", negated=False),
    dict(text="I previously used AWS.",
         subject="user", predicate="deploy_platform", object="AWS",
         confidence_band="high", negated=False),
    dict(text="Java was used in an earlier version of the app.",
         subject="user", predicate="historical_language_use", object="Java",
         confidence_band="high", negated=False),

    # -- E. strong-confidence statements --------------------------------------
    dict(text="I definitely use Python.",
         subject="user", predicate="backend_language", object="Python",
         confidence_band="high", negated=False),
    dict(text="Python is my primary backend language.",
         subject="user", predicate="backend_language", object="Python",
         confidence_band="high", negated=False),
    dict(text="I'm certain we should use PostgreSQL.",
         subject="user", predicate="database", object="PostgreSQL",
         confidence_band="high", negated=False),

    # -- F. different predicate types --------------------------------------
    dict(text="I tested Kotlin on a small project.",
         subject="user", predicate="tested_language", object="Kotlin",
         confidence_band="high", negated=False),
    dict(text="I teach Ruby to junior developers.",
         subject="user", predicate="teaches_language", object="Ruby",
         confidence_band="high", negated=False),
    dict(text="I really dislike working with Java.",
         subject="user", predicate="dislikes_language", object="Java",
         confidence_band="high", negated=False),
    dict(text="The backend depends on Redis being available.",
         subject="user", predicate="depends_on", object="Redis",
         confidence_band="high", negated=False),
    dict(text="The outage happened because of a corrupted config file.",
         subject="user", predicate="known_failure", object="a corrupted config file",
         confidence_band="high", negated=False),
    dict(text="We should always run integration tests before release.",
         subject="user", predicate="lesson", object="run integration tests before release",
         confidence_band="high", negated=False),

    # -- G. different subjects --------------------------------------------
    dict(text="My colleague might switch to Go.",
         subject="colleague", predicate="prefers_language", object="Go",
         confidence_band="low", negated=False),
    dict(text="Our company is considering MySQL.",
         subject="company", predicate="database", object="MySQL",
         confidence_band="low", negated=False),
    dict(text="My coworker teaches TypeScript.",
         subject="coworker", predicate="teaches_language", object="TypeScript",
         confidence_band="high", negated=False),

    # -- H. unseen paraphrases (structurally different from anything above) --
    dict(text="There's a chance we'll end up adopting Go for this service.",
         subject="user", predicate="backend_language", object="Go",
         confidence_band="low", negated=False),
    dict(text="It's still up in the air whether Kotlin makes sense here.",
         subject="user", predicate="backend_language", object="Kotlin",
         confidence_band="low", negated=False),
]

assert len(EVAL_CASES) >= 30, f"only {len(EVAL_CASES)} cases -- need >= 30"
