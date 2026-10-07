"""
Deterministic synthetic-conversation generator with hidden ground truth.

Each conversation mixes: a reinforced stable fact (stated several ways, to
test whether paraphrase-diverse statements consolidate into one memory
unit), several single-mention facts, a failure/lesson pair, a relationship,
and distractor sentences with no extractable content.

IMPORTANT LIMITATION (stated up front, not discovered later): the sentence
templates here and the regex patterns in ConversationExtractor are a
matched pair. This tests whether the *memory/consolidation layer* works
correctly given successful extraction -- it does not test freeform NLP
extraction quality. A production system would need a real extraction
model; that is out of scope for this phase (see Section 10 of the design
doc: "the first prototype should use existing models rather than
attempting to invent a completely new encoder").
"""

from __future__ import annotations

import random

LANGUAGES = ["Python", "Rust", "Go", "TypeScript", "Java", "Ruby"]
FRAMEWORKS = ["FastAPI", "Django", "Express", "Actix", "Spring"]
DATABASES = ["PostgreSQL", "MySQL", "MongoDB", "SQLite", "Redis"]
PLATFORMS = ["AWS", "GCP", "Azure", "Heroku", "Fly.io"]
FAILURE_LESSON_PAIRS = [
    ("a missing environment variable", "verify environment variables before deploying"),
    ("an expired API key", "rotate API keys automatically before they expire"),
    ("a database migration that was never run", "run pending migrations as part of deployment"),
    ("insufficient memory on the instance", "load-test before promoting to production"),
]

DISTRACTORS = [
    "It rained most of the afternoon.",
    "I had coffee before starting work today.",
    "The office was pretty quiet this week.",
    "I need to reply to a few emails later.",
    "Traffic was bad on the way in this morning.",
    "I'm thinking about repainting my kitchen.",
    "Lunch was decent today, nothing special.",
    "I watched a documentary over the weekend.",
]

# --- templates for the "backend_language" fact, deliberately paraphrased
# so several different sentences all describe the same underlying fact.
LANGUAGE_TEMPLATES = [
    "I use {lang} for backend development.",
    "{lang} is my preferred language for backend work.",
    "My API projects normally use {lang}.",
    "I built my latest backend using {lang}.",
    "Most of my backend code is written in {lang}.",
]


def generate_conversation(person_id: int, rng: random.Random,
                           reinforce_count: int = 3) -> tuple[str, dict, list[str]]:
    """
    Returns (full_conversation_text, ground_truth, sentence_list).
    ground_truth is hidden from the extractor -- used only for scoring.
    """
    subject = f"person_{person_id}"
    lang = rng.choice(LANGUAGES)
    framework = rng.choice(FRAMEWORKS)
    database = rng.choice(DATABASES)
    platform = rng.choice(PLATFORMS)
    failure, lesson = rng.choice(FAILURE_LESSON_PAIRS)

    sentences = []

    # Reinforced fact: same underlying fact, several paraphrasings.
    chosen_templates = rng.sample(LANGUAGE_TEMPLATES, k=min(reinforce_count, len(LANGUAGE_TEMPLATES)))
    for tmpl in chosen_templates:
        sentences.append(tmpl.format(lang=lang))

    # Single-mention facts.
    sentences.append(f"I'm using {framework} as the framework.")
    sentences.append(f"The project uses {database} for the database.")
    sentences.append(f"The application is deployed on {platform}.")

    # Relationship (project depends_on database).
    sentences.append(f"The backend depends on {database} being available.")

    # Failure + lesson.
    sentences.append(f"Deployment failed once because of {failure}.")
    sentences.append(f"I decided that deployments should always {lesson}.")

    # Distractors.
    n_distractors = rng.randint(1, 3)
    sentences.extend(rng.sample(DISTRACTORS, k=n_distractors))

    rng.shuffle(sentences)
    text = " ".join(sentences)

    ground_truth = {
        "subject": subject,
        "backend_language": lang,
        "framework": framework,
        "database": database,
        "deploy_platform": platform,
        "known_failure": failure,
        "lesson": lesson,
        "depends_on": database,
        "n_raw_statements": len(chosen_templates) + 6,  # facts, excludes distractors
        "n_distractors": n_distractors,
    }
    return text, ground_truth, sentences


def generate_dataset(n: int, seed: int = 7, reinforce_count: int = 3):
    rng = random.Random(seed)
    conversations = []
    for i in range(n):
        text, gt, sentences = generate_conversation(i, rng, reinforce_count)
        conversations.append({"text": text, "ground_truth": gt, "sentences": sentences})
    return conversations
