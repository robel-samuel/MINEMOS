import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "evaluation/resolution/v4_value_extraction/cases.jsonl"
MANIFEST = ROOT / "evaluation/resolution/v4_value_extraction/manifest.json"


def fact(pred, val, subject="speaker", polarity="positive", evidence=None):
    return {"predicate": pred, "value": val, "subject": subject,
            "polarity": polarity,
            "evidence": evidence if evidence is not None else val}


C = []


def case(cid, category, text, expect, reason=None, v3=False, gap=None,
         note=None):
    record = {"case_id": cid, "category": category, "text": text,
              "expect": expect}
    if reason:
        record["no_fact_reason"] = reason
    if v3:
        record["v3_shape"] = True
    if gap:
        record["known_gap"] = gap
    if note:
        record["note"] = note
    C.append(record)


# V3 over-capture shape 1: running speech continues with an interrogative.
case("V4-001", "trailing_interrogative",
     "i live in ethiopia are they offering a visa for a job",
     [fact("lives_in", "ethiopia")], v3=True)
case("V4-002", "trailing_interrogative",
     "i have 3k followers on youtube how much can i make per month",
     [fact("has", "3 k followers on youtube",
           evidence="3k followers on youtube")], v3=True)
case("V4-003", "trailing_interrogative",
     "i speak amharic what about you", [fact("speaks", "amharic")])
case("V4-004", "trailing_interrogative",
     "i work at acme do you know them", [fact("works_at", "acme")])

# V3 over-capture shape 2: a new first-person clause without punctuation.
case("V4-005", "trailing_new_clause",
     "my salary is crypto the i use p 2 p to change it to birr on binance",
     [fact("salary", "crypto")], v3=True)
case("V4-006", "trailing_new_clause",
     "i have 4 crates of mangoes i need 20 crates for the market i have 7 left",
     [fact("has", "4 crates of mangoes")], v3=True)
case("V4-007", "trailing_new_clause",
     "i work as a nurse i want to move to dubai next year",
     [fact("works_as", "nurse")])
case("V4-008", "trailing_new_clause",
     "i have two kids i need a bigger flat", [fact("has", "two kids")])

# V3 over-capture shape 3: a trailing language adverbial.
case("V4-009", "trailing_language_adverbial",
     "i have migrated to superbase, in english",
     [fact("has", "migrated to superbase")], v3=True)
case("V4-010", "trailing_language_adverbial",
     "i work at acme, in english", [fact("works_at", "acme")])
case("V4-011", "trailing_language_adverbial",
     "i study medicine, in english", [fact("studies", "medicine")])
case("V4-012", "trailing_language_adverbial",
     "i live in addis ababa, in amharic", [fact("lives_in", "addis ababa")])
case("V4-013", "trailing_language_adverbial",
     "my favorite language is english", [fact("favorite_language", "english")],
     note="A language value without the adverbial comma is untouched.")

# A relative clause belongs to the value; trimming it deletes asserted text.
case("V4-014", "relative_clause_guard",
     "my favorite book is the one i love most",
     [fact("favorite_book", "one i love most")])
case("V4-015", "relative_clause_guard",
     "i have a tattoo that says i love you",
     [fact("has", "tattoo that says i love you")])
case("V4-016", "relative_clause_guard",
     "i have a dog that lives with my parents",
     [fact("has", "dog that lives with my parents")])
case("V4-017", "relative_clause_guard",
     "i have a car that my father gave me",
     [fact("has", "car that my father gave me")])

# Several facts in one sentence, including a coordinated verb.
case("V4-018", "multiple_facts", "i live in berlin and work at acme",
     [fact("lives_in", "berlin"), fact("works_at", "acme")])
case("V4-019", "multiple_facts", "i study medicine and speak french",
     [fact("studies", "medicine"), fact("speaks", "french")])
case("V4-020", "multiple_facts", "i drink tea and i like coffee",
     [fact("drinks", "tea"), fact("likes", "coffee")])
case("V4-021", "multiple_facts", "i speak amharic and study medicine",
     [fact("speaks", "amharic"), fact("studies", "medicine")])
case("V4-022", "multiple_facts", "i work as a data analyst at acme",
     [fact("works_as", "data analyst"), fact("works_at", "acme")],
     note="One clause names both the role and the employer.")
case("V4-023", "multiple_facts", "i have a dog and a cat",
     [fact("has", "dog and a cat")],
     note="Coordination inside a noun phrase is one value, not two facts.")

# Corrections replace the rejected spelling with the supplied one.
case("V4-024", "correction", "my brother is called sam, i mean sam waters",
     [fact("name", "sam waters", subject="derived:brother")])
case("V4-025", "correction", "my name is adam, i mean adam smith",
     [fact("name", "adam smith")])

# Negation is polarity on the affirmative predicate.
case("V4-026", "negation", "i don't drink coffee",
     [fact("drinks", "coffee", polarity="negative")])
case("V4-027", "negation", "i don't eat meat",
     [fact("eats", "meat", polarity="negative")])
case("V4-028", "negation", "my brother doesn't live in oslo",
     [fact("lives_in", "oslo", subject="derived:brother",
           polarity="negative")],
     gap="A negated third-party clause is not matched by the relation rules; "
         "today the safe outcome is no fact, so this row is a measured miss.")
case("V4-029", "negation", "i don't own a car",
     [fact("has", "car", polarity="negative")],
     gap="Negated possession has no speaker rule; no fact is emitted today.")

# A possessive subject is the relative, never the speaker.
case("V4-030", "possessive", "my brother lives in oslo",
     [fact("lives_in", "oslo", subject="derived:brother")])
case("V4-031", "possessive", "my sister works at acme",
     [fact("works_at", "acme", subject="derived:sister")])
case("V4-032", "possessive", "my mother lives in paris",
     [fact("lives_in", "paris", subject="derived:mother")])

# Temporal modifiers stay out of the value.
case("V4-033", "temporal", "i live in berlin now",
     [fact("lives_in", "berlin")])
case("V4-034", "temporal", "i moved to lisbon in 2023",
     [fact("lives_in", "lisbon")])
case("V4-035", "temporal", "i used to live in rome",
     [fact("lives_in", "rome")],
     note="A past marker is recorded on the fact; the value is the place.")

# Organisation, place and role values keep their exact wording.
case("V4-036", "org_location_role",
     "i work at the addis ababa city administration",
     [fact("works_at", "addis ababa city administration")])
case("V4-037", "org_location_role", "i work as a senior nurse",
     [fact("works_as", "senior nurse")])
case("V4-038", "org_location_role", "our team uses postgres",
     [fact("uses", "postgres", subject="derived:team")])
case("V4-039", "org_location_role", "i work for the united nations",
     [fact("works_at", "united nations")])
case("V4-040", "org_location_role", "i live in addis abeba",
     [fact("lives_in", "addis abeba")],
     note="A misspelled value is kept as written, never repaired into a guess.")
case("V4-041", "org_location_role", "i live in nyc",
     [fact("lives_in", "New York City", evidence="nyc")],
     note="A documented alias is canonical; the original stays in evidence.")
case("V4-042", "org_location_role", "i live in rome",
     [fact("lives_in", "rome")],
     note="A place name is never rewritten by the typo corrector.")

# Hypothetical, interrogative, intended and opinion clauses assert nothing.
case("V4-043", "hypothetical", "if i moved to berlin i would need a visa",
     [], reason="hypothetical")
case("V4-044", "hypothetical", "maybe i live in paris", [],
     reason="hypothetical")
case("V4-045", "question", "do i live in berlin?", [], reason="question")
case("V4-046", "question", "what is my name?", [], reason="question")
case("V4-047", "intention", "i want to move to dubai", [], reason="intention")
case("V4-048", "intention", "i'm planning to study medicine", [],
     reason="intention")
case("V4-049", "opinion", "i think paris is beautiful", [], reason="opinion")
case("V4-050", "opinion", "i love coffee", [fact("likes", "coffee")],
     note="A stated preference is an asserted fact about the speaker, not a "
          "claim about the world.")

# Contrasts and coordination are scoped to their own fragment.
case("V4-051", "contrast", "i have a website, not her",
     [fact("has", "website")])
case("V4-052", "contrast", "i work as a broker, or i flip houses",
     [fact("works_as", "broker")])
case("V4-053", "contrast", "i live in paris, but i work in london",
     [fact("lives_in", "paris"), fact("works_in", "london")])

# Padding and evidence.
case("V4-054", "span_and_padding", "i have a silver flute",
     [fact("has", "silver flute")])
case("V4-055", "span_and_padding", "i live in berlin.",
     [fact("lives_in", "berlin")])
case("V4-056", "span_and_padding",
     "i have been asked this question before", [], reason="not_asserted",
     note="A present-perfect passive is an experience, not a possession.")

with OUT.open("w", encoding="utf-8", newline="\n") as fh:
    for record in C:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
MANIFEST.write_text(json.dumps({
    "dataset": "v4_value_extraction",
    "version": "1.0.0",
    "frozen": "2026-11-17",
    "cases": len(C),
    "sha256": digest,
    "language": "en",
    "provenance": "synthetic; authored for the V4 brief, no conversation text",
}, indent=2) + "\n", encoding="utf-8")
print(len(C), digest)
