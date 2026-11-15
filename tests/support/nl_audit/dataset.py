"""Deterministic paired noisy-language dataset for the MINEMOS audit.

The dataset is generated from explicit templates and rule-based text
transformations.  No external language model, network call, or learned
paraphraser is used, so regenerating from the recorded seed reproduces every
case byte-for-byte (after JSON serialization).

Ground-truth discipline
-----------------------
``expected`` is defined by the case template before the engine runs.  The
engine never receives it: ingestion receives only ``observations`` (the clean
or noisy entity/attribute/value triple that an upstream extractor would have
produced), and querying receives only query text and a timestamp.  The
harness uses ``expected`` exclusively for scoring and failure analysis.

For unsupported queries ``expected`` is ``None``: the dataset asserts that no
stored fact answers the query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DATASET_VERSION = 1
DEFAULT_SEED = 20261105
CASES_PER_LEVEL = 120
LIFECYCLE_PER_CATEGORY = 10

LEVEL_CATEGORIES: Dict[str, List[Tuple[str, int]]] = {
    "A": [
        ("spelling_typo", 30),
        ("punctuation", 20),
        ("lowercase_spacing", 20),
        ("apostrophe_casing", 20),
        ("minor_mixed", 30),
    ],
    "B": [
        ("abbreviations", 25),
        ("grammar", 25),
        ("missing_word", 20),
        ("repeated_word", 20),
        ("informal_expression", 30),
    ],
    "C": [
        ("multiple_misspellings", 25),
        ("incomplete_query", 20),
        ("phonetic_stt", 25),
        ("disorganized_order", 20),
        ("heavy_mixed", 30),
    ],
    "D": [
        ("paraphrase", 30),
        ("indirect_reference", 25),
        ("context_followup", 20),
        ("synonym", 25),
        ("word_order", 20),
    ],
    "E": [
        ("near_miss_object", 25),
        ("near_miss_entity", 20),
        ("correction", 25),
        ("negation", 20),
        ("conflicting_numbers", 15),
        ("unsupported", 15),
    ],
}

NOISE_FAMILY = {
    "A": "surface",
    "B": "informal",
    "C": "corruption",
    "D": "semantic",
    "E": "ambiguity",
}

LIFECYCLE_CATEGORIES = [
    ("exact_restatement", 10),
    ("noisy_restatement", 10),
    ("paraphrase_restatement", 10),
    ("correction_exclusive", 10),
    ("correction_numeric", 10),
    ("negation_change", 10),
    ("historical_statement", 10),
    ("similar_wording_distinct", 10),
    ("near_miss_entity", 10),
    ("cumulative_distinct", 10),
    ("repeated_noisy", 10),
    ("contradiction_marker", 10),
]

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Avery", "Rowan", "Seraphina", "Thaddeus", "Valerie", "Zephyr",
    "Cassian", "Orion", "Lyra", "Eldridge", "Morrigan", "Persephone",
    "Barnaby", "Corinna", "Leopold", "Astrid", "Theron", "Ignatius",
    "Solene", "Amara", "Benedict", "Clara", "Darius", "Elowen",
    "Fenwick", "Greta", "Hollis", "Iris", "Jasper", "Keira",
    "Lucian", "Maren", "Nadia", "Oscar", "Priya", "Quentin",
    "Rosalind", "Silas", "Tamsin", "Ulric", "Vera", "Wren",
]

LAST_NAMES = [
    "Ashford", "Bellamy", "Carrow", "Dunmore", "Ellery", "Fairbanks",
    "Galloway", "Harlow", "Ingram", "Jarvis", "Kingsley", "Lockhart",
    "Merrivale", "Northcott", "Ormsby", "Pemberton", "Quill",
    "Ravenscroft", "Sutherland", "Thorne", "Underhill", "Vance",
    "Whitlock", "Yarrow", "Zeller", "Ainsley", "Blackwood", "Caldwell",
    "Davenport", "Everhart",
]

COLORS = [
    "teal", "crimson", "amber", "indigo", "olive", "maroon", "cobalt",
    "beige", "scarlet", "turquoise", "lavender", "charcoal", "emerald",
    "ochre", "ivory", "magenta", "navy", "peach", "plum", "rust",
    "sapphire", "silver", "vermilion", "violet", "wheat", "coral",
    "jade", "onyx", "pearl", "topaz",
]

CITIES = [
    "Lisbon", "Osaka", "Nairobi", "Toronto", "Reykjavik", "Bogota",
    "Hanoi", "Perth", "Wellington", "Marrakesh", "Helsinki", "Chennai",
    "Porto", "Tallinn", "Ljubljana", "Bratislava", "Sarajevo", "Tbilisi",
    "Yerevan", "Baku", "Almaty", "Tashkent", "Ulaanbaatar", "Chiang Mai",
    "Da Nang", "Cebu", "Hobart", "Dunedin", "Valparaiso", "Montevideo",
    "Asuncion", "La Paz", "Quito", "Medellin", "Cartagena", "Barranquilla",
    "Salvador", "Recife", "Fortaleza", "Belem",
]

COMPANIES = [
    "Northwind Labs", "Bluepeak Systems", "Cobalt Analytics",
    "Meridian Robotics", "Harborlight Media", "Ironwood Finance",
    "Larkspur Health", "Redwood Logistics", "Silverline Energy",
    "Thornfield Legal", "Umbra Software", "Verdant Foods",
    "Westgate Insurance", "Astra Biotech", "Brindle Construction",
    "Copperfield Retail", "Driftwood Studios", "Emberline Games",
    "Foxglove Publishing", "Granite Peak Mining", "Highwater Marine",
    "Juniper Systems", "Keystone Foundry", "Lantern Bay Hotels",
    "Maple Ridge Farms", "Nightjar Security", "Opaline Cosmetics",
    "Pinecrest Education", "Quarrystone Materials", "Riverbend Transit",
]

ROLES = [
    "data engineer", "product designer", "research scientist",
    "backend developer", "office manager", "financial analyst",
    "UX researcher", "site reliability engineer", "content strategist",
    "machine learning engineer", "laboratory technician",
    "supply chain analyst", "security consultant", "technical writer",
    "operations coordinator", "quality inspector", "graphic designer",
    "systems administrator", "clinical researcher",
    "environmental planner", "structural engineer", "policy analyst",
    "museum curator", "sound engineer", "urban planner",
    "data journalist", "game designer", "patent examiner",
    "logistics planner", "customer success lead",
]

BEVERAGES = [
    "green tea", "black coffee", "ginger ale", "oat milk",
    "sparkling water", "chai latte", "hot chocolate", "lemonade",
    "apple cider", "coconut water", "iced americano", "earl grey tea",
    "matcha latte", "root beer", "tonic water", "pomegranate juice",
    "barley tea", "soy milk", "peppermint tea", "cold brew",
    "mango lassi", "kombucha", "mineral water", "orange juice",
    "jasmine tea", "espresso", "horchata", "ginger tea",
    "cranberry juice", "rice milk",
]

FIELDS = [
    "marine biology", "quantum optics", "volcanic geology",
    "protein folding", "urban hydrology", "stellar evolution",
    "coral ecology", "robot swarms", "synthetic chemistry",
    "glacial dynamics", "neural circuits", "materials fatigue",
    "atmospheric chemistry", "plant genetics", "seismic tomography",
    "game theory", "cryptographic protocols", "linguistic typology",
    "archaeological dating", "fluid turbulence", "catalyst design",
    "immune signalling", "deep sea ecology", "asteroid dynamics",
    "soil microbiology", "power grid stability", "acoustic engineering",
    "paleoclimate modelling", "drug metabolism", "swarm intelligence",
]

INSTRUMENTS = [
    "cello", "banjo", "clarinet", "bassoon", "mandolin", "oboe",
    "trombone", "ukulele", "viola", "harp", "tuba", "accordion",
    "harmonica", "sitar", "lute", "xylophone", "double bass",
    "French horn", "electric guitar", "drum kit", "pan flute",
    "harpsichord", "marimba", "saxophone", "theremin",
]

ALLERGENS = [
    "penicillin", "peanuts", "shellfish", "pollen", "latex", "soy",
    "sesame", "eggs", "walnuts", "almonds", "strawberries", "aspirin",
    "sulfa drugs", "dust mites", "bee stings", "gluten", "lactose",
    "mustard", "celery", "nickel",
]

POSSESSIONS = [
    "a red kayak", "a vintage motorcycle", "a wooden canoe",
    "a brass telescope", "a mechanical watch", "a spinning wheel",
    "a ceramic kiln", "a vintage typewriter", "a silver flute",
    "a folding bicycle", "a hand loom", "a model railway",
    "a cast iron skillet", "a sailing dinghy", "a film projector",
    "a rock tumbler", "a sewing machine", "a telescope mount",
    "a woodworking lathe", "a camping trailer", "a rowing machine",
    "a chess set", "a record player", "a drone", "a leather satchel",
    "a birdhouse", "a pottery wheel", "a metal detector",
    "a telescope tripod", "a vintage radio",
]

LANGUAGES = [
    "Mandarin", "Portuguese", "Swahili", "Icelandic", "Korean",
    "Turkish", "Polish", "Dutch", "Hindi", "Arabic", "Greek",
    "Hebrew", "Vietnamese", "Thai", "Finnish", "Norwegian", "Czech",
    "Romanian", "Hungarian", "Bengali",
]

GENRES = [
    "historical fiction", "science fiction", "detective novels", "poetry",
    "biographies", "graphic novels", "travel writing", "philosophy",
    "true crime", "fantasy", "literary fiction", "essays", "horror",
    "romance", "nature writing", "maritime history", "economics",
    "art criticism", "memoir", "short stories",
]

TEACHING_SUBJECTS = [
    "algebra", "world history", "chemistry", "creative writing",
    "music theory", "computer science", "geography", "physics",
    "art history", "philosophy", "statistics", "biology", "economics",
    "literature", "civics", "calculus", "astronomy",
    "environmental science", "psychology", "linguistics", "drama",
    "photography", "anatomy", "robotics", "ethics",
]

SALARIES = [str(value) for value in range(62000, 166000, 3500)]
RENTS = [str(value) for value in range(980, 3020, 70)]
AGES = [str(value) for value in (24, 27, 31, 34, 38, 41, 45, 48, 52, 55,
                                58, 61, 64, 67, 70, 73, 76, 79, 82, 85,
                                28, 33, 36, 39, 43, 47, 50, 53, 57, 60)]

AUTHORS = [
    "Ursula Le Guin", "Jorge Luis Borges", "Toni Morrison",
    "Italo Calvino", "Chimamanda Ngozi Adichie", "Haruki Murakami",
    "Octavia Butler", "Gabriel Garcia Marquez", "Virginia Woolf",
    "Salman Rushdie", "Kazuo Ishiguro", "Margaret Atwood", "Ted Chiang",
    "N. K. Jemisin", "Arundhati Roy", "Elena Ferrante", "Ismail Kadare",
    "Ngugi wa Thiong'o", "Clarice Lispector", "Wole Soyinka",
    "Han Kang", "Olga Tokarczuk", "Javier Marias", "Louise Erdrich",
    "Colson Whitehead",
]

FILMS = [
    "The Quiet Harbor", "A Map of Small Rooms", "The Longest Winter",
    "Paper Moonlight", "The Glass Orchard", "Midnight in Trieste",
    "The Salt Road", "A Theory of Everything Else",
    "The Last Cartographer", "Blue Hour", "The Borrowed Garden",
    "Static and Snow", "The Northern Line", "A House of Leaves",
    "The Second Sun", "Pale Fireflies", "The Understudy",
    "Seven Kinds of Silence", "The Amber Coast", "A Room in Winter",
    "The Echo Chamber", "Small Gods of the Kitchen",
    "The Unfinished Bridge", "Night Train to Ostrava",
    "The Weight of Water",
]

COMMUTE_MODES = [
    "bicycle", "tram", "subway", "electric scooter", "ferry",
    "commuter train", "motorbike", "carpool", "on foot", "bus",
    "light rail", "kayak", "skateboard", "horse", "company shuttle",
]

ORGANIZATIONS = [
    "the Harbor Food Bank", "the Riverside Shelter",
    "the City Literacy Project", "the Coastal Cleanup Trust",
    "the Community Garden Network", "the Winter Warmth Fund",
    "the Migrant Support Centre", "the Public Library Friends",
    "the Youth Music Collective", "the Urban Tree Initiative",
    "the Senior Companion Service", "the Wildlife Rescue League",
    "the Free Clinic", "the Housing Advice Bureau",
    "the Refugee Language School", "the Repair Cafe",
    "the Night Kitchen", "the Toy Library", "the Cycling Alliance",
    "the Wetland Trust",
]

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "does", "do", "did",
    "what", "which", "where", "who", "whom", "whose", "how", "of", "in",
    "at", "to", "for", "by", "and", "or", "on", "with", "their", "his",
    "her", "they", "he", "she", "it", "this", "that", "about", "most",
    "best", "usually", "can", "you", "tell", "me", "i", "my", "we",
    "our", "from", "as", "has", "have", "had", "been", "being", "into",
}

ABBREVIATIONS = [
    ("because", "bc"), ("with", "w/"), ("without", "w/o"),
    ("tomorrow", "tmrw"), ("about", "abt"), ("please", "pls"),
    ("before", "b4"), ("tonight", "2nite"), ("going to", "gonna"),
    ("want to", "wanna"), ("kind of", "kinda"), ("a lot", "alot"),
    ("though", "tho"), ("through", "thru"), ("probably", "prob"),
    ("definitely", "def"), ("favorite", "fav"), ("favourite", "fav"),
    ("information", "info"), ("appointment", "appt"),
    ("available", "avail"), ("really", "rly"),
    # Texting shortenings of common predicate verbs.  These are informal
    # abbreviations, not historical engine aliases; MINEMOS should not be
    # assumed to expand them without evidence.
    ("drinks", "drnks"), ("studies", "studys"), ("speaks", "spks"),
    ("reads", "rds"), ("works", "wrks"), ("teaches", "tchs"),
    ("plays", "plys"), ("lives in", "livs in"), ("commutes by", "commutz by"),
]

PHONETIC = {
    "night": "nite", "light": "lite", "write": "rite", "right": "rite",
    "their": "there", "there": "their", "to": "too", "too": "to",
    "two": "too", "for": "four", "four": "for", "know": "no",
    "no": "know", "new": "knew", "knew": "new", "sea": "see",
    "see": "sea", "tea": "tee", "be": "bee", "meet": "meat",
    "meat": "meet", "week": "weak", "weak": "week",
    "coffee": "cawfee", "water": "wadder", "city": "siddy",
    "little": "liddle", "probably": "probly", "going": "goin",
    "something": "somethin", "because": "becuz", "science": "sighnce",
    "biology": "byology", "color": "culler", "favorite": "fayvrit",
    "lisbon": "lizbon", "toronto": "tronno", "osaka": "ohsaka",
    "nairobi": "nairobee", "ashford": "ashferd", "bellamy": "bellamee",
    "dunmore": "dunmor", "harlow": "harlo", "ingram": "ingrum",
    "jarvis": "jarves", "kingsley": "kingslee", "lockhart": "lockheart",
    "merrivale": "merryvale", "northcott": "northcutt",
    "ormsby": "ormsbee", "quill": "quil", "sutherland": "southerland",
    "thorne": "thorn", "vance": "vanse", "whitlock": "whitlok",
    "yarrow": "yaro", "zeller": "zellar",
}

VERB_AGREEMENT = {
    "drinks": "drink", "studies": "study", "plays": "play",
    "lives": "live", "works": "work", "teaches": "teach",
    "reads": "read", "owns": "own", "speaks": "speak",
    "commutes": "commute", "volunteers": "volunteer", "loves": "love",
    "likes": "like", "prefers": "prefer", "earns": "earn",
    "pays": "pay", "does": "do", "is": "are", "was": "were",
}

INFORMAL_QUERY = {
    "favorite_color": "what's {name} really into?",
    "lives_in": "where's {name} based?",
    "works_at": "where does {name} work at?",
    "role": "what does {name} do for work?",
    "drinks": "what does {name} drink nowadays?",
    "studies": "what's {name} studying?",
    "plays": "what's {name} learning?",
    "allergic_to": "what does {name} have a problem with?",
    "owns": "what has {name} got?",
    "born_in": "where's {name} from?",
    "speaks": "what languages does {name} know?",
    "reads": "what's {name} into reading?",
    "teaches": "what does {name} teach?",
    "salary": "how much does {name} earn?",
    "rent": "how much rent does {name} pay?",
    "favorite_author": "who's {name}'s fav author?",
    "favorite_film": "what's {name}'s fav film?",
    "commutes_by": "how does {name} get around?",
    "volunteers_at": "where does {name} help out?",
    "age": "how old is {name}?",
}

INFORMAL_PREDICATE = {
    "favorite_color": "is really into",
    "lives_in": "is based in",
    "works_at": "works over at",
    "role": "does work as",
    "drinks": "is really into",
    "studies": "is studying",
    "plays": "is learning",
    "allergic_to": "has a problem with",
    "owns": "has got",
    "born_in": "comes from",
    "speaks": "knows",
    "reads": "is into reading",
    "teaches": "teaches",
    "salary": "earns",
    "rent": "pays",
    "favorite_author": "really likes the author",
    "favorite_film": "really likes the film",
    "commutes_by": "gets around by",
    "volunteers_at": "helps out at",
    "age": "is",
}

SYNONYM_PREDICATE = {
    "favorite_color": "favourite colour is",
    "lives_in": "resides in",
    "works_at": "is employed by",
    "role": "job title is",
    "drinks": "prefers to drink",
    "studies": "is researching",
    "plays": "plays the",
    "allergic_to": "has an allergy to",
    "owns": "has",
    "born_in": "originates from",
    "speaks": "is fluent in",
    "reads": "enjoys reading",
    "teaches": "instructs in",
    "salary": "earns",
    "rent": "pays rent of",
    "favorite_author": "preferred author is",
    "favorite_film": "favourite movie is",
    "commutes_by": "travels by",
    "volunteers_at": "volunteers with",
    "age": "is aged",
}


def _family(
    family_id: str,
    predicate: str,
    objects: Sequence[str],
    queries: Sequence[str],
    attribute: str,
) -> Dict[str, Any]:
    return {
        "id": family_id,
        "predicate": predicate,
        "objects": list(objects),
        "queries": list(queries),
        "attribute": attribute,
        "informal_predicate": INFORMAL_PREDICATE[family_id],
        "informal_query": INFORMAL_QUERY[family_id],
        "synonym_predicate": SYNONYM_PREDICATE[family_id],
    }


FACT_FAMILIES = [
    _family("favorite_color", "favorite color is", COLORS,
            ["What is {name}'s favorite color?",
             "Which color does {name} like best?"], "favorite color"),
    _family("lives_in", "lives in", CITIES,
            ["Where does {name} live?",
             "Which city does {name} live in?"], "current city"),
    _family("works_at", "works at", COMPANIES,
            ["Where does {name} work?",
             "Which company does {name} work for?"], "employer"),
    _family("role", "role is", ROLES,
            ["What is {name}'s role?",
             "What does {name} do for work?"], "job role"),
    _family("drinks", "drinks", BEVERAGES,
            ["What does {name} drink?",
             "Which drink does {name} usually have?"], "preferred drink"),
    _family("studies", "studies", FIELDS,
            ["What does {name} study?",
             "Which field does {name} study?"], "field of study"),
    _family("plays", "plays", INSTRUMENTS,
            ["What instrument does {name} play?",
             "Which instrument does {name} play?"], "instrument"),
    _family("allergic_to", "is allergic to", ALLERGENS,
            ["What is {name} allergic to?",
             "Which allergen affects {name}?"], "allergy"),
    _family("owns", "owns", POSSESSIONS,
            ["What does {name} own?",
             "What possession does {name} have?"], "main possession"),
    _family("born_in", "was born in", CITIES,
            ["Where was {name} born?",
             "Which city was {name} born in?"], "birth city"),
    _family("speaks", "speaks", LANGUAGES,
            ["What language does {name} speak?",
             "Which language does {name} speak?"], "spoken language"),
    _family("reads", "reads", GENRES,
            ["What does {name} read?",
             "Which genre does {name} read?"], "reading genre"),
    _family("teaches", "teaches", TEACHING_SUBJECTS,
            ["What does {name} teach?",
             "Which subject does {name} teach?"], "subject taught"),
    _family("salary", "salary is", SALARIES,
            ["What is {name}'s salary?",
             "How much does {name} earn?"], "salary"),
    _family("rent", "rent is", RENTS,
            ["What is {name}'s rent?",
             "How much rent does {name} pay?"], "monthly rent"),
    _family("favorite_author", "favorite author is", AUTHORS,
            ["Who is {name}'s favorite author?",
             "Which author does {name} like best?"], "favorite author"),
    _family("favorite_film", "favorite film is", FILMS,
            ["What is {name}'s favorite film?",
             "Which film does {name} like best?"], "favorite film"),
    _family("commutes_by", "commutes by", COMMUTE_MODES,
            ["How does {name} commute?",
             "What does {name} use to commute?"], "commute mode"),
    _family("volunteers_at", "volunteers at", ORGANIZATIONS,
            ["Where does {name} volunteer?",
             "Which organization does {name} volunteer at?"],
            "volunteer organization"),
    _family("age", "is aged", AGES,
            ["How old is {name}?",
             "What is {name}'s age?"], "age"),
]

FAMILY_BY_ID = {family["id"]: family for family in FACT_FAMILIES}
EXCLUSIVE_FAMILIES = ["lives_in", "works_at", "role"]
NUMERIC_FAMILIES = ["salary", "rent", "age"]
D_ANCHOR_FAMILY = FAMILY_BY_ID["studies"]

NEAR_MISS_OBJECT_PAIRS = [
    ("The Quiet Harbor", "The Silent Harbor", "substitution"),
    ("The Quiet Harbor", "The Quiet Harbour", "insertion"),
    ("The Glass Orchard", "The Glass Orphan", "substitution"),
    ("The Salt Road", "The Salt Roads", "insertion"),
    ("Ursula Le Guin", "Ursula Le Guen", "substitution"),
    ("Ursula Le Guin", "Ursula Le Guinn", "insertion"),
    ("Toni Morrison", "Toni Morison", "deletion"),
    ("Italo Calvino", "Italo Calvin", "deletion"),
    ("Haruki Murakami", "Haruki Murakam", "deletion"),
    ("Margaret Atwood", "Margret Atwood", "substitution"),
]

NEAR_MISS_ENTITY_PAIRS = [
    ("Mira Ashford", "Mira Ahsford", "transposition"),
    ("Rowan Bellamy", "Rowan Bellany", "substitution"),
    ("Seraphina Carrow", "Seraphina Carow", "deletion"),
    ("Thaddeus Dunmore", "Thaddeus Dunmoire", "insertion"),
    ("Valerie Ellery", "Valerie Ellary", "substitution"),
    ("Zephyr Fairbanks", "Zephyr Fairbank", "deletion"),
    ("Cassian Galloway", "Cassian Gallowai", "substitution"),
    ("Orion Harlow", "Orion Harlo", "deletion"),
    ("Lyra Ingram", "Lyra Ingrum", "substitution"),
    ("Eldridge Jarvis", "Eldridge Jarviss", "insertion"),
]

SIMILAR_WORDING_PAIRS = [
    ("the Roman Republic", "the Roman Empire", "distinct"),
    ("the Northern Line", "the Southern Line", "distinct"),
    ("Northwind Labs", "Northwind Lab", "typo_equivalent"),
    ("Bluepeak Systems", "Bluepeak System", "typo_equivalent"),
    ("the Community Garden Network", "the Community Gardens Network",
     "distinct"),
    ("the City Literacy Project", "the City Literature Project",
     "distinct"),
    ("a vintage motorcycle", "a vintage motorscooter", "insertion"),
    ("a brass telescope", "a brass telescopes", "insertion"),
    ("the Coastal Cleanup Trust", "the Coastal Cleanup Group", "distinct"),
    ("the Public Library Friends", "the Public Library Friend",
     "typo_equivalent"),
]

UNSUPPORTED_ATTRIBUTE = {
    "lives_in": ("salary", "What is {name}'s salary?"),
    "works_at": ("rent", "What is {name}'s rent?"),
    "role": ("age", "How old is {name}?"),
    "studies": ("favorite film", "What is {name}'s favorite film?"),
    "drinks": ("favorite author", "Who is {name}'s favorite author?"),
    "speaks": ("commute mode", "How does {name} commute?"),
    "owns": ("allergy", "What is {name} allergic to?"),
    "born_in": ("salary", "What is {name}'s salary?"),
}

_FALLBACK_SUBJECTS = [
    "Zara Voss", "Ilya Marchetti", "Noor Haddad", "Pavel Orlov",
    "Mei Lin", "Tomas Reyes", "Anika Sharma", "Dmitri Volkov",
]

# ---------------------------------------------------------------------------
# Deterministic randomness and text transforms
# ---------------------------------------------------------------------------

_TOKEN_CORE = re.compile(r"^([^A-Za-z]*)([A-Za-z]+)(.*)$")


def _rng(seed: int, *parts: Any) -> random.Random:
    payload = "|".join([str(seed)] + [str(part) for part in parts])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _split_core(token: str) -> Tuple[str, str, str]:
    match = _TOKEN_CORE.match(token)
    if not match:
        return token, "", ""
    return match.group(1), match.group(2), match.group(3)


def _join_core(prefix: str, core: str, suffix: str) -> str:
    return f"{prefix}{core}{suffix}"


def _typo_core(core: str, rng: random.Random, kind: str) -> str:
    if len(core) < 3:
        return core
    letters = list(core)
    if kind == "transposition":
        indices = [i for i in range(1, len(letters) - 1)
                   if letters[i].isalpha() and letters[i + 1].isalpha()]
        if not indices:
            return core
        index = rng.choice(indices)
        letters[index], letters[index + 1] = (
            letters[index + 1], letters[index])
    elif kind == "deletion":
        indices = [i for i in range(1, len(letters)) if letters[i].isalpha()]
        if not indices:
            return core
        del letters[rng.choice(indices)]
    elif kind == "insertion":
        index = rng.randrange(1, len(letters))
        letters.insert(index, rng.choice("abcdefghijklmnopqrstuvwxyz"))
    elif kind == "substitution":
        indices = [i for i in range(1, len(letters)) if letters[i].isalpha()]
        if not indices:
            return core
        index = rng.choice(indices)
        replacement = rng.choice(
            [c for c in "abcdefghijklmnopqrstuvwxyz"
             if c != letters[index].lower()])
        letters[index] = replacement
    else:
        raise ValueError(f"unknown typo kind: {kind}")
    result = "".join(letters)
    if core[0].isupper() and result:
        result = result[0].upper() + result[1:]
    return result


def _typo_candidates(text: str) -> List[int]:
    words = text.split(" ")
    candidates = []
    for index, word in enumerate(words):
        _prefix, core, _suffix = _split_core(word)
        if len(core) >= 4 and core.lower() not in STOPWORDS:
            candidates.append(index)
    return candidates


def _apply_typos(
    text: str,
    rng: random.Random,
    count: int,
    kinds: Sequence[str],
    protected: Optional[Iterable[str]] = None,
) -> Tuple[str, List[str]]:
    protected_words = {word.lower() for word in (protected or ())}
    words = text.split(" ")
    used: set[int] = set()
    ops: List[str] = []
    for _ in range(count):
        candidates = [
            index for index in _typo_candidates(text)
            if index not in used
            and _split_core(words[index])[1].lower() not in protected_words
        ]
        if not candidates:
            break
        index = rng.choice(candidates)
        prefix, core, suffix = _split_core(words[index])
        kind = rng.choice(list(kinds))
        mutated = _typo_core(core, rng, kind)
        if mutated == core:
            continue
        words[index] = _join_core(prefix, mutated, suffix)
        used.add(index)
        ops.append(f"typo_{kind}:{core}->{mutated}")
    return " ".join(words), ops


def _abbreviate_text(
    text: str, rng: random.Random, count: int = 2
) -> Tuple[str, List[str]]:
    # Only abbreviate phrases that actually occur in the field; choosing
    # blindly from the full table made most cases a no-op.
    matches = []
    for phrase, abbreviation in ABBREVIATIONS:
        pattern = re.compile(
            r"(?i)(?<![A-Za-z])" + re.escape(phrase) + r"(?![A-Za-z])")
        if pattern.search(text):
            matches.append((phrase, abbreviation, pattern))
    if not matches:
        return text, []
    rng.shuffle(matches)
    result = text
    ops: List[str] = []
    for phrase, abbreviation, pattern in matches[:count]:
        result = pattern.sub(abbreviation, result, count=1)
        ops.append(f"abbrev:{phrase}->{abbreviation}")
    return result, ops


def _drop_articles(text: str, rng: random.Random) -> Tuple[str, List[str]]:
    matches = list(re.finditer(r"(?i)(?<![A-Za-z])(the|a|an)\s+", text))
    if not matches:
        return text, []
    match = rng.choice(matches)
    result = text[:match.start()] + text[match.end():]
    return result, [f"drop_article:{match.group(1)}"]


def _agreement(text: str, rng: random.Random) -> Tuple[str, List[str]]:
    verbs = list(VERB_AGREEMENT)
    rng.shuffle(verbs)
    for verb in verbs:
        pattern = re.compile(rf"(?i)(?<![A-Za-z]){re.escape(verb)}(?![A-Za-z])")
        if pattern.search(text):
            replacement = VERB_AGREEMENT[verb]
            return pattern.sub(replacement, text, count=1), [
                f"agreement:{verb}->{replacement}"]
    return text, []


def _drop_function_word(
    text: str, rng: random.Random
) -> Tuple[str, List[str]]:
    words = text.split(" ")
    candidates = [
        index for index, word in enumerate(words)
        if index > 0 and word.lower().strip(".,;:!?") in {
            "is", "in", "at", "to", "the", "a", "an", "of", "does",
            "do", "did", "was", "are",
        }
    ]
    if len(words) <= 3 or not candidates:
        return text, []
    index = rng.choice(candidates)
    removed = words.pop(index)
    return " ".join(words), [f"drop_word:{removed}"]


def _repeat_word(text: str, rng: random.Random) -> Tuple[str, List[str]]:
    words = text.split(" ")
    candidates = [
        index for index, word in enumerate(words)
        if len(_split_core(word)[1]) >= 4
        and _split_core(word)[1].lower() not in STOPWORDS
    ]
    if not candidates:
        return text, []
    index = rng.choice(candidates)
    words.insert(index + 1, words[index])
    return " ".join(words), [f"repeat:{words[index]}"]


def _remove_apostrophes(text: str) -> Tuple[str, List[str]]:
    result = text.replace("'s", "s").replace("'", "")
    return result, (["apostrophe_removed"] if result != text else [])


def _extra_spaces(text: str, rng: random.Random) -> Tuple[str, List[str]]:
    result = re.sub(r" ", "  ", text.strip())
    if rng.random() < 0.5:
        result = " " + result
    if rng.random() < 0.5:
        result = result + " "
    return result, ["extra_spaces"]


def _random_caps(text: str, rng: random.Random) -> Tuple[str, List[str]]:
    words = text.split(" ")
    candidates = [index for index, word in enumerate(words)
                  if word and word[0].islower()]
    if not candidates:
        return text, []
    index = rng.choice(candidates)
    if rng.random() < 0.5:
        words[index] = words[index][0].upper() + words[index][1:]
    else:
        words[index] = words[index].upper()
    return " ".join(words), [f"caps:{words[index]}"]


def _phonetic_text(
    text: str, rng: random.Random
) -> Tuple[str, List[str]]:
    words = text.split(" ")
    ops: List[str] = []
    for index, word in enumerate(words):
        prefix, core, suffix = _split_core(word)
        lowered = core.lower()
        if lowered in PHONETIC:
            replacement = PHONETIC[lowered]
            if core[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            words[index] = _join_core(prefix, replacement, suffix)
            ops.append(f"phonetic:{core}->{replacement}")
    if not ops:
        candidates = _typo_candidates(text)
        if candidates:
            index = rng.choice(candidates)
            prefix, core, suffix = _split_core(words[index])
            replacement = core.lower()
            for source, target in (("ph", "f"), ("ck", "k"), ("qu", "kw"),
                                   ("ing", "in")):
                if source in replacement:
                    replacement = replacement.replace(source, target)
                    break
            if replacement != core.lower():
                if core[0].isupper():
                    replacement = replacement[0].upper() + replacement[1:]
                words[index] = _join_core(prefix, replacement, suffix)
                ops.append(f"phonetic_rule:{core}->{replacement}")
    return " ".join(words), ops


def _shuffle_non_proper(
    text: str, rng: random.Random
) -> Tuple[str, List[str]]:
    words = text.split(" ")
    fixed = {
        index for index, word in enumerate(words)
        if word and word[0].isupper()
    }
    movable = [index for index in range(len(words)) if index not in fixed]
    if len(movable) < 3:
        return text, []
    values = [words[index] for index in movable]
    rng.shuffle(values)
    for index, value in zip(movable, values):
        words[index] = value
    result = " ".join(words)
    return result, (["disorganized_order"] if result != text else [])


def _shuffle_words(text: str, rng: random.Random) -> Tuple[str, List[str]]:
    words = text.split(" ")
    if len(words) < 2:
        return text, []
    rng.shuffle(words)
    result = " ".join(words)
    return result, (["object_word_order"] if result != text else [])


def _truncate_query(
    text: str, rng: random.Random
) -> Tuple[str, List[str]]:
    words = text.split(" ")
    if len(words) <= 3:
        return text, []
    if rng.random() < 0.5:
        words = words[:-1]
        op = "truncate_last_word"
    else:
        words = words[:max(2, len(words) // 2)]
        op = "truncate_half"
    result = " ".join(words)
    if not result.endswith(("?", ".")):
        result = result.rstrip(".,;:!?")
    return result, [op]


def _lowercase(text: str) -> Tuple[str, List[str]]:
    result = text.lower()
    return result, (["lowercase"] if result != text else [])


def _depunct(text: str) -> Tuple[str, List[str]]:
    result = text.replace(",", "").replace(";", "").replace(":", "")
    result = result.replace(".", "").replace("!", "").replace("?", "")
    result = result.strip()
    return result, (["punctuation_removed"] if result != text else [])


def _compose_sentence(
    subject: str,
    predicate: str,
    obj: str,
    level: str,
    category: str,
    rng: random.Random,
    prefix: str = "",
    suffix: str = "",
) -> Tuple[str, List[str]]:
    text = f"{subject} {predicate} {obj}."
    ops: List[str] = []
    if level in ("A", "B", "C"):
        if category in ("punctuation", "minor_mixed", "heavy_mixed"):
            text, new_ops = _depunct(text)
            ops.extend(new_ops)
        if category in ("lowercase_spacing", "minor_mixed", "heavy_mixed"):
            text, new_ops = _lowercase(text)
            ops.extend(new_ops)
        if category == "lowercase_spacing":
            text, new_ops = _extra_spaces(text, rng)
            ops.extend(new_ops)
        if category == "apostrophe_casing":
            text, new_ops = _remove_apostrophes(text)
            ops.extend(new_ops)
            text, new_ops = _random_caps(text, rng)
            ops.extend(new_ops)
        if category == "heavy_mixed":
            text, new_ops = _extra_spaces(text, rng)
            ops.extend(new_ops)
    if suffix:
        # The suffix is an in-sentence continuation ("... now"), so move the
        # terminal period to the end instead of leaving it mid-sentence.
        text = f"{prefix}{text.rstrip('.')}{suffix}"
        if not text.endswith((".", "!", "?")):
            text = f"{text}."
    else:
        text = f"{prefix}{text}"
    return text, ops


# ---------------------------------------------------------------------------
# Field and query noise
# ---------------------------------------------------------------------------


def _noise_triple(
    subject: str,
    predicate: str,
    obj: str,
    level: str,
    category: str,
    rng: random.Random,
    family_id: str,
) -> Tuple[Dict[str, str], List[str]]:
    data = {"subject": subject, "predicate": predicate, "object": obj}
    ops: List[str] = []

    if level == "A":
        if category == "spelling_typo":
            equivalent = rng.random() < 0.5
            kinds = (("transposition", "deletion", "insertion")
                     if equivalent else ("substitution",))
            field = rng.choice(("subject", "object", "predicate"))
            data[field], new_ops = _apply_typos(
                data[field], rng, 1, kinds,
                protected=(predicate if field == "subject" else ()))
            ops.extend(new_ops)
        elif category == "lowercase_spacing":
            for field in ("subject", "predicate", "object"):
                data[field], new_ops = _lowercase(data[field])
                ops.extend(new_ops)
        elif category == "apostrophe_casing":
            data["subject"], new_ops = _remove_apostrophes(data["subject"])
            ops.extend(new_ops)
        elif category == "minor_mixed":
            field = rng.choice(("subject", "object"))
            data[field], new_ops = _apply_typos(
                data[field], rng, 1, ("transposition", "deletion"))
            ops.extend(new_ops)
            for name in ("subject", "predicate", "object"):
                data[name], new_ops = _lowercase(data[name])
                ops.extend(new_ops)
    elif level == "B":
        if category == "abbreviations":
            # The predicate is the field most likely to carry a mapped
            # abbreviation, so try it first; fall back to the object only
            # when no predicate phrase matched.
            data["predicate"], new_ops = _abbreviate_text(
                data["predicate"], rng, count=2)
            ops.extend(new_ops)
            if not new_ops and obj:
                data["object"], new_ops = _abbreviate_text(
                    data["object"], rng, count=2)
                ops.extend(new_ops)
        elif category == "grammar":
            data["predicate"], new_ops = _agreement(data["predicate"], rng)
            ops.extend(new_ops)
            if not new_ops:
                data["object"], new_ops = _drop_articles(data["object"], rng)
                ops.extend(new_ops)
            data["predicate"], new_ops = _lowercase(data["predicate"])
            ops.extend(new_ops)
        elif category == "missing_word":
            # A dropped function word keeps the meaning intact.  Cutting a
            # content word short would make the pair semantically different
            # and confound the robustness measurement; the query-side noise
            # still exercises incomplete phrasing.
            field = rng.choice(("predicate", "object"))
            data[field], new_ops = _drop_function_word(data[field], rng)
            ops.extend(new_ops)
        elif category == "repeated_word":
            field = rng.choice(("predicate", "object"))
            data[field], new_ops = _repeat_word(data[field], rng)
            ops.extend(new_ops)
        elif category == "informal_expression":
            data["predicate"] = FAMILY_BY_ID[family_id]["informal_predicate"]
            ops.append(f"informal_predicate:{data['predicate']}")
    elif level == "C":
        if category == "multiple_misspellings":
            for field in ("subject", "object"):
                data[field], new_ops = _apply_typos(
                    data[field], rng, 2, ("substitution",))
                ops.extend(new_ops)
            data["predicate"], new_ops = _apply_typos(
                data["predicate"], rng, 1, ("substitution",))
            ops.extend(new_ops)
        elif category == "phonetic_stt":
            for field in ("subject", "object"):
                data[field], new_ops = _phonetic_text(data[field], rng)
                ops.extend(new_ops)
            if not ops:
                data["predicate"], new_ops = _phonetic_text(
                    data["predicate"], rng)
                ops.extend(new_ops)
        elif category == "disorganized_order":
            if rng.random() < 0.5:
                data["object"], new_ops = _shuffle_words(data["object"], rng)
                ops.extend(new_ops)
        elif category == "heavy_mixed":
            data["subject"], new_ops = _apply_typos(
                data["subject"], rng, 2, ("substitution",))
            ops.extend(new_ops)
            data["object"], new_ops = _apply_typos(
                data["object"], rng, 2, ("substitution",))
            ops.extend(new_ops)
            data["predicate"], new_ops = _abbreviate_text(
                data["predicate"], rng, count=1)
            ops.extend(new_ops)
            for name in ("subject", "predicate", "object"):
                data[name], new_ops = _lowercase(data[name])
                ops.extend(new_ops)
        elif category == "incomplete_query":
            pass
    elif level == "D":
        if category in ("paraphrase", "synonym"):
            data["predicate"] = FAMILY_BY_ID[family_id]["synonym_predicate"]
            ops.append(f"synonym_predicate:{data['predicate']}")
    elif level == "E":
        # Level E isolates meaning discrimination (near misses, corrections,
        # negation, conflicting numbers, unsupported attributes).  Any
        # ingestion typo here would confound the semantic contrast with
        # surface noise, so E pairs keep their stored fields exact.
        pass

    for name in ("subject", "predicate", "object"):
        if not str(data[name]).strip():
            data[name] = {"subject": subject, "predicate": predicate,
                          "object": obj}[name]
    return data, ops


def _noise_query(
    clean_query: str,
    subject: str,
    noisy_subject: str,
    family: Dict[str, Any],
    level: str,
    category: str,
    rng: random.Random,
) -> Tuple[str, List[str]]:
    query = clean_query
    ops: List[str] = []
    if noisy_subject and noisy_subject != subject:
        query = query.replace(subject, noisy_subject)
        ops.append("query_subject_noisy")

    if level == "A":
        if category == "spelling_typo":
            query, new_ops = _apply_typos(
                query, rng, 1,
                ("transposition", "deletion", "substitution"))
            ops.extend(new_ops)
        elif category == "punctuation":
            query, new_ops = _depunct(query)
            ops.extend(new_ops)
        elif category == "lowercase_spacing":
            query, new_ops = _lowercase(query)
            ops.extend(new_ops)
            if rng.random() < 0.5:
                query, new_ops = _extra_spaces(query, rng)
                ops.extend(new_ops)
        elif category == "apostrophe_casing":
            query, new_ops = _remove_apostrophes(query)
            ops.extend(new_ops)
            query, new_ops = _random_caps(query, rng)
            ops.extend(new_ops)
        elif category == "minor_mixed":
            query, new_ops = _apply_typos(
                query, rng, 1, ("transposition", "deletion"))
            ops.extend(new_ops)
            query, new_ops = _lowercase(query)
            ops.extend(new_ops)
            query, new_ops = _depunct(query)
            ops.extend(new_ops)
    elif level == "B":
        if category == "abbreviations":
            query, new_ops = _abbreviate_text(query, rng, count=2)
            ops.extend(new_ops)
        elif category == "grammar":
            query, new_ops = _agreement(query, rng)
            ops.extend(new_ops)
            if not new_ops:
                query, new_ops = _drop_articles(query, rng)
                ops.extend(new_ops)
            query, new_ops = _lowercase(query)
            ops.extend(new_ops)
        elif category == "missing_word":
            query, new_ops = _drop_function_word(query, rng)
            ops.extend(new_ops)
        elif category == "repeated_word":
            query, new_ops = _repeat_word(query, rng)
            ops.extend(new_ops)
        elif category == "informal_expression":
            query = family["informal_query"].format(name=noisy_subject)
            ops.append("informal_query")
    elif level == "C":
        if category == "multiple_misspellings":
            query, new_ops = _apply_typos(
                query, rng, 3, ("substitution",))
            ops.extend(new_ops)
        elif category == "incomplete_query":
            query, new_ops = _truncate_query(query, rng)
            ops.extend(new_ops)
        elif category == "phonetic_stt":
            query, new_ops = _phonetic_text(query, rng)
            ops.extend(new_ops)
        elif category == "disorganized_order":
            query, new_ops = _shuffle_non_proper(query, rng)
            ops.extend(new_ops)
        elif category == "heavy_mixed":
            query, new_ops = _apply_typos(
                query, rng, 3, ("substitution",))
            ops.extend(new_ops)
            query, new_ops = _abbreviate_text(query, rng, count=1)
            ops.extend(new_ops)
            query, new_ops = _lowercase(query)
            ops.extend(new_ops)
            query, new_ops = _depunct(query)
            ops.extend(new_ops)
    elif level == "D":
        if category == "paraphrase":
            query = "Can you tell me " + clean_query[0].lower() + clean_query[1:]
            ops.append("paraphrase_query")
        elif category == "synonym":
            query = family["queries"][1].format(name=noisy_subject)
            ops.append("synonym_query")
        elif category == "word_order":
            query = f"What is the {family['attribute']} of {noisy_subject}?"
            ops.append("late_name_query")
        elif category in ("indirect_reference", "context_followup"):
            # The case builder supplies the indirect/context query.
            pass
    elif level == "E":
        # E contrasts meaning, not surface form.  Lowercasing and removed
        # punctuation are harmless, but a typo could blur a near-miss or
        # change an unknown subject into a known one, so none is applied.
        if rng.random() < 0.5:
            query, new_ops = _lowercase(query)
            ops.extend(new_ops)
        if rng.random() < 0.5:
            query, new_ops = _depunct(query)
            ops.extend(new_ops)

    if query == clean_query:
        query = clean_query.lower()
        ops.append("fallback_lowercase")
    return query, ops


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------


def _observation(
    obs_id: str,
    subject: str,
    predicate: str,
    obj: str,
    timestamp: float,
    noisy: Dict[str, str],
    clean_text: str,
    noisy_text: str,
) -> Dict[str, Any]:
    return {
        "id": obs_id,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "noisy_subject": noisy["subject"],
        "noisy_predicate": noisy["predicate"],
        "noisy_object": noisy["object"],
        "timestamp": timestamp,
        "clean_text": clean_text,
        "noisy_text": noisy_text,
    }


def _base_timestamp(index: int, offset: float = 0.0) -> float:
    return 1_000_000.0 + float(index) * 10_000.0 + offset


def _simple_case(
    case_id: str,
    level: str,
    category: str,
    subject: str,
    family: Dict[str, Any],
    rng: random.Random,
    timestamp: float,
) -> Dict[str, Any]:
    predicate = family["predicate"]
    obj = rng.choice(family["objects"])
    if level == "D" and category == "synonym":
        # _noise_query rewrites the synonym condition to queries[1]; using
        # the same template for the clean side would make the two queries
        # identical and invalidate the clean/noisy comparison.
        clean_query = family["queries"][0].format(name=subject)
    else:
        clean_query = rng.choice(family["queries"]).format(name=subject)
    noisy, ingest_ops = _noise_triple(
        subject, predicate, obj, level, category, rng, family["id"])
    noisy_query, query_ops = _noise_query(
        clean_query, subject, noisy["subject"], family, level, category, rng)
    clean_sentence = f"{subject} {predicate} {obj}."
    noisy_sentence, sentence_ops = _compose_sentence(
        noisy["subject"], noisy["predicate"], noisy["object"],
        level, category, rng)
    observation = _observation(
        "target", subject, predicate, obj, timestamp, noisy,
        clean_sentence, noisy_sentence)
    return {
        "case_id": case_id,
        "level": level,
        "category": category,
        "noise_family": NOISE_FAMILY[level],
        "kind": "retrieval",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "clean_sentence": clean_sentence,
        "noisy_sentence": noisy_sentence,
        "clean_query": clean_query,
        "noisy_query": noisy_query,
        "expected": {"subject": subject, "predicate": predicate,
                     "object": obj},
        "observations": [observation],
        "target_observation": "target",
        "expected_active_count": 1,
        "as_of": None,
        "expected_historical_object": None,
        "noise_ops": ingest_ops + sentence_ops + query_ops,
        "metadata": {
            "family": family["id"],
            "attribute": family["attribute"],
            "timestamp": timestamp,
        },
    }


def _indirect_case(
    case_id: str,
    level: str,
    category: str,
    subject: str,
    target_family: Dict[str, Any],
    anchor_family: Dict[str, Any],
    rng: random.Random,
    timestamp: float,
) -> Dict[str, Any]:
    predicate = target_family["predicate"]
    obj = rng.choice(target_family["objects"])
    anchor_obj = rng.choice(anchor_family["objects"])
    clean_query = rng.choice(target_family["queries"]).format(name=subject)
    if rng.random() < 0.5:
        indirect_query = (
            f"What does the person who studies {anchor_obj} drink?")
        target_family = FAMILY_BY_ID["drinks"]
        predicate = target_family["predicate"]
        obj = rng.choice(target_family["objects"])
        clean_query = f"What does {subject} drink?"
    else:
        indirect_query = (
            f"What is the favorite color of the person who studies "
            f"{anchor_obj}?")
        target_family = FAMILY_BY_ID["favorite_color"]
        predicate = target_family["predicate"]
        obj = rng.choice(target_family["objects"])
        clean_query = f"What is {subject}'s favorite color?"
    if category == "context_followup":
        indirect_query = (
            f"Earlier we discussed {subject}, who studies {anchor_obj}. "
            f"{indirect_query}")
    noisy_query, query_ops = _noise_query(
        indirect_query, subject, subject, target_family, "D", category, rng)
    target_clean = f"{subject} {predicate} {obj}."
    anchor_clean = f"{subject} studies {anchor_obj}."
    target_noisy, target_ops = _noise_triple(
        subject, predicate, obj, "D", category, rng, target_family["id"])
    anchor_noisy, anchor_ops = _noise_triple(
        subject, "studies", anchor_obj, "D", category, rng, "studies")
    target_noisy_sentence, _ = _compose_sentence(
        target_noisy["subject"], target_noisy["predicate"],
        target_noisy["object"], "D", category, rng)
    anchor_noisy_sentence, _ = _compose_sentence(
        anchor_noisy["subject"], anchor_noisy["predicate"],
        anchor_noisy["object"], "D", category, rng)
    observations = [
        _observation("anchor", subject, "studies", anchor_obj,
                     timestamp, anchor_noisy, anchor_clean,
                     anchor_noisy_sentence),
        _observation("target", subject, predicate, obj,
                     timestamp + 100.0, target_noisy, target_clean,
                     target_noisy_sentence),
    ]
    return {
        "case_id": case_id,
        "level": level,
        "category": category,
        "noise_family": NOISE_FAMILY[level],
        "kind": "retrieval",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "clean_sentence": target_clean,
        "noisy_sentence": target_noisy_sentence,
        "clean_query": clean_query,
        "noisy_query": noisy_query,
        "expected": {"subject": subject, "predicate": predicate,
                     "object": obj},
        "observations": observations,
        "target_observation": "target",
        "expected_active_count": 2,
        "as_of": None,
        "expected_historical_object": None,
        "anchor": {"subject": subject, "predicate": "studies",
                   "object": anchor_obj},
        "noise_ops": target_ops + anchor_ops + query_ops,
        "metadata": {
            "family": target_family["id"],
            "attribute": target_family["attribute"],
            "anchor_family": anchor_family["id"],
            "timestamp": timestamp,
        },
    }


def _near_miss_object_case(
    case_id: str,
    subject: str,
    family: Dict[str, Any],
    target_obj: str,
    near_obj: str,
    relation: str,
    rng: random.Random,
    timestamp: float,
) -> Dict[str, Any]:
    predicate = family["predicate"]
    clean_query = f"Is {subject}'s {family['attribute']} {target_obj}?"
    noisy_query, query_ops = _noise_query(
        clean_query, subject, subject, family, "E", "near_miss_object", rng)
    target_noisy, target_ops = _noise_triple(
        subject, predicate, target_obj, "E", "near_miss_object", rng,
        family["id"])
    near_noisy, near_ops = _noise_triple(
        subject, predicate, near_obj, "E", "near_miss_object", rng,
        family["id"])
    observations = [
        _observation("near_miss", subject, predicate, near_obj,
                     timestamp, near_noisy,
                     f"{subject} {predicate} {near_obj}.",
                     f"{near_noisy['subject']} {near_noisy['predicate']} "
                     f"{near_noisy['object']}."),
        _observation("target", subject, predicate, target_obj,
                     timestamp + 100.0, target_noisy,
                     f"{subject} {predicate} {target_obj}.",
                     f"{target_noisy['subject']} "
                     f"{target_noisy['predicate']} "
                     f"{target_noisy['object']}."),
    ]
    return {
        "case_id": case_id,
        "level": "E",
        "category": "near_miss_object",
        "noise_family": NOISE_FAMILY["E"],
        "kind": "paired",
        "subcategory": relation,
        "subject": subject,
        "predicate": predicate,
        "object": target_obj,
        "clean_sentence": f"{subject} {predicate} {target_obj}.",
        "noisy_sentence": (
            f"{target_noisy['subject']} {target_noisy['predicate']} "
            f"{target_noisy['object']}."),
        "clean_query": clean_query,
        "noisy_query": noisy_query,
        "expected": {"subject": subject, "predicate": predicate,
                     "object": target_obj},
        "observations": observations,
        "target_observation": "target",
        "near_miss_observation": "near_miss",
        "expected_active_count": 2,
        "as_of": None,
        "expected_historical_object": None,
        "noise_ops": target_ops + near_ops + query_ops,
        "metadata": {
            "family": family["id"],
            "attribute": family["attribute"],
            "near_miss_object": near_obj,
            "timestamp": timestamp,
        },
    }


def _near_miss_entity_case(
    case_id: str,
    subject_a: str,
    subject_b: str,
    relation: str,
    family: Dict[str, Any],
    rng: random.Random,
    timestamp: float,
) -> Dict[str, Any]:
    predicate = family["predicate"]
    obj = rng.choice(family["objects"])
    clean_query = f"Where does {subject_a} work?" if family["id"] == "works_at" \
        else family["queries"][0].format(name=subject_a)
    noisy_query, query_ops = _noise_query(
        clean_query, subject_a, subject_a, family, "E",
        "near_miss_entity", rng)
    a_noisy, a_ops = _noise_triple(
        subject_a, predicate, obj, "E", "near_miss_entity", rng,
        family["id"])
    b_noisy, b_ops = _noise_triple(
        subject_b, predicate, obj, "E", "near_miss_entity", rng,
        family["id"])
    observations = [
        _observation("target", subject_a, predicate, obj,
                     timestamp, a_noisy,
                     f"{subject_a} {predicate} {obj}.",
                     f"{a_noisy['subject']} {a_noisy['predicate']} "
                     f"{a_noisy['object']}."),
        _observation("near_miss", subject_b, predicate, obj,
                     timestamp + 100.0, b_noisy,
                     f"{subject_b} {predicate} {obj}.",
                     f"{b_noisy['subject']} {b_noisy['predicate']} "
                     f"{b_noisy['object']}."),
    ]
    return {
        "case_id": case_id,
        "level": "E",
        "category": "near_miss_entity",
        "noise_family": NOISE_FAMILY["E"],
        "kind": "paired",
        "subcategory": relation,
        "subject": subject_a,
        "predicate": predicate,
        "object": obj,
        "clean_sentence": f"{subject_a} {predicate} {obj}.",
        "noisy_sentence": (
            f"{a_noisy['subject']} {a_noisy['predicate']} "
            f"{a_noisy['object']}."),
        "clean_query": clean_query,
        "noisy_query": noisy_query,
        "expected": {"subject": subject_a, "predicate": predicate,
                     "object": obj},
        "observations": observations,
        "target_observation": "target",
        "near_miss_observation": "near_miss",
        "near_miss_subject": subject_b,
        "expected_active_count": 2,
        "as_of": None,
        "expected_historical_object": None,
        "noise_ops": a_ops + b_ops + query_ops,
        "metadata": {
            "family": family["id"],
            "attribute": family["attribute"],
            "near_miss_subject": subject_b,
            "timestamp": timestamp,
        },
    }


def _correction_case(
    case_id: str,
    subject: str,
    family: Dict[str, Any],
    old_obj: str,
    new_obj: str,
    rng: random.Random,
    timestamp: float,
    category: str = "correction",
) -> Dict[str, Any]:
    predicate = family["predicate"]
    clean_query = family["queries"][0].format(name=subject)
    noisy_query, query_ops = _noise_query(
        clean_query, subject, subject, family, "E", category, rng)
    old_noisy, old_ops = _noise_triple(
        subject, predicate, old_obj, "E", category, rng, family["id"])
    new_noisy, new_ops = _noise_triple(
        subject, predicate, new_obj, "E", category, rng, family["id"])
    old_sentence = f"{subject} {predicate} {old_obj}."
    new_sentence = f"Actually, {subject} {predicate} {new_obj} now."
    noisy_new_sentence, sentence_ops = _compose_sentence(
        new_noisy["subject"], new_noisy["predicate"], new_noisy["object"],
        "B", "grammar", rng, prefix="actually ", suffix=" now")
    observations = [
        _observation("old", subject, predicate, old_obj,
                     timestamp, old_noisy, old_sentence,
                     f"{old_noisy['subject']} {old_noisy['predicate']} "
                     f"{old_noisy['object']}."),
        _observation("target", subject, predicate, new_obj,
                     timestamp + 100.0, new_noisy, new_sentence,
                     noisy_new_sentence),
    ]
    return {
        "case_id": case_id,
        "level": "E",
        "category": category,
        "noise_family": NOISE_FAMILY["E"],
        "kind": "correction",
        "subject": subject,
        "predicate": predicate,
        "object": new_obj,
        "clean_sentence": new_sentence,
        "noisy_sentence": noisy_new_sentence,
        "clean_query": clean_query,
        "noisy_query": noisy_query,
        "expected": {"subject": subject, "predicate": predicate,
                     "object": new_obj},
        "observations": observations,
        "target_observation": "target",
        "old_observation": "old",
        "expected_active_count": 1,
        "as_of": timestamp + 50.0,
        "expected_historical_object": old_obj,
        "noise_ops": old_ops + new_ops + sentence_ops + query_ops,
        "metadata": {
            "family": family["id"],
            "attribute": family["attribute"],
            "old_object": old_obj,
            "timestamp": timestamp,
        },
    }


def _negation_case(
    case_id: str,
    subject: str,
    positive_predicate: str,
    negative_predicate: str,
    obj: str,
    rng: random.Random,
    timestamp: float,
    positive_first: bool = True,
) -> Dict[str, Any]:
    if positive_first:
        old_pred, new_pred = positive_predicate, negative_predicate
        old_obj, new_obj = obj, obj
        expected_pred = negative_predicate
    else:
        old_pred, new_pred = negative_predicate, positive_predicate
        old_obj, new_obj = obj, obj
        expected_pred = positive_predicate
    attribute_query = {
        "is allergic to": f"What is {subject} allergic to?",
        "is not allergic to": f"What is {subject} allergic to?",
        "eats": f"Does {subject} eat meat?",
        "does not eat": f"Does {subject} eat meat?",
    }.get(positive_predicate, f"Does {subject} eat meat?")
    clean_query = attribute_query
    noisy_query, query_ops = _noise_query(
        clean_query, subject, subject, FAMILY_BY_ID["allergic_to"],
        "E", "negation", rng)
    old_noisy, old_ops = _noise_triple(
        subject, old_pred, old_obj, "E", "negation", rng, "allergic_to")
    new_noisy, new_ops = _noise_triple(
        subject, new_pred, new_obj, "E", "negation", rng, "allergic_to")
    old_sentence = f"{subject} {old_pred} {old_obj}."
    new_sentence = f"{subject} {new_pred} {new_obj} now."
    observations = [
        _observation("old", subject, old_pred, old_obj,
                     timestamp, old_noisy, old_sentence,
                     f"{old_noisy['subject']} {old_noisy['predicate']} "
                     f"{old_noisy['object']}."),
        _observation("target", subject, new_pred, new_obj,
                     timestamp + 100.0, new_noisy, new_sentence,
                     f"{new_noisy['subject']} {new_noisy['predicate']} "
                     f"{new_noisy['object']} now."),
    ]
    return {
        "case_id": case_id,
        "level": "E",
        "category": "negation",
        "noise_family": NOISE_FAMILY["E"],
        "kind": "negation",
        "subject": subject,
        "predicate": expected_pred,
        "object": new_obj,
        "clean_sentence": new_sentence,
        "noisy_sentence": (
            f"{new_noisy['subject']} {new_noisy['predicate']} "
            f"{new_noisy['object']}."),
        "clean_query": clean_query,
        "noisy_query": noisy_query,
        "expected": {"subject": subject, "predicate": expected_pred,
                     "object": new_obj},
        "observations": observations,
        "target_observation": "target",
        "old_observation": "old",
        "expected_active_count": 1,
        "as_of": timestamp + 50.0,
        "expected_historical_object": obj,
        "noise_ops": old_ops + new_ops + query_ops,
        "metadata": {
            "family": "negation",
            "positive_predicate": positive_predicate,
            "negative_predicate": negative_predicate,
            "timestamp": timestamp,
        },
    }


def _unsupported_case(
    case_id: str,
    subject: str,
    stored_family: Dict[str, Any],
    rng: random.Random,
    timestamp: float,
    unknown_subject: bool = False,
) -> Dict[str, Any]:
    stored_obj = rng.choice(stored_family["objects"])
    attribute, query = UNSUPPORTED_ATTRIBUTE[stored_family["id"]]
    query_subject = subject
    if unknown_subject:
        query_subject = rng.choice(_FALLBACK_SUBJECTS)
    clean_query = query.format(name=query_subject)
    noisy_query, query_ops = _noise_query(
        clean_query, query_subject, query_subject, stored_family, "E",
        "unsupported", rng)
    stored_noisy, stored_ops = _noise_triple(
        subject, stored_family["predicate"], stored_obj, "E",
        "unsupported", rng, stored_family["id"])
    stored_sentence = (
        f"{subject} {stored_family['predicate']} {stored_obj}.")
    observations = [
        _observation("stored", subject, stored_family["predicate"],
                     stored_obj, timestamp, stored_noisy, stored_sentence,
                     f"{stored_noisy['subject']} "
                     f"{stored_noisy['predicate']} "
                     f"{stored_noisy['object']}."),
    ]
    return {
        "case_id": case_id,
        "level": "E",
        "category": "unsupported",
        "noise_family": NOISE_FAMILY["E"],
        "kind": "unsupported",
        "subcategory": "unknown_subject" if unknown_subject else "known_subject",
        "subject": subject,
        "query_subject": query_subject,
        "predicate": stored_family["predicate"],
        "object": stored_obj,
        "clean_sentence": stored_sentence,
        "noisy_sentence": (
            f"{stored_noisy['subject']} {stored_noisy['predicate']} "
            f"{stored_noisy['object']}."),
        "clean_query": clean_query,
        "noisy_query": noisy_query,
        "expected": None,
        "unsupported_attribute": attribute,
        "observations": observations,
        "target_observation": None,
        "expected_active_count": 1,
        "as_of": None,
        "expected_historical_object": None,
        "noise_ops": stored_ops + query_ops,
        "metadata": {
            "family": stored_family["id"],
            "attribute": stored_family["attribute"],
            "timestamp": timestamp,
        },
    }


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def _subjects(count: int, seed: int) -> List[str]:
    pairs = [(first, last) for first in FIRST_NAMES for last in LAST_NAMES]
    rng = _rng(seed, "subjects")
    rng.shuffle(pairs)
    return [f"{first} {last}" for first, last in pairs[:count]]


def _abbreviatable(family: Dict[str, Any]) -> bool:
    text = f"{family['predicate']} {family['queries'][0]}"
    return any(
        re.search(
            r"(?i)(?<![A-Za-z])" + re.escape(phrase) + r"(?![A-Za-z])",
            text)
        for phrase, _ in ABBREVIATIONS)


ABBREVIATABLE_FAMILIES = [
    family for family in FACT_FAMILIES if _abbreviatable(family)]


def generate_cases(seed: int = DEFAULT_SEED) -> List[Dict[str, Any]]:
    subjects = _subjects(CASES_PER_LEVEL * 5 + 300, seed)
    cursor = 0
    cases: List[Dict[str, Any]] = []
    global_index = 0
    family_cursor = 0
    abbrev_cursor = 0
    for level in ("A", "B", "C", "D", "E"):
        level_index = 0
        for category, count in LEVEL_CATEGORIES[level]:
            for category_index in range(count):
                subject = subjects[cursor]
                cursor += 1
                case_id = f"{level}{level_index + 1:03d}"
                timestamp = _base_timestamp(global_index, 100.0)
                rng = _rng(seed, case_id, "case")
                if level in ("A", "B", "C", "D"):
                    if category == "abbreviations":
                        family = ABBREVIATABLE_FAMILIES[
                            abbrev_cursor % len(ABBREVIATABLE_FAMILIES)]
                        abbrev_cursor += 1
                    else:
                        family = FACT_FAMILIES[
                            family_cursor % len(FACT_FAMILIES)]
                        family_cursor += 1
                    if level == "D" and category in (
                            "indirect_reference", "context_followup"):
                        target = rng.choice(
                            [FAMILY_BY_ID["drinks"],
                             FAMILY_BY_ID["favorite_color"]])
                        case = _indirect_case(
                            case_id, level, category, subject, target,
                            D_ANCHOR_FAMILY, rng, timestamp)
                    else:
                        case = _simple_case(
                            case_id, level, category, subject, family,
                            rng, timestamp)
                else:
                    case = _e_case(
                        case_id, category, subject, rng, timestamp,
                        category_index)
                case["case_index"] = global_index
                cases.append(case)
                global_index += 1
                level_index += 1
    return cases


def _e_case(
    case_id: str,
    category: str,
    subject: str,
    rng: random.Random,
    timestamp: float,
    category_index: int,
) -> Dict[str, Any]:
    if category == "near_miss_object":
        target_obj, near_obj, relation = NEAR_MISS_OBJECT_PAIRS[
            category_index % len(NEAR_MISS_OBJECT_PAIRS)]
        family = FAMILY_BY_ID["favorite_film"]
        if "Le Guin" in target_obj or "Morrison" in target_obj \
                or "Calvino" in target_obj or "Murakami" in target_obj \
                or "Atwood" in target_obj:
            family = FAMILY_BY_ID["favorite_author"]
        return _near_miss_object_case(
            case_id, subject, family, target_obj, near_obj, relation,
            rng, timestamp)
    if category == "near_miss_entity":
        subject_a, subject_b, relation = NEAR_MISS_ENTITY_PAIRS[
            category_index % len(NEAR_MISS_ENTITY_PAIRS)]
        family = FACT_FAMILIES[category_index % len(FACT_FAMILIES)]
        return _near_miss_entity_case(
            case_id, subject_a, subject_b, relation, family, rng, timestamp)
    if category == "correction":
        family = FAMILY_BY_ID[EXCLUSIVE_FAMILIES[
            category_index % len(EXCLUSIVE_FAMILIES)]]
        objects = list(family["objects"])
        old_obj = rng.choice(objects)
        new_obj = rng.choice([item for item in objects if item != old_obj])
        return _correction_case(
            case_id, subject, family, old_obj, new_obj, rng, timestamp)
    if category == "negation":
        if category_index % 2 == 0:
            return _negation_case(
                case_id, subject, "is allergic to", "is not allergic to",
                rng.choice(ALLERGENS), rng, timestamp,
                positive_first=bool(category_index % 4))
        return _negation_case(
            case_id, subject, "eats", "does not eat", "meat", rng,
            timestamp, positive_first=bool(category_index % 4))
    if category == "conflicting_numbers":
        family = FAMILY_BY_ID[NUMERIC_FAMILIES[
            category_index % len(NUMERIC_FAMILIES)]]
        objects = list(family["objects"])
        old_obj = rng.choice(objects)
        new_obj = rng.choice([item for item in objects if item != old_obj])
        return _correction_case(
            case_id, subject, family, old_obj, new_obj, rng, timestamp,
            category="conflicting_numbers")
    if category == "unsupported":
        supported = [family for family in FACT_FAMILIES
                     if family["id"] in UNSUPPORTED_ATTRIBUTE]
        family = supported[category_index % len(supported)]
        return _unsupported_case(
            case_id, subject, family, rng, timestamp,
            unknown_subject=category_index >= 8)
    raise ValueError(f"unknown E category: {category}")


# ---------------------------------------------------------------------------
# Lifecycle scenarios
# ---------------------------------------------------------------------------


def _lifecycle_observation(
    obs_id: str,
    subject: str,
    predicate: str,
    obj: str,
    timestamp: float,
    noisy: Dict[str, str],
    clean_text: str,
    noisy_text: str,
) -> Dict[str, Any]:
    return _observation(
        obs_id, subject, predicate, obj, timestamp, noisy,
        clean_text, noisy_text)


def _lifecycle_scenario(
    scenario_id: str,
    category: str,
    subject: str,
    predicate: str,
    object_value: Optional[str],
    observations: List[Dict[str, Any]],
    expected_final_action: Optional[str],
    expected_active_count: int,
    query: str,
    *,
    noisy_query: Optional[str] = None,
    as_of: Optional[float] = None,
    expected_historical_object: Optional[str] = None,
    expected_current_object: Optional[str] = None,
    expected_update_count: Optional[int] = None,
    expected_total_count: Optional[int] = None,
    distinct_facts: Optional[List[Dict[str, str]]] = None,
    subcategory: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "category": category,
        "subcategory": subcategory,
        "subject": subject,
        "predicate": predicate,
        "object": object_value,
        "observations": observations,
        "expected_final_action": expected_final_action,
        "expected_active_count": expected_active_count,
        "expected_total_count": (
            expected_total_count if expected_total_count is not None
            else expected_active_count),
        "expected_current_object": expected_current_object,
        "expected_historical_object": expected_historical_object,
        "expected_update_count": expected_update_count,
        "distinct_facts": distinct_facts or [],
        "query": query,
        "noisy_query": noisy_query or query.lower(),
        "as_of": as_of,
    }


def generate_lifecycle(seed: int = DEFAULT_SEED) -> List[Dict[str, Any]]:
    subjects = _subjects(
        CASES_PER_LEVEL * 5 + 300, seed + 17)[-260:]
    cursor = 0
    scenarios: List[Dict[str, Any]] = []
    index = 0
    for category, count in LIFECYCLE_CATEGORIES:
        for category_index in range(count):
            subject = subjects[cursor]
            cursor += 1
            scenario_id = f"L{index + 1:03d}"
            timestamp = 2_000_000.0 + float(index) * 10_000.0 + 100.0
            rng = _rng(seed, scenario_id, "lifecycle")
            scenario = _lifecycle_case(
                scenario_id, category, subject, rng, timestamp,
                category_index)
            scenarios.append(scenario)
            index += 1
    return scenarios


def _lifecycle_case(
    scenario_id: str,
    category: str,
    subject: str,
    rng: random.Random,
    timestamp: float,
    category_index: int,
) -> Dict[str, Any]:
    if category == "exact_restatement":
        family = FACT_FAMILIES[category_index % len(FACT_FAMILIES)]
        predicate = family["predicate"]
        obj = rng.choice(family["objects"])
        clean_text = f"{subject} {predicate} {obj}."
        obs = [
            _lifecycle_observation(
                "first", subject, predicate, obj, timestamp, {
                    "subject": subject, "predicate": predicate,
                    "object": obj}, clean_text, clean_text),
            _lifecycle_observation(
                "second", subject, predicate, obj, timestamp + 10.0, {
                    "subject": subject, "predicate": predicate,
                    "object": obj}, clean_text, clean_text.lower()),
        ]
        query = family["queries"][0].format(name=subject)
        return _lifecycle_scenario(
            scenario_id, category, subject, predicate, obj, obs,
            "consolidate", 1, query,
            expected_current_object=obj, expected_update_count=2)
    if category == "noisy_restatement":
        family = FACT_FAMILIES[category_index % len(FACT_FAMILIES)]
        predicate = family["predicate"]
        obj = rng.choice(family["objects"])
        noisy, ops = _noise_triple(
            subject, predicate, obj, "A", "spelling_typo", rng,
            family["id"])
        clean_text = f"{subject} {predicate} {obj}."
        obs = [
            _lifecycle_observation(
                "first", subject, predicate, obj, timestamp, {
                    "subject": subject, "predicate": predicate,
                    "object": obj}, clean_text, clean_text),
            _lifecycle_observation(
                "second", subject, predicate, obj, timestamp + 10.0, noisy,
                clean_text,
                f"{noisy['subject']} {noisy['predicate']} "
                f"{noisy['object']}.".lower()),
        ]
        query = family["queries"][0].format(name=subject)
        return _lifecycle_scenario(
            scenario_id, category, subject, predicate, obj, obs,
            "consolidate", 1, query,
            expected_current_object=obj, expected_update_count=2)
    if category == "paraphrase_restatement":
        family = FAMILY_BY_ID[EXCLUSIVE_FAMILIES[
            category_index % len(EXCLUSIVE_FAMILIES)]]
        predicate = family["predicate"]
        obj = rng.choice(family["objects"])
        synonym = family["synonym_predicate"]
        clean_text = f"{subject} {predicate} {obj}."
        noisy_text = f"{subject} {synonym} {obj}."
        obs = [
            _lifecycle_observation(
                "first", subject, predicate, obj, timestamp, {
                    "subject": subject, "predicate": predicate,
                    "object": obj}, clean_text, clean_text),
            _lifecycle_observation(
                "second", subject, synonym, obj, timestamp + 10.0, {
                    "subject": subject, "predicate": synonym,
                    "object": obj}, clean_text, noisy_text.lower()),
        ]
        query = family["queries"][0].format(name=subject)
        return _lifecycle_scenario(
            scenario_id, category, subject, predicate, obj, obs,
            "consolidate", 1, query,
            expected_current_object=obj, expected_update_count=2)
    if category == "correction_exclusive":
        family = FAMILY_BY_ID[EXCLUSIVE_FAMILIES[
            category_index % len(EXCLUSIVE_FAMILIES)]]
        predicate = family["predicate"]
        objects = list(family["objects"])
        old_obj = rng.choice(objects)
        new_obj = rng.choice([item for item in objects if item != old_obj])
        old_noisy, old_ops = _noise_triple(
            subject, predicate, old_obj, "E", "correction", rng,
            family["id"])
        new_noisy, new_ops = _noise_triple(
            subject, predicate, new_obj, "E", "correction", rng,
            family["id"])
        obs = [
            _lifecycle_observation(
                "old", subject, predicate, old_obj, timestamp, old_noisy,
                f"{subject} {predicate} {old_obj}.",
                f"{old_noisy['subject']} {old_noisy['predicate']} "
                f"{old_noisy['object']}."),
            _lifecycle_observation(
                "new", subject, predicate, new_obj, timestamp + 100.0,
                new_noisy, f"Actually, {subject} {predicate} {new_obj} now.",
                f"actually {new_noisy['subject']} "
                f"{new_noisy['predicate']} {new_noisy['object']} now"),
        ]
        query = family["queries"][0].format(name=subject)
        return _lifecycle_scenario(
            scenario_id, category, subject, predicate, new_obj, obs,
            "temporal_update", 1, query,
            as_of=timestamp + 50.0,
            expected_historical_object=old_obj,
            expected_current_object=new_obj,
            expected_total_count=2)
    if category == "correction_numeric":
        family = FAMILY_BY_ID[NUMERIC_FAMILIES[
            category_index % len(NUMERIC_FAMILIES)]]
        predicate = family["predicate"]
        objects = list(family["objects"])
        old_obj = rng.choice(objects)
        new_obj = rng.choice([item for item in objects if item != old_obj])
        obs = [
            _lifecycle_observation(
                "old", subject, predicate, old_obj, timestamp, {
                    "subject": subject, "predicate": predicate,
                    "object": old_obj},
                f"{subject} {predicate} {old_obj}.",
                f"{subject} {predicate} {old_obj}.".lower()),
            _lifecycle_observation(
                "new", subject, predicate, new_obj, timestamp + 100.0, {
                    "subject": subject, "predicate": predicate,
                    "object": new_obj},
                f"Actually, {subject} {predicate} {new_obj} now.",
                f"actually {subject} {predicate} {new_obj} now"),
        ]
        query = family["queries"][0].format(name=subject)
        return _lifecycle_scenario(
            scenario_id, category, subject, predicate, new_obj, obs,
            "temporal_update", 1, query,
            as_of=timestamp + 50.0,
            expected_historical_object=old_obj,
            expected_current_object=new_obj,
            expected_total_count=2)
    if category == "negation_change":
        if category_index % 2 == 0:
            positive, negative = "is allergic to", "is not allergic to"
            obj = rng.choice(ALLERGENS)
            query = f"What is {subject} allergic to?"
        else:
            positive, negative = "eats", "does not eat"
            obj = "meat"
            query = f"Does {subject} eat meat?"
        positive_first = bool(category_index % 4 == 0)
        if positive_first:
            old_pred, new_pred = positive, negative
        else:
            old_pred, new_pred = negative, positive
        obs = [
            _lifecycle_observation(
                "old", subject, old_pred, obj, timestamp, {
                    "subject": subject, "predicate": old_pred,
                    "object": obj},
                f"{subject} {old_pred} {obj}.",
                f"{subject} {old_pred} {obj}.".lower()),
            _lifecycle_observation(
                "new", subject, new_pred, obj, timestamp + 100.0, {
                    "subject": subject, "predicate": new_pred,
                    "object": obj},
                f"{subject} {new_pred} {obj} now.",
                f"{subject} {new_pred} {obj} now".lower()),
        ]
        return _lifecycle_scenario(
            scenario_id, category, subject, new_pred, obj, obs,
            "temporal_update", 1, query,
            as_of=timestamp + 50.0,
            expected_historical_object=obj,
            expected_current_object=obj,
            expected_total_count=2)
    if category == "historical_statement":
        family = FAMILY_BY_ID["lives_in"]
        current_obj = rng.choice(CITIES)
        historical_obj = rng.choice(
            [city for city in CITIES if city != current_obj])
        obs = [
            _lifecycle_observation(
                "current", subject, "lives in", current_obj, timestamp, {
                    "subject": subject, "predicate": "lives in",
                    "object": current_obj},
                f"{subject} lives in {current_obj}.",
                f"{subject} lives in {current_obj}.".lower()),
            _lifecycle_observation(
                "historical", subject, "used to live in", historical_obj,
                timestamp + 100.0, {
                    "subject": subject, "predicate": "used to live in",
                    "object": historical_obj},
                f"{subject} used to live in {historical_obj}.",
                f"{subject} used to live in {historical_obj}.".lower()),
        ]
        query = f"Where does {subject} live?"
        return _lifecycle_scenario(
            scenario_id, category, subject, "lives in", current_obj, obs,
            "historical_observation", 1, query,
            expected_current_object=current_obj,
            expected_historical_object=historical_obj,
            expected_total_count=2)
    if category == "similar_wording_distinct":
        target_obj, near_obj, relation = SIMILAR_WORDING_PAIRS[
            category_index % len(SIMILAR_WORDING_PAIRS)]
        if "Northwind" in target_obj or "Bluepeak" in target_obj:
            # Keep the predicate non-exclusive: 'works at' would make two
            # employers a legitimate temporal replacement, not a merge test.
            predicate = "volunteers at"
            query = f"Where does {subject} volunteer?"
        elif "motorcycle" in target_obj or "telescope" in target_obj:
            predicate = "owns"
            query = f"What does {subject} own?"
        elif "Roman" in target_obj or "Northern" in target_obj:
            predicate = "loves"
            query = f"What does {subject} love?"
        else:
            predicate = "volunteers at"
            query = f"Where does {subject} volunteer?"
        obs = [
            _lifecycle_observation(
                "near", subject, predicate, near_obj, timestamp, {
                    "subject": subject, "predicate": predicate,
                    "object": near_obj},
                f"{subject} {predicate} {near_obj}.",
                f"{subject} {predicate} {near_obj}.".lower()),
            _lifecycle_observation(
                "target", subject, predicate, target_obj,
                timestamp + 100.0, {
                    "subject": subject, "predicate": predicate,
                    "object": target_obj},
                f"{subject} {predicate} {target_obj}.",
                f"{subject} {predicate} {target_obj}.".lower()),
        ]
        return _lifecycle_scenario(
            scenario_id, category, subject, predicate, target_obj, obs,
            None, 2, query,
            expected_current_object=target_obj,
            distinct_facts=[
                {"subject": subject, "predicate": predicate,
                 "object": target_obj},
                {"subject": subject, "predicate": predicate,
                 "object": near_obj},
            ],
            subcategory=relation)
    if category == "near_miss_entity":
        subject_a, subject_b, relation = NEAR_MISS_ENTITY_PAIRS[
            category_index % len(NEAR_MISS_ENTITY_PAIRS)]
        subject = subject_a
        family = FACT_FAMILIES[category_index % len(FACT_FAMILIES)]
        predicate = family["predicate"]
        obj = rng.choice(family["objects"])
        obs = [
            _lifecycle_observation(
                "target", subject_a, predicate, obj, timestamp, {
                    "subject": subject_a, "predicate": predicate,
                    "object": obj},
                f"{subject_a} {predicate} {obj}.",
                f"{subject_a} {predicate} {obj}.".lower()),
            _lifecycle_observation(
                "near_miss", subject_b, predicate, obj, timestamp + 100.0, {
                    "subject": subject_b, "predicate": predicate,
                    "object": obj},
                f"{subject_b} {predicate} {obj}.",
                f"{subject_b} {predicate} {obj}.".lower()),
        ]
        query = family["queries"][0].format(name=subject_a)
        return _lifecycle_scenario(
            scenario_id, category, subject, predicate, obj, obs,
            None, 2, query,
            expected_current_object=obj,
            distinct_facts=[
                {"subject": subject_a, "predicate": predicate,
                 "object": obj},
                {"subject": subject_b, "predicate": predicate,
                 "object": obj},
            ],
            subcategory=relation)
    if category == "cumulative_distinct":
        subject = subject
        obj_a = rng.choice(BEVERAGES)
        obj_b = rng.choice([item for item in BEVERAGES if item != obj_a])
        obs = [
            _lifecycle_observation(
                "first", subject, "likes", obj_a, timestamp, {
                    "subject": subject, "predicate": "likes",
                    "object": obj_a},
                f"{subject} likes {obj_a}.",
                f"{subject} likes {obj_a}.".lower()),
            _lifecycle_observation(
                "second", subject, "likes", obj_b, timestamp + 100.0, {
                    "subject": subject, "predicate": "likes",
                    "object": obj_b},
                f"{subject} likes {obj_b}.",
                f"{subject} likes {obj_b}.".lower()),
        ]
        query = f"What does {subject} like?"
        return _lifecycle_scenario(
            scenario_id, category, subject, "likes", obj_a, obs,
            None, 2, query,
            expected_current_object=obj_b,
            distinct_facts=[
                {"subject": subject, "predicate": "likes", "object": obj_a},
                {"subject": subject, "predicate": "likes", "object": obj_b},
            ])
    if category == "repeated_noisy":
        family = FACT_FAMILIES[category_index % len(FACT_FAMILIES)]
        predicate = family["predicate"]
        obj = rng.choice(family["objects"])
        noisy, ops = _noise_triple(
            subject, predicate, obj, "A", "spelling_typo", rng,
            family["id"])
        clean_text = f"{subject} {predicate} {obj}."
        obs = [
            _lifecycle_observation(
                "first", subject, predicate, obj, timestamp, {
                    "subject": subject, "predicate": predicate,
                    "object": obj}, clean_text, clean_text),
            _lifecycle_observation(
                "second", subject, predicate, obj, timestamp + 10.0, {
                    "subject": subject, "predicate": predicate,
                    "object": obj}, clean_text, clean_text.lower()),
            _lifecycle_observation(
                "third", subject, predicate, obj, timestamp + 20.0, noisy,
                clean_text,
                f"{noisy['subject']} {noisy['predicate']} "
                f"{noisy['object']}.".lower()),
        ]
        query = family["queries"][0].format(name=subject)
        return _lifecycle_scenario(
            scenario_id, category, subject, predicate, obj, obs,
            "consolidate", 1, query,
            expected_current_object=obj, expected_update_count=3)
    if category == "contradiction_marker":
        family = FAMILY_BY_ID["works_at"]
        predicate = "works at"
        objects = list(COMPANIES)
        old_obj = rng.choice(objects)
        if category_index % 2 == 0:
            marker_pred = "now works at"
            new_obj = rng.choice([item for item in objects
                                  if item != old_obj])
            new_sentence = (
                f"{subject} now works at {new_obj}.")
        else:
            marker_pred = "no longer works at"
            new_obj = old_obj
            new_sentence = (
                f"{subject} no longer works at {old_obj}.")
        obs = [
            _lifecycle_observation(
                "old", subject, predicate, old_obj, timestamp, {
                    "subject": subject, "predicate": predicate,
                    "object": old_obj},
                f"{subject} {predicate} {old_obj}.",
                f"{subject} {predicate} {old_obj}.".lower()),
            _lifecycle_observation(
                "new", subject, marker_pred, new_obj, timestamp + 100.0, {
                    "subject": subject, "predicate": marker_pred,
                    "object": new_obj},
                new_sentence, new_sentence.lower()),
        ]
        query = f"Where does {subject} work?"
        return _lifecycle_scenario(
            scenario_id, category, subject, marker_pred, new_obj, obs,
            "temporal_update", 1, query,
            as_of=timestamp + 50.0,
            expected_historical_object=old_obj,
            expected_current_object=new_obj,
            expected_total_count=2,
            subcategory="now" if category_index % 2 == 0 else "no_longer")
    raise ValueError(f"unknown lifecycle category: {category}")


# ---------------------------------------------------------------------------
# Validation and persistence
# ---------------------------------------------------------------------------


def validate_cases(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not 500 <= len(cases) <= 1000:
        raise ValueError(f"expected 500-1000 cases, got {len(cases)}")
    ids: set[str] = set()
    by_level: Dict[str, int] = {}
    by_category: Dict[str, int] = {}
    for case in cases:
        case_id = case["case_id"]
        if case_id in ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        ids.add(case_id)
        level = case["level"]
        by_level[level] = by_level.get(level, 0) + 1
        by_category[case["category"]] = by_category.get(
            case["category"], 0) + 1
        if not case["observations"]:
            raise ValueError(f"{case_id} has no observations")
        if case["kind"] != "unsupported":
            expected = case["expected"]
            if not expected or not all(
                    expected.get(key) for key in
                    ("subject", "predicate", "object")):
                raise ValueError(f"{case_id} has an incomplete expectation")
            target = case["target_observation"]
            if target is None or not any(
                    obs["id"] == target for obs in case["observations"]):
                raise ValueError(f"{case_id} has no target observation")
        for observation in case["observations"]:
            for key in ("subject", "predicate", "object",
                        "noisy_subject", "noisy_predicate",
                        "noisy_object", "clean_text", "noisy_text"):
                if not str(observation[key]).strip():
                    raise ValueError(
                        f"{case_id} observation {observation['id']} "
                        f"has empty {key}")
        if case["level"] != "D" and case["kind"] == "retrieval":
            if case["clean_query"] == case["noisy_query"]:
                raise ValueError(f"{case_id} clean/noisy query are equal")
    for level, count in by_level.items():
        if count < 100:
            raise ValueError(f"level {level} has only {count} cases")
    if set(by_level) != set("ABCDE"):
        raise ValueError(f"missing levels: {sorted(set('ABCDE') - set(by_level))}")
    return {
        "cases": len(cases),
        "by_level": dict(sorted(by_level.items())),
        "by_category": dict(sorted(by_category.items())),
        "by_noise_family": {
            family: sum(1 for case in cases
                        if case["noise_family"] == family)
            for family in ("surface", "informal", "corruption",
                           "semantic", "ambiguity")
        },
    }


def validate_lifecycle(scenarios: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if len(scenarios) < 100:
        raise ValueError("lifecycle set is smaller than 100 scenarios")
    ids: set[str] = set()
    by_category: Dict[str, int] = {}
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        if scenario_id in ids:
            raise ValueError(f"duplicate scenario_id: {scenario_id}")
        ids.add(scenario_id)
        category = scenario["category"]
        by_category[category] = by_category.get(category, 0) + 1
        if not scenario["observations"]:
            raise ValueError(f"{scenario_id} has no observations")
    return {
        "scenarios": len(scenarios),
        "by_category": dict(sorted(by_category.items())),
    }


def _jsonl_bytes(records: Sequence[Dict[str, Any]]) -> bytes:
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True)
             for record in records]
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_jsonl(path: str | Path,
                records: Sequence[Dict[str, Any]]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _jsonl_bytes(records)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    records = []
    for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number} is not valid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{number} is not a JSON object")
        records.append(record)
    return records


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the natural-language robustness dataset")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--cases", default=str(
            Path(__file__).parent / "data" / "cases.jsonl"))
    parser.add_argument(
        "--lifecycle", default=str(
            Path(__file__).parent / "data" / "lifecycle.jsonl"))
    args = parser.parse_args(argv)
    cases = generate_cases(args.seed)
    lifecycle = generate_lifecycle(args.seed)
    case_stats = validate_cases(cases)
    lifecycle_stats = validate_lifecycle(lifecycle)
    case_hash = write_jsonl(args.cases, cases)
    lifecycle_hash = write_jsonl(args.lifecycle, lifecycle)
    print(json.dumps({
        "seed": args.seed,
        "dataset_version": DATASET_VERSION,
        "cases": case_stats,
        "cases_sha256": case_hash,
        "lifecycle": lifecycle_stats,
        "lifecycle_sha256": lifecycle_hash,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
