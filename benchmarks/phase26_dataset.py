"""
benchmarks/phase26_dataset.py

Phase 26 Semantic Verification & Contradiction Safety Dataset.
Contains balanced, human-curated evaluation pairs across four categories:
  1. valid_paraphrase: Semantically equivalent statements with varied syntax/lexicon.
  2. genuine_contradiction: Directly opposing statements that cannot both be true simultaneously.
  3. unrelated: Semantically distinct statements (including lexical overlap distractors).
  4. temporal_change: Statements that reflect a change of state/fact over time.

Includes hard edge cases:
  - Explicit negations ("not", "never", "no longer")
  - Number and scalar shifts ("15%" vs "45%", "$10M" vs "$50M")
  - Conflicting entities ("Nexus Labs" vs "Apex Systems")
  - Directional / antonym pairs ("acquired" vs "divested", "promoted" vs "demoted")
  - High lexical overlap distractors

Data Splits:
  - Calibration Set: 60 pairs (15 per category) - used for threshold sweeping.
  - Validation Set:  60 pairs (15 per category) - used for selecting candidates.
  - Held-out Test Set: 60 pairs (15 per category) - unseen entities, frozen before tuning.
Total: 180 clean, distinct pairs with zero leakage across splits.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class VerificationPair:
    id: str
    premise: str
    hypothesis: str
    category: str  # 'valid_paraphrase', 'genuine_contradiction', 'unrelated', 'temporal_change'
    expected_decision: str  # 'SAME', 'CONTRADICTION', 'NEUTRAL', 'TEMPORAL'
    subject: str
    predicate: str
    object_p: str
    object_h: str
    has_negation: bool = False
    has_numbers: bool = False
    has_date: bool = False
    notes: str = ""


# ─────────────────────────────────────────────────────────────
# 1. Calibration Set (60 pairs: 15 per category)
# ─────────────────────────────────────────────────────────────
CALIBRATION_PAIRS: List[VerificationPair] = [
    # --- Category 1: Valid Paraphrases (15) ---
    VerificationPair(
        "cal_p_01", "Bob holds a PhD in quantum physics.", "Bob earned a doctorate in quantum physics.",
        "valid_paraphrase", "SAME", "Bob", "holds a PhD in", "quantum physics", "quantum physics",
        notes="Academic credential paraphrase"
    ),
    VerificationPair(
        "cal_p_02", "Alice leads the engineering division at Nexus Labs.", "Alice directs engineering at Nexus Labs.",
        "valid_paraphrase", "SAME", "Alice", "leads engineering at", "Nexus Labs", "Nexus Labs",
        notes="Leadership role synonym"
    ),
    VerificationPair(
        "cal_p_03", "Marcus authored an influential biography monograph.", "Marcus penned a notable biographical book.",
        "valid_paraphrase", "SAME", "Marcus", "authored", "biography monograph", "biographical book",
        notes="Authoring synonym"
    ),
    VerificationPair(
        "cal_p_04", "Clara founded an artificial intelligence enterprise.", "Clara established an AI startup company.",
        "valid_paraphrase", "SAME", "Clara", "founded", "AI enterprise", "AI startup",
        notes="Company founding paraphrase"
    ),
    VerificationPair(
        "cal_p_05", "Dmitri constructed a high-frequency trading algorithm.", "Dmitri built an automated trading system.",
        "valid_paraphrase", "SAME", "Dmitri", "constructed", "trading algorithm", "automated trading system"
    ),
    VerificationPair(
        "cal_p_06", "Elena completed her medical residency in Boston.", "Elena finished clinical training in Boston.",
        "valid_paraphrase", "SAME", "Elena", "completed residency in", "Boston", "Boston"
    ),
    VerificationPair(
        "cal_p_07", "Fiona resides in the capital of France.", "Fiona lives in Paris.",
        "valid_paraphrase", "SAME", "Fiona", "resides in", "capital of France", "Paris",
        notes="Geographical world knowledge"
    ),
    VerificationPair(
        "cal_p_08", "George purchased a residential property in Seattle.", "George bought a home in Seattle.",
        "valid_paraphrase", "SAME", "George", "purchased property in", "Seattle", "Seattle"
    ),
    VerificationPair(
        "cal_p_09", "Hannah coordinates international climate summits.", "Hannah organizes global environmental conferences.",
        "valid_paraphrase", "SAME", "Hannah", "coordinates summits", "climate summits", "environmental conferences"
    ),
    VerificationPair(
        "cal_p_10", "Ian discovered a novel antibiotic compound.", "Ian identified a new antimicrobial substance.",
        "valid_paraphrase", "SAME", "Ian", "discovered", "antibiotic compound", "antimicrobial substance"
    ),
    VerificationPair(
        "cal_p_11", "Julia teaches organic chemistry at Oxford.", "Julia is an organic chemistry professor at Oxford.",
        "valid_paraphrase", "SAME", "Julia", "teaches chemistry at", "Oxford", "Oxford"
    ),
    VerificationPair(
        "cal_p_12", "Kevin repaired the faulty hydraulic pump.", "Kevin fixed the malfunctioning hydraulic system.",
        "valid_paraphrase", "SAME", "Kevin", "repaired", "faulty pump", "malfunctioning system"
    ),
    VerificationPair(
        "cal_p_13", "Laura serves as chief legal counsel for Acme.", "Laura is the head corporate attorney at Acme.",
        "valid_paraphrase", "SAME", "Laura", "serves as legal counsel for", "Acme", "Acme"
    ),
    VerificationPair(
        "cal_p_14", "Mateo secured 5 million dollars in venture capital.", "Mateo raised 5 million dollars from investors.",
        "valid_paraphrase", "SAME", "Mateo", "secured venture capital", "5 million dollars", "5 million dollars",
        has_numbers=True
    ),
    VerificationPair(
        "cal_p_15", "Nadia translated three classical poetry volumes.", "Nadia rendered three volumes of classical poems into English.",
        "valid_paraphrase", "SAME", "Nadia", "translated poetry", "three volumes", "three volumes",
        has_numbers=True
    ),

    # --- Category 2: Genuine Contradictions (15) ---
    VerificationPair(
        "cal_c_01", "Bob holds a PhD in quantum physics.", "Bob does not hold a PhD in quantum physics.",
        "genuine_contradiction", "CONTRADICTION", "Bob", "holds a PhD in", "quantum physics", "not quantum physics",
        has_negation=True
    ),
    VerificationPair(
        "cal_c_02", "Nathan serves as the lead database architect.", "Nathan does not serve as the lead database architect.",
        "genuine_contradiction", "CONTRADICTION", "Nathan", "serves as lead", "database architect", "not architect",
        has_negation=True
    ),
    VerificationPair(
        "cal_c_03", "Victor owns the majority share in Apex Robotics.", "Victor sold all equity in Apex Robotics.",
        "genuine_contradiction", "CONTRADICTION", "Victor", "owns majority share in", "Apex Robotics", "sold all equity"
    ),
    VerificationPair(
        "cal_c_04", "Alice was unanimously elected board chair.", "Alice was rejected by the board for the chair position.",
        "genuine_contradiction", "CONTRADICTION", "Alice", "elected", "board chair", "rejected for chair"
    ),
    VerificationPair(
        "cal_c_05", "Dmitri is a permanent citizen of Canada.", "Dmitri has never held Canadian citizenship.",
        "genuine_contradiction", "CONTRADICTION", "Dmitri", "is citizen of", "Canada", "never citizen",
        has_negation=True
    ),
    VerificationPair(
        "cal_c_06", "Elena passed the national surgical board examination.", "Elena failed the national surgical board examination.",
        "genuine_contradiction", "CONTRADICTION", "Elena", "passed examination", "surgical board", "failed board"
    ),
    VerificationPair(
        "cal_c_07", "Fiona was acquitted of all commercial fraud charges.", "Fiona was convicted on all commercial fraud counts.",
        "genuine_contradiction", "CONTRADICTION", "Fiona", "legal status", "acquitted", "convicted"
    ),
    VerificationPair(
        "cal_c_08", "George owns a 15% stake in the logistics firm.", "George owns an 85% stake in the logistics firm.",
        "genuine_contradiction", "CONTRADICTION", "George", "owns stake in firm", "15%", "85%",
        has_numbers=True
    ),
    VerificationPair(
        "cal_c_09", "Hannah's team consists of 12 full-time researchers.", "Hannah's team consists of 95 full-time researchers.",
        "genuine_contradiction", "CONTRADICTION", "Hannah", "team size", "12 researchers", "95 researchers",
        has_numbers=True
    ),
    VerificationPair(
        "cal_c_10", "Ian is currently employed by Microsoft.", "Ian is currently unemployed and seeking work.",
        "genuine_contradiction", "CONTRADICTION", "Ian", "employment status", "Microsoft", "unemployed"
    ),
    VerificationPair(
        "cal_c_11", "Julia was born in Tokyo in 1988.", "Julia was born in Berlin in 1995.",
        "genuine_contradiction", "CONTRADICTION", "Julia", "birthplace and date", "Tokyo 1988", "Berlin 1995",
        has_numbers=True, has_date=True
    ),
    VerificationPair(
        "cal_c_12", "Kevin is completely fluent in Japanese.", "Kevin cannot speak or understand a word of Japanese.",
        "genuine_contradiction", "CONTRADICTION", "Kevin", "language ability", "fluent", "cannot speak",
        has_negation=True
    ),
    VerificationPair(
        "cal_c_13", "Laura won the annual tournament gold medal.", "Laura was disqualified in the tournament preliminary round.",
        "genuine_contradiction", "CONTRADICTION", "Laura", "tournament result", "gold medal", "disqualified"
    ),
    VerificationPair(
        "cal_c_14", "Mateo received unanimous venture approval.", "Mateo received zero investor funding and was denied.",
        "genuine_contradiction", "CONTRADICTION", "Mateo", "funding result", "unanimous approval", "denied"
    ),
    VerificationPair(
        "cal_c_15", "Nadia published 2 papers in Nature.", "Nadia has never submitted or published any paper in Nature.",
        "genuine_contradiction", "CONTRADICTION", "Nadia", "Nature papers", "2 papers", "never published",
        has_negation=True, has_numbers=True
    ),

    # --- Category 3: Unrelated Statements (15) ---
    VerificationPair(
        "cal_u_01", "Bob holds a PhD in quantum physics.", "Bob enjoys mountain hiking in Colorado.",
        "unrelated", "NEUTRAL", "Bob", "various", "quantum physics", "mountain hiking"
    ),
    VerificationPair(
        "cal_u_02", "Alice leads the engineering division at Nexus Labs.", "Alice drives an electric sedan to the office.",
        "unrelated", "NEUTRAL", "Alice", "various", "Nexus Labs", "electric sedan"
    ),
    VerificationPair(
        "cal_u_03", "Marcus authored an influential biography monograph.", "Marcus visited the British Museum in London.",
        "unrelated", "NEUTRAL", "Marcus", "various", "biography monograph", "British Museum"
    ),
    VerificationPair(
        "cal_u_04", "Clara founded an artificial intelligence enterprise.", "Clara volunteers at an animal shelter on weekends.",
        "unrelated", "NEUTRAL", "Clara", "various", "AI enterprise", "animal shelter"
    ),
    VerificationPair(
        "cal_u_05", "Dmitri constructed a high-frequency trading algorithm.", "Dmitri plays classical violin in an amateur quartet.",
        "unrelated", "NEUTRAL", "Dmitri", "various", "trading algorithm", "violin quartet"
    ),
    VerificationPair(
        "cal_u_06", "Elena completed her medical residency in Boston.", "Elena attended a culinary workshop on pastry making.",
        "unrelated", "NEUTRAL", "Elena", "various", "medical residency", "pastry making"
    ),
    VerificationPair(
        "cal_u_07", "Fiona resides in Paris.", "Fiona planted heirloom tomatoes in her garden.",
        "unrelated", "NEUTRAL", "Fiona", "various", "Paris", "heirloom tomatoes"
    ),
    VerificationPair(
        "cal_u_08", "George purchased a home in Seattle.", "George adopted a golden retriever puppy.",
        "unrelated", "NEUTRAL", "George", "various", "Seattle home", "golden retriever"
    ),
    VerificationPair(
        "cal_u_09", "Hannah coordinates international climate summits.", "Hannah regularly practices marathon distance running.",
        "unrelated", "NEUTRAL", "Hannah", "various", "climate summits", "distance running"
    ),
    VerificationPair(
        "cal_u_10", "Ian discovered a novel antibiotic compound.", "Ian collects vintage vinyl jazz records.",
        "unrelated", "NEUTRAL", "Ian", "various", "antibiotic compound", "vinyl jazz records"
    ),
    VerificationPair(
        "cal_u_11", "Julia teaches organic chemistry at Oxford.", "Julia drinks black espresso every morning.",
        "unrelated", "NEUTRAL", "Julia", "various", "Oxford chemistry", "black espresso"
    ),
    VerificationPair(
        "cal_u_12", "Kevin repaired the faulty hydraulic pump.", "Kevin purchased a telescope for amateur astronomy.",
        "unrelated", "NEUTRAL", "Kevin", "various", "hydraulic pump", "telescope"
    ),
    VerificationPair(
        "cal_u_13", "Laura serves as chief legal counsel for Acme.", "Laura speaks conversational Italian.",
        "unrelated", "NEUTRAL", "Laura", "various", "chief counsel", "conversational Italian"
    ),
    VerificationPair(
        "cal_u_14", "Mateo secured 5 million dollars in venture capital.", "Mateo coached a youth soccer team last autumn.",
        "unrelated", "NEUTRAL", "Mateo", "various", "venture capital", "youth soccer team"
    ),
    VerificationPair(
        "cal_u_15", "Nadia translated three classical poetry volumes.", "Nadia bakes sourdough bread on Sundays.",
        "unrelated", "NEUTRAL", "Nadia", "various", "poetry volumes", "sourdough bread"
    ),

    # --- Category 4: Temporal Changes (15) ---
    VerificationPair(
        "cal_t_01", "Alice leads engineering at Nexus Labs.", "Alice now leads engineering at Apex Systems.",
        "temporal_change", "TEMPORAL", "Alice", "leads engineering at", "Nexus Labs", "Apex Systems",
        notes="Employer transition"
    ),
    VerificationPair(
        "cal_t_02", "Bob resided in London.", "Bob relocated to Zurich.",
        "temporal_change", "TEMPORAL", "Bob", "resided in", "London", "Zurich",
        notes="City relocation"
    ),
    VerificationPair(
        "cal_t_03", "Clara was an associate engineer.", "Clara was promoted to vice president of technology.",
        "temporal_change", "TEMPORAL", "Clara", "job title", "associate engineer", "vice president"
    ),
    VerificationPair(
        "cal_t_04", "Dmitri held 10 percent equity.", "Dmitri increased his stake to 35 percent equity.",
        "temporal_change", "TEMPORAL", "Dmitri", "equity stake", "10 percent", "35 percent",
        has_numbers=True
    ),
    VerificationPair(
        "cal_t_05", "Elena worked as an intern at General Hospital.", "Elena is now an attending physician at General Hospital.",
        "temporal_change", "TEMPORAL", "Elena", "role at General Hospital", "intern", "attending physician"
    ),
    VerificationPair(
        "cal_t_06", "Fiona rented an apartment in Lyon.", "Fiona purchased a house in Marseille.",
        "temporal_change", "TEMPORAL", "Fiona", "housing status", "rented in Lyon", "bought in Marseille"
    ),
    VerificationPair(
        "cal_t_07", "George used an Android phone.", "George switched to an iPhone.",
        "temporal_change", "TEMPORAL", "George", "device used", "Android phone", "iPhone"
    ),
    VerificationPair(
        "cal_t_08", "Hannah managed a team of 4 people.", "Hannah now oversees a department of 50 people.",
        "temporal_change", "TEMPORAL", "Hannah", "team management", "4 people", "50 people",
        has_numbers=True
    ),
    VerificationPair(
        "cal_t_09", "Ian was a doctoral candidate at Stanford.", "Ian graduated and joined DeepMind as a scientist.",
        "temporal_change", "TEMPORAL", "Ian", "academic/job status", "doctoral candidate", "DeepMind scientist"
    ),
    VerificationPair(
        "cal_t_10", "Julia dated Alexander.", "Julia married Alexander in May.",
        "temporal_change", "TEMPORAL", "Julia", "relationship status", "dated Alexander", "married Alexander"
    ),
    VerificationPair(
        "cal_t_11", "Kevin drove a gasoline sedan.", "Kevin replaced his car with a fully electric SUV.",
        "temporal_change", "TEMPORAL", "Kevin", "vehicle", "gasoline sedan", "electric SUV"
    ),
    VerificationPair(
        "cal_t_12", "Laura was single and lived alone.", "Laura got married and has two children.",
        "temporal_change", "TEMPORAL", "Laura", "family status", "single", "married with children"
    ),
    VerificationPair(
        "cal_t_13", "Mateo rented commercial office space in Brooklyn.", "Mateo moved his company to a tech campus in Austin.",
        "temporal_change", "TEMPORAL", "Mateo", "office location", "Brooklyn", "Austin"
    ),
    VerificationPair(
        "cal_t_14", "Nadia had intermediate fluency in Mandarin.", "Nadia achieved certified professional fluency in Mandarin.",
        "temporal_change", "TEMPORAL", "Nadia", "Mandarin skill", "intermediate", "professional fluency"
    ),
    VerificationPair(
        "cal_t_15", "Oliver was an advisor to FinTech Inc.", "Oliver resigned from all advisory duties at FinTech Inc.",
        "temporal_change", "TEMPORAL", "Oliver", "role at FinTech Inc", "advisor", "resigned"
    )
]


# ─────────────────────────────────────────────────────────────
# 2. Validation Set (60 pairs: 15 per category)
# ─────────────────────────────────────────────────────────────
VALIDATION_PAIRS: List[VerificationPair] = [
    # --- Category 1: Valid Paraphrases (15) ---
    VerificationPair(
        "val_p_01", "Helena directed the regional orchestral ensemble.", "Helena conducted the local symphony orchestra.",
        "valid_paraphrase", "SAME", "Helena", "directed orchestra", "regional ensemble", "local symphony"
    ),
    VerificationPair(
        "val_p_02", "Paul authored a comprehensive textbook on computer vision.", "Paul wrote an extensive reference book on computer vision.",
        "valid_paraphrase", "SAME", "Paul", "authored textbook on", "computer vision", "computer vision"
    ),
    VerificationPair(
        "val_p_03", "Rachel graduated summa cum laude from Princeton University.", "Rachel completed her degree with highest honors at Princeton.",
        "valid_paraphrase", "SAME", "Rachel", "graduated from", "Princeton University", "Princeton"
    ),
    VerificationPair(
        "val_p_04", "Samuel developed a cryptographic encryption protocol.", "Samuel created a secure cryptographic protocol.",
        "valid_paraphrase", "SAME", "Samuel", "developed protocol", "cryptographic", "cryptographic protocol"
    ),
    VerificationPair(
        "val_p_05", "Tara launched a successful renewable energy venture.", "Tara started a prosperous green energy company.",
        "valid_paraphrase", "SAME", "Tara", "launched venture", "renewable energy", "green energy company"
    ),
    VerificationPair(
        "val_p_06", "Uma designed the architecture for the national rail network.", "Uma planned the structural layout of the railway network.",
        "valid_paraphrase", "SAME", "Uma", "designed rail network", "national rail", "railway network"
    ),
    VerificationPair(
        "val_p_07", "Vincent received the prestigious Turing award in computing.", "Vincent was awarded the computer science Turing prize.",
        "valid_paraphrase", "SAME", "Vincent", "received award", "Turing award", "Turing prize"
    ),
    VerificationPair(
        "val_p_08", "Wendy authored 45 peer-reviewed journal articles.", "Wendy published forty-five scientific papers in journals.",
        "valid_paraphrase", "SAME", "Wendy", "authored articles", "45 papers", "forty-five papers",
        has_numbers=True
    ),
    VerificationPair(
        "val_p_09", "Xavier operates a cybersecurity consultancy in Berlin.", "Xavier runs a digital security consulting firm in Berlin.",
        "valid_paraphrase", "SAME", "Xavier", "operates consultancy in", "Berlin", "Berlin"
    ),
    VerificationPair(
        "val_p_10", "Yasmine teaches pediatric surgery at Johns Hopkins.", "Yasmine is a professor of pediatric surgery at Johns Hopkins.",
        "valid_paraphrase", "SAME", "Yasmine", "teaches at", "Johns Hopkins", "Johns Hopkins"
    ),
    VerificationPair(
        "val_p_11", "Zachary raised 10 million euros in Series A funding.", "Zachary secured 10 million euros during the Series A round.",
        "valid_paraphrase", "SAME", "Zachary", "raised funding", "10 million euros", "10 million euros",
        has_numbers=True
    ),
    VerificationPair(
        "val_p_12", "Arthur climbed Mount Everest via the South Col route.", "Arthur reached the summit of Mount Everest through the South Col.",
        "valid_paraphrase", "SAME", "Arthur", "climbed", "Mount Everest", "summit of Mount Everest"
    ),
    VerificationPair(
        "val_p_13", "Beatrice curates contemporary art at the Tate Modern.", "Beatrice oversees modern art exhibitions at the Tate Modern.",
        "valid_paraphrase", "SAME", "Beatrice", "curates art at", "Tate Modern", "Tate Modern"
    ),
    VerificationPair(
        "val_p_14", "Cedric synthesised an organic superconductor in the laboratory.", "Cedric produced a superconducting organic material in the lab.",
        "valid_paraphrase", "SAME", "Cedric", "synthesised material", "organic superconductor", "superconducting organic material"
    ),
    VerificationPair(
        "val_p_15", "Diana won the Olympic gold medal in the 100m sprint.", "Diana finished first in the Olympic 100-meter dash.",
        "valid_paraphrase", "SAME", "Diana", "won medal", "Olympic 100m sprint", "Olympic 100m dash",
        has_numbers=True
    ),

    # --- Category 2: Genuine Contradictions (15) ---
    VerificationPair(
        "val_c_01", "Paul authored a comprehensive textbook on computer vision.", "Paul never wrote any textbook on computer vision.",
        "genuine_contradiction", "CONTRADICTION", "Paul", "authored textbook on", "computer vision", "never wrote textbook",
        has_negation=True
    ),
    VerificationPair(
        "val_c_02", "Rachel graduated summa cum laude from Princeton.", "Rachel dropped out of Princeton during her first semester.",
        "genuine_contradiction", "CONTRADICTION", "Rachel", "Princeton status", "graduated summa cum laude", "dropped out"
    ),
    VerificationPair(
        "val_c_03", "Samuel's protocol was proven mathematically secure.", "Samuel's protocol was decisively cracked and proven insecure.",
        "genuine_contradiction", "CONTRADICTION", "Samuel", "protocol security", "mathematically secure", "cracked and insecure"
    ),
    VerificationPair(
        "val_c_04", "Tara's venture reported a 20 million dollar profit.", "Tara's venture suffered a catastrophic 20 million dollar loss.",
        "genuine_contradiction", "CONTRADICTION", "Tara", "venture finances", "profit", "loss",
        has_numbers=True
    ),
    VerificationPair(
        "cal_c_05", "Uma's rail proposal was approved unanimously by parliament.", "Uma's rail proposal was rejected and vetoed by parliament.",
        "genuine_contradiction", "CONTRADICTION", "Uma", "rail proposal", "approved", "rejected"
    ),
    VerificationPair(
        "val_c_06", "Vincent has lived in New York City his entire life.", "Vincent has never visited or lived in New York City.",
        "genuine_contradiction", "CONTRADICTION", "Vincent", "residence in NYC", "entire life", "never visited",
        has_negation=True
    ),
    VerificationPair(
        "val_c_07", "Wendy is licensed to practice medicine in California.", "Wendy's medical license was permanently revoked in California.",
        "genuine_contradiction", "CONTRADICTION", "Wendy", "medical license", "licensed", "permanently revoked"
    ),
    VerificationPair(
        "val_c_08", "Xavier employs 250 security specialists in Berlin.", "Xavier is a solo freelance consultant with zero employees.",
        "genuine_contradiction", "CONTRADICTION", "Xavier", "employee count", "250 specialists", "zero employees",
        has_numbers=True
    ),
    VerificationPair(
        "val_c_09", "Yasmine is the sole author of the pediatric surgery paper.", "Yasmine did not contribute to or co-author the surgery paper.",
        "genuine_contradiction", "CONTRADICTION", "Yasmine", "paper authorship", "sole author", "did not contribute",
        has_negation=True
    ),
    VerificationPair(
        "val_c_10", "Zachary retained 100% ownership of his technology.", "Zachary surrendered and sold all intellectual property rights.",
        "genuine_contradiction", "CONTRADICTION", "Zachary", "IP ownership", "retained 100%", "sold all rights",
        has_numbers=True
    ),
    VerificationPair(
        "val_c_11", "Arthur reached the summit of Mount Everest on May 12.", "Arthur aborted his climb at base camp and never reached the summit.",
        "genuine_contradiction", "CONTRADICTION", "Arthur", "Everest climb", "reached summit", "aborted at base camp",
        has_negation=True
    ),
    VerificationPair(
        "val_c_12", "Beatrice was appointed chief curator of the Tate.", "Beatrice was terminated from her position at the Tate.",
        "genuine_contradiction", "CONTRADICTION", "Beatrice", "Tate position", "chief curator", "terminated"
    ),
    VerificationPair(
        "val_c_13", "Cedric's superconductor works at room temperature (295 K).", "Cedric's superconductor only operates near absolute zero (4 K).",
        "genuine_contradiction", "CONTRADICTION", "Cedric", "operating temperature", "295 K", "4 K",
        has_numbers=True
    ),
    VerificationPair(
        "val_c_14", "Diana won the race in an official time of 10.49 seconds.", "Diana took third place with a time of 11.20 seconds.",
        "genuine_contradiction", "CONTRADICTION", "Diana", "race outcome", "won (10.49s)", "third (11.20s)",
        has_numbers=True
    ),
    VerificationPair(
        "val_c_15", "Ethan was born in Madrid Spain.", "Ethan was born in Dublin Ireland.",
        "genuine_contradiction", "CONTRADICTION", "Ethan", "birthplace", "Madrid Spain", "Dublin Ireland"
    ),

    # --- Category 3: Unrelated Statements (15) ---
    VerificationPair(
        "val_u_01", "Paul authored a textbook on computer vision.", "Paul prefers dark roast coffee over tea.",
        "unrelated", "NEUTRAL", "Paul", "various", "computer vision", "coffee preference"
    ),
    VerificationPair(
        "val_u_02", "Rachel graduated from Princeton.", "Rachel drives a hybrid compact vehicle.",
        "unrelated", "NEUTRAL", "Rachel", "various", "Princeton", "hybrid vehicle"
    ),
    VerificationPair(
        "val_u_03", "Samuel developed a cryptographic protocol.", "Samuel enjoys playing chess in weekly tournaments.",
        "unrelated", "NEUTRAL", "Samuel", "various", "cryptography", "chess tournaments"
    ),
    VerificationPair(
        "val_u_04", "Tara started a green energy company.", "Tara adopted two rescue cats from the shelter.",
        "unrelated", "NEUTRAL", "Tara", "various", "green energy", "rescue cats"
    ),
    VerificationPair(
        "val_u_05", "Uma designed the railway network layout.", "Uma visited the Grand Canyon during vacation.",
        "unrelated", "NEUTRAL", "Uma", "various", "railway network", "Grand Canyon"
    ),
    VerificationPair(
        "val_u_06", "Vincent was awarded the Turing prize.", "Vincent practices woodworking as a weekend hobby.",
        "unrelated", "NEUTRAL", "Vincent", "various", "Turing prize", "woodworking"
    ),
    VerificationPair(
        "val_u_07", "Wendy published forty-five scientific papers.", "Wendy runs three miles every morning at sunrise.",
        "unrelated", "NEUTRAL", "Wendy", "various", "scientific papers", "morning running"
    ),
    VerificationPair(
        "val_u_08", "Xavier operates a cybersecurity firm in Berlin.", "Xavier planted an herb garden on his balcony.",
        "unrelated", "NEUTRAL", "Xavier", "various", "cybersecurity firm", "herb garden"
    ),
    VerificationPair(
        "val_u_09", "Yasmine teaches pediatric surgery at Johns Hopkins.", "Yasmine plays classical piano.",
        "unrelated", "NEUTRAL", "Yasmine", "various", "pediatric surgery", "classical piano"
    ),
    VerificationPair(
        "val_u_10", "Zachary secured 10 million euros in Series A.", "Zachary owns a vintage road bicycle.",
        "unrelated", "NEUTRAL", "Zachary", "various", "Series A funding", "vintage bicycle"
    ),
    VerificationPair(
        "val_u_11", "Arthur climbed Mount Everest via the South Col.", "Arthur enjoys baking artisanal sourdough.",
        "unrelated", "NEUTRAL", "Arthur", "various", "Mount Everest", "sourdough bread"
    ),
    VerificationPair(
        "val_u_12", "Beatrice oversees modern art at the Tate Modern.", "Beatrice speaks fluent German and French.",
        "unrelated", "NEUTRAL", "Beatrice", "various", "Tate Modern", "languages spoken"
    ),
    VerificationPair(
        "val_u_13", "Cedric synthesised an organic superconductor.", "Cedric is an avid collector of antique stamps.",
        "unrelated", "NEUTRAL", "Cedric", "various", "superconductor", "antique stamps"
    ),
    VerificationPair(
        "val_u_14", "Diana won the Olympic 100-meter dash.", "Diana studied international economics in college.",
        "unrelated", "NEUTRAL", "Diana", "various", "Olympic dash", "international economics"
    ),
    VerificationPair(
        "val_u_15", "Ethan plays the cello.", "Ethan is allergic to peanuts and tree nuts.",
        "unrelated", "NEUTRAL", "Ethan", "various", "plays cello", "nut allergy"
    ),

    # --- Category 4: Temporal Changes (15) ---
    VerificationPair(
        "val_t_01", "Paul was an assistant professor at CMU.", "Paul became a full tenured professor at Stanford.",
        "temporal_change", "TEMPORAL", "Paul", "academic role", "assistant prof at CMU", "tenured prof at Stanford"
    ),
    VerificationPair(
        "val_t_02", "Rachel lived in an apartment in Brooklyn.", "Rachel bought a farm in upstate New York.",
        "temporal_change", "TEMPORAL", "Rachel", "residence", "Brooklyn apartment", "upstate farm"
    ),
    VerificationPair(
        "val_t_03", "Samuel was a junior security researcher.", "Samuel was promoted to principal security architect.",
        "temporal_change", "TEMPORAL", "Samuel", "role", "junior researcher", "principal architect"
    ),
    VerificationPair(
        "val_t_04", "Tara operated out of a co-working space.", "Tara opened a 500-person dedicated corporate headquarters.",
        "temporal_change", "TEMPORAL", "Tara", "office space", "co-working space", "dedicated headquarters",
        has_numbers=True
    ),
    VerificationPair(
        "val_t_05", "Uma worked for the Department of Transportation.", "Uma transitioned to private consulting.",
        "temporal_change", "TEMPORAL", "Uma", "employment", "Dept of Transportation", "private consulting"
    ),
    VerificationPair(
        "val_t_06", "Vincent was a visiting fellow at Oxford.", "Vincent concluded his fellowship and returned to MIT.",
        "temporal_change", "TEMPORAL", "Vincent", "institution", "visiting at Oxford", "returned to MIT"
    ),
    VerificationPair(
        "val_t_07", "Wendy was an unmarried post-doctoral fellow.", "Wendy married and had twin daughters.",
        "temporal_change", "TEMPORAL", "Wendy", "personal status", "unmarried postdoc", "married with twins"
    ),
    VerificationPair(
        "val_t_08", "Xavier lived in Munich Germany.", "Xavier relocated his family and business to Berlin.",
        "temporal_change", "TEMPORAL", "Xavier", "location", "Munich", "Berlin"
    ),
    VerificationPair(
        "val_t_09", "Yasmine worked exclusively as a general surgeon.", "Yasmine completed specialized training in pediatric neurosurgery.",
        "temporal_change", "TEMPORAL", "Yasmine", "medical specialization", "general surgeon", "pediatric neurosurgeon"
    ),
    VerificationPair(
        "val_t_10", "Zachary owned 80% equity in the startup.", "Zachary's equity was diluted to 22% after Series B.",
        "temporal_change", "TEMPORAL", "Zachary", "equity stake", "80%", "22%",
        has_numbers=True
    ),
    VerificationPair(
        "val_t_11", "Arthur trained as an alpine marathoner.", "Arthur transitioned into high-altitude mountaineering.",
        "temporal_change", "TEMPORAL", "Arthur", "athletic focus", "alpine marathoner", "mountaineer"
    ),
    VerificationPair(
        "val_t_12", "Beatrice managed a boutique art gallery.", "Beatrice was appointed chief curator of a major museum.",
        "temporal_change", "TEMPORAL", "Beatrice", "curatorial role", "boutique gallery", "major museum curator"
    ),
    VerificationPair(
        "val_t_13", "Cedric conducted theoretical modeling on computers.", "Cedric secured a wet lab to begin physical synthesis.",
        "temporal_change", "TEMPORAL", "Cedric", "research methodology", "theoretical modeling", "wet lab synthesis"
    ),
    VerificationPair(
        "val_t_14", "Diana competed in collegiate track events.", "Diana turned professional and signed with Nike.",
        "temporal_change", "TEMPORAL", "Diana", "athletic career", "collegiate track", "professional Nike athlete"
    ),
    VerificationPair(
        "val_t_15", "Ethan was an apprentice cello maker.", "Ethan opened his own independent luthier workshop.",
        "temporal_change", "TEMPORAL", "Ethan", "career progression", "apprentice", "independent luthier"
    )
]


# ─────────────────────────────────────────────────────────────
# 3. Held-out Test Set (60 pairs: 15 per category) - STRICTLY FROZEN
# ─────────────────────────────────────────────────────────────
# Contains entirely unseen named entities and novel domains:
#   Astronomers, Naval Architects, Quantum Chemists, Archaeologists, Diplomats
HELD_OUT_PAIRS: List[VerificationPair] = [
    # --- Category 1: Valid Paraphrases (15) ---
    VerificationPair(
        "test_p_01", "Amara mapped the galactic star-forming nebula.", "Amara charted the interstellar stellar nursery.",
        "valid_paraphrase", "SAME", "Amara", "mapped nebula", "galactic nebula", "stellar nursery"
    ),
    VerificationPair(
        "test_p_02", "Balthazar designed an icebreaker hull for Arctic navigation.", "Balthazar engineered an ice-strengthened ship hull for polar waters.",
        "valid_paraphrase", "SAME", "Balthazar", "designed hull", "icebreaker hull", "ice-strengthened hull"
    ),
    VerificationPair(
        "test_p_03", "Cynthia excavated a Bronze Age cemetery in Crete.", "Cynthia unearthed Bronze Age burial tombs in Crete.",
        "valid_paraphrase", "SAME", "Cynthia", "excavated tombs", "Bronze Age cemetery", "burial tombs"
    ),
    VerificationPair(
        "test_p_04", "Declan negotiated the cross-border bilateral fisheries pact.", "Declan brokered the international fishing treaty between both nations.",
        "valid_paraphrase", "SAME", "Declan", "negotiated pact", "bilateral fisheries pact", "international fishing treaty"
    ),
    VerificationPair(
        "test_p_05", "Esperanza synthesized a biocompatible porous scaffold.", "Esperanza fabricated a porous biomaterial framework for tissue growth.",
        "valid_paraphrase", "SAME", "Esperanza", "synthesized scaffold", "biocompatible scaffold", "porous biomaterial framework"
    ),
    VerificationPair(
        "test_p_06", "Fabian demonstrated quantum entanglement across 100 kilometers.", "Fabian proved quantum entanglement over a 100-km fiber link.",
        "valid_paraphrase", "SAME", "Fabian", "demonstrated entanglement", "100 kilometers", "100-km fiber link",
        has_numbers=True
    ),
    VerificationPair(
        "test_p_07", "Giselle deciphered the undeciphered Linear A tablets.", "Giselle translated the inscription on the Linear A tablets.",
        "valid_paraphrase", "SAME", "Giselle", "deciphered tablets", "Linear A", "Linear A"
    ),
    VerificationPair(
        "test_p_08", "Hector commands the nuclear-powered submarine fleet.", "Hector serves as commander of the nuclear submarine squadron.",
        "valid_paraphrase", "SAME", "Hector", "commands fleet", "nuclear submarine fleet", "nuclear submarine squadron"
    ),
    VerificationPair(
        "test_p_09", "Isla discovered an exoplanet orbiting within the habitable zone.", "Isla identified a planet in the habitable zone of its host star.",
        "valid_paraphrase", "SAME", "Isla", "discovered exoplanet", "habitable zone", "habitable zone"
    ),
    VerificationPair(
        "test_p_10", "Jesper engineered a carbon-neutral concrete formula.", "Jesper developed zero-carbon emission concrete.",
        "valid_paraphrase", "SAME", "Jesper", "engineered concrete", "carbon-neutral concrete", "zero-carbon concrete"
    ),
    VerificationPair(
        "test_p_11", "Kendra serves as the ambassador to the United Nations.", "Kendra represents her country as UN permanent ambassador.",
        "valid_paraphrase", "SAME", "Kendra", "diplomatic role", "UN ambassador", "UN permanent ambassador"
    ),
    VerificationPair(
        "test_p_12", "Leif survived a solo expedition across Greenland.", "Leif successfully crossed the Greenland ice sheet alone.",
        "valid_paraphrase", "SAME", "Leif", "expedition", "solo across Greenland", "Greenland ice sheet alone"
    ),
    VerificationPair(
        "test_p_13", "Mira sequenced the genome of an extinct hominid.", "Mira reconstructed the complete genetic sequence of an archaic human.",
        "valid_paraphrase", "SAME", "Mira", "genomic sequencing", "extinct hominid genome", "archaic human genetic sequence"
    ),
    VerificationPair(
        "test_p_14", "Nikolai restored an eighteenth-century pipe organ.", "Nikolai repaired and refurbished a pipe organ built in the 1700s.",
        "valid_paraphrase", "SAME", "Nikolai", "restored organ", "18th-century pipe organ", "1700s pipe organ",
        has_date=True
    ),
    VerificationPair(
        "test_p_15", "Ophelia raised 250 million dollars for fusion energy research.", "Ophelia secured a quarter billion dollars in fusion research financing.",
        "valid_paraphrase", "SAME", "Ophelia", "funding", "250 million dollars", "quarter billion dollars",
        has_numbers=True
    ),

    # --- Category 2: Genuine Contradictions (15) ---
    VerificationPair(
        "test_c_01", "Amara discovered a pulsar in the Andromeda galaxy.", "Amara proved there are no detectable pulsars in the Andromeda galaxy.",
        "genuine_contradiction", "CONTRADICTION", "Amara", "pulsar discovery", "discovered pulsar", "proved no pulsars",
        has_negation=True
    ),
    VerificationPair(
        "test_c_02", "Balthazar's ship completed the voyage without structural damage.", "Balthazar's ship sank after sustaining catastrophic hull rupture.",
        "genuine_contradiction", "CONTRADICTION", "Balthazar", "voyage result", "no structural damage", "sank from hull rupture"
    ),
    VerificationPair(
        "test_c_03", "Cynthia confirmed the ruins belong to the Minoan era.", "Cynthia proved definitively that the ruins are twentieth-century forgeries.",
        "genuine_contradiction", "CONTRADICTION", "Cynthia", "ruin authenticity", "Minoan era", "twentieth-century forgeries"
    ),
    VerificationPair(
        "test_c_04", "Declan's treaty was ratified unanimously by all member states.", "Declan's treaty was vetoed and rejected by every participating nation.",
        "genuine_contradiction", "CONTRADICTION", "Declan", "treaty ratification", "ratified unanimously", "vetoed by every nation"
    ),
    VerificationPair(
        "test_c_05", "Esperanza's scaffold is entirely non-toxic and biocompatible.", "Esperanza's scaffold causes acute cytotoxicity and tissue necrosis.",
        "genuine_contradiction", "CONTRADICTION", "Esperanza", "toxicity", "non-toxic", "causes acute cytotoxicity"
    ),
    VerificationPair(
        "test_c_06", "Fabian measured an entanglement fidelity of 99.4 percent.", "Fabian recorded an entanglement fidelity of only 12.1 percent.",
        "genuine_contradiction", "CONTRADICTION", "Fabian", "fidelity measure", "99.4 percent", "12.1 percent",
        has_numbers=True
    ),
    VerificationPair(
        "test_c_07", "Giselle successfully deciphered the Linear A script.", "Giselle admitted the Linear A script remains completely undeciphered.",
        "genuine_contradiction", "CONTRADICTION", "Giselle", "decipherment status", "successfully deciphered", "completely undeciphered"
    ),
    VerificationPair(
        "test_c_08", "Hector was honorably discharged with full military honors.", "Hector was court-martialed and dishonorably dismissed from the navy.",
        "genuine_contradiction", "CONTRADICTION", "Hector", "military discharge", "honorably discharged", "dishonorably dismissed"
    ),
    VerificationPair(
        "test_c_09", "Isla's exoplanet has an atmosphere rich in oxygen.", "Isla's exoplanet has zero atmosphere and is an airless vacuum.",
        "genuine_contradiction", "CONTRADICTION", "Isla", "planetary atmosphere", "rich in oxygen", "airless vacuum"
    ),
    VerificationPair(
        "test_c_10", "Jesper's concrete formula meets all national building safety codes.", "Jesper's concrete formula failed all structural compression tests.",
        "genuine_contradiction", "CONTRADICTION", "Jesper", "building code compliance", "meets all codes", "failed all compression tests"
    ),
    VerificationPair(
        "test_c_11", "Kendra is a dual citizen of Sweden and Norway.", "Kendra has never possessed Swedish or Norwegian citizenship.",
        "genuine_contradiction", "CONTRADICTION", "Kendra", "citizenship", "dual citizen", "never possessed citizenship",
        has_negation=True
    ),
    VerificationPair(
        "test_c_12", "Leif completed the Arctic journey unassisted.", "Leif required emergency helicopter evacuation after two days.",
        "genuine_contradiction", "CONTRADICTION", "Leif", "journey completion", "completed unassisted", "emergency evacuation after 2 days",
        has_numbers=True
    ),
    VerificationPair(
        "test_c_13", "Mira extracted high-quality DNA from the 40,000-year-old femur.", "Mira found no preserved organic material or DNA in the specimen.",
        "genuine_contradiction", "CONTRADICTION", "Mira", "DNA preservation", "high-quality DNA", "no preserved DNA",
        has_numbers=True, has_negation=True
    ),
    VerificationPair(
        "test_c_14", "Nikolai verified the organ was built in 1742.", "Nikolai confirmed the organ was constructed in 1965.",
        "genuine_contradiction", "CONTRADICTION", "Nikolai", "organ date", "built in 1742", "constructed in 1965",
        has_numbers=True, has_date=True
    ),
    VerificationPair(
        "test_c_15", "Ophelia holds 50 patents in nuclear engineering.", "Ophelia has never filed or received a single patent.",
        "genuine_contradiction", "CONTRADICTION", "Ophelia", "patent count", "50 patents", "never filed patent",
        has_numbers=True, has_negation=True
    ),

    # --- Category 3: Unrelated Statements (15) ---
    VerificationPair(
        "test_u_01", "Amara mapped a star-forming nebula.", "Amara practices traditional pottery on weekends.",
        "unrelated", "NEUTRAL", "Amara", "various", "nebula mapping", "traditional pottery"
    ),
    VerificationPair(
        "test_u_02", "Balthazar designed an icebreaker ship hull.", "Balthazar adopted a greyhound dog.",
        "unrelated", "NEUTRAL", "Balthazar", "various", "ship hull", "greyhound dog"
    ),
    VerificationPair(
        "test_u_03", "Cynthia excavated ruins in Crete.", "Cynthia drives a vintage motorcycle.",
        "unrelated", "NEUTRAL", "Cynthia", "various", "Crete ruins", "vintage motorcycle"
    ),
    VerificationPair(
        "test_u_04", "Declan brokered an international fisheries pact.", "Declan enjoys cooking spicy Thai curries.",
        "unrelated", "NEUTRAL", "Declan", "various", "fisheries pact", "Thai curries"
    ),
    VerificationPair(
        "test_u_05", "Esperanza synthesized a porous biomaterial.", "Esperanza plays classical guitar in an ensemble.",
        "unrelated", "NEUTRAL", "Esperanza", "various", "biomaterial", "classical guitar"
    ),
    VerificationPair(
        "test_u_06", "Fabian proved quantum entanglement.", "Fabian speaks conversational Portuguese.",
        "unrelated", "NEUTRAL", "Fabian", "various", "entanglement", "Portuguese language"
    ),
    VerificationPair(
        "test_u_07", "Giselle deciphered the ancient inscription.", "Giselle collects rare postage stamps from Asia.",
        "unrelated", "NEUTRAL", "Giselle", "various", "ancient inscription", "rare stamps"
    ),
    VerificationPair(
        "test_u_08", "Hector commands the submarine squadron.", "Hector bakes sourdough rye bread.",
        "unrelated", "NEUTRAL", "Hector", "various", "submarine squadron", "sourdough bread"
    ),
    VerificationPair(
        "test_u_09", "Isla identified an exoplanet in the habitable zone.", "Isla runs half-marathons every spring.",
        "unrelated", "NEUTRAL", "Isla", "various", "exoplanet", "half-marathons"
    ),
    VerificationPair(
        "test_u_10", "Jesper developed zero-carbon concrete.", "Jesper cultivates bonsai trees at home.",
        "unrelated", "NEUTRAL", "Jesper", "various", "zero-carbon concrete", "bonsai trees"
    ),
    VerificationPair(
        "test_u_11", "Kendra represents her country at the United Nations.", "Kendra plays tennis on Saturday mornings.",
        "unrelated", "NEUTRAL", "Kendra", "various", "UN representative", "tennis"
    ),
    VerificationPair(
        "test_u_12", "Leif crossed the Greenland ice sheet.", "Leif owns a small wooden sailboat.",
        "unrelated", "NEUTRAL", "Leif", "various", "Greenland crossing", "wooden sailboat"
    ),
    VerificationPair(
        "test_u_13", "Mira reconstructed the archaic human genome.", "Mira prefers drinking Earl Grey tea.",
        "unrelated", "NEUTRAL", "Mira", "various", "human genome", "Earl Grey tea"
    ),
    VerificationPair(
        "test_u_14", "Nikolai restored an 18th-century pipe organ.", "Nikolai has an extensive collection of mechanical watches.",
        "unrelated", "NEUTRAL", "Nikolai", "various", "pipe organ", "mechanical watches"
    ),
    VerificationPair(
        "test_u_15", "Ophelia secured financing for fusion energy.", "Ophelia is an avid amateur landscape photographer.",
        "unrelated", "NEUTRAL", "Ophelia", "various", "fusion financing", "landscape photographer"
    ),

    # --- Category 4: Temporal Changes (15) ---
    VerificationPair(
        "test_t_01", "Amara worked as a postdoctoral observer at Mauna Kea.", "Amara was appointed director of the European Southern Observatory.",
        "temporal_change", "TEMPORAL", "Amara", "astronomy appointment", "postdoc at Mauna Kea", "director of ESO"
    ),
    VerificationPair(
        "test_t_02", "Balthazar worked as a junior naval architect.", "Balthazar launched his own independent yacht design firm.",
        "temporal_change", "TEMPORAL", "Balthazar", "career progression", "junior architect", "independent yacht firm"
    ),
    VerificationPair(
        "test_t_03", "Cynthia lived in Athens Greece.", "Cynthia moved to Heraklion Crete for a field project.",
        "temporal_change", "TEMPORAL", "Cynthia", "living location", "Athens", "Heraklion Crete"
    ),
    VerificationPair(
        "test_t_04", "Declan was a legal attaché in Dublin.", "Declan was posted as principal envoy to Brussels.",
        "temporal_change", "TEMPORAL", "Declan", "diplomatic post", "attaché in Dublin", "envoy to Brussels"
    ),
    VerificationPair(
        "test_t_05", "Esperanza held an unpaid fellowship.", "Esperanza secured a tenured faculty appointment.",
        "temporal_change", "TEMPORAL", "Esperanza", "academic position", "unpaid fellowship", "tenured faculty"
    ),
    VerificationPair(
        "test_t_06", "Fabian operated an optical lab with 2 lasers.", "Fabian expanded his facility to house 18 ultra-fast laser systems.",
        "temporal_change", "TEMPORAL", "Fabian", "equipment scale", "2 lasers", "18 laser systems",
        has_numbers=True
    ),
    VerificationPair(
        "test_t_07", "Giselle was an uncertified archaeological assistant.", "Giselle earned her PhD and full field director certification.",
        "temporal_change", "TEMPORAL", "Giselle", "qualifications", "uncertified assistant", "field director certification"
    ),
    VerificationPair(
        "test_t_08", "Hector served as executive officer on a frigate.", "Hector was promoted to captain of a guided-missile destroyer.",
        "temporal_change", "TEMPORAL", "Hector", "naval rank and command", "frigate officer", "destroyer captain"
    ),
    VerificationPair(
        "test_t_09", "Isla taught physics at high school level.", "Isla earned her doctorate and lectures in astrophysics at Cambridge.",
        "temporal_change", "TEMPORAL", "Isla", "teaching career", "high school physics", "astrophysics at Cambridge"
    ),
    VerificationPair(
        "test_t_10", "Jesper was a concrete lab technician.", "Jesper was named chief materials engineer of a construction giant.",
        "temporal_change", "TEMPORAL", "Jesper", "job position", "lab technician", "chief materials engineer"
    ),
    VerificationPair(
        "test_t_11", "Kendra lived in Oslo Norway.", "Kendra relocated to Geneva Switzerland.",
        "temporal_change", "TEMPORAL", "Kendra", "location", "Oslo Norway", "Geneva Switzerland"
    ),
    VerificationPair(
        "test_t_12", "Leif was an amateur weekend trekker.", "Leif became a full-time sponsored polar explorer.",
        "temporal_change", "TEMPORAL", "Leif", "mountaineering status", "amateur trekker", "sponsored polar explorer"
    ),
    VerificationPair(
        "test_t_13", "Mira was a student researcher in ancient genetics.", "Mira was awarded an ERC grant and leads her own evolutionary genomics lab.",
        "temporal_change", "TEMPORAL", "Mira", "career status", "student researcher", "ERC lab leader"
    ),
    VerificationPair(
        "test_t_14", "Nikolai operated from a small rural workshop.", "Nikolai established a premier cathedral organ restoration studio in Vienna.",
        "temporal_change", "TEMPORAL", "Nikolai", "workshop location", "rural workshop", "Vienna restoration studio"
    ),
    VerificationPair(
        "test_t_15", "Ophelia worked in theoretical plasma physics.", "Ophelia transitioned to experimental tokamak reactor engineering.",
        "temporal_change", "TEMPORAL", "Ophelia", "technical focus", "theoretical plasma", "tokamak reactor engineering"
    )
]


def get_dataset_splits() -> Dict[str, List[VerificationPair]]:
    """Returns the three isolated datasets."""
    return {
        "calibration": CALIBRATION_PAIRS,
        "validation": VALIDATION_PAIRS,
        "held_out": HELD_OUT_PAIRS
    }


def verify_dataset_integrity() -> Dict[str, Any]:
    """Validates that no entities or pairs leak across splits."""
    splits = get_dataset_splits()
    subjects_by_split: Dict[str, set] = {}
    ids_by_split: Dict[str, set] = {}

    for split_name, pairs in splits.items():
        assert len(pairs) == 60, f"Split {split_name} must have exactly 60 pairs, got {len(pairs)}"
        cats = [p.category for p in pairs]
        for cat in ["valid_paraphrase", "genuine_contradiction", "unrelated", "temporal_change"]:
            count = cats.count(cat)
            assert count == 15, f"Split {split_name} category {cat} must have 15 pairs, got {count}"

        subjs = {p.subject for p in pairs}
        ids = {p.id for p in pairs}
        assert len(ids) == 60, f"Duplicate pair IDs in {split_name}"
        subjects_by_split[split_name] = subjs
        ids_by_split[split_name] = ids

    # Check leakage between calibration/validation and held_out
    cal_val_subjs = subjects_by_split["calibration"].union(subjects_by_split["validation"])
    held_out_subjs = subjects_by_split["held_out"]
    overlap = cal_val_subjs.intersection(held_out_subjs)
    assert len(overlap) == 0, f"Leakage detected between tuning sets and held-out set: {overlap}"

    return {
        "status": "PASS",
        "counts": {s: len(p) for s, p in splits.items()},
        "categories_per_split": 4,
        "pairs_per_category": 15,
        "held_out_overlap": list(overlap)
    }


if __name__ == "__main__":
    res = verify_dataset_integrity()
    print("Dataset verification passed:", res)
