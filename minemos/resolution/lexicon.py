"""Deterministic lexicons used by the resolution stages.

Everything here is static data with no learned component, so stage output is
reproducible across processes.  The vocabulary is general-purpose English plus
memory/attribute words; it is deliberately not tuned to a specific corpus.
"""

from __future__ import annotations

import re

# ── normalization ────────────────────────────────────────────────────────── #

CONTRACTIONS = {
    "i'm": "i am",
    "im": "i am",
    "ive": "i have",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "you're": "you are",
    "youre": "you are",
    "you've": "you have",
    "you'll": "you will",
    "it's": "it is",
    "its": "it is",
    "that's": "that is",
    "what's": "what is",
    "whats": "what is",
    "who's": "who is",
    "where's": "where is",
    "when's": "when is",
    "how's": "how is",
    "there's": "there is",
    "he's": "he is",
    "she's": "she is",
    "they're": "they are",
    "theyre": "they are",
    "we're": "we are",
    "we've": "we have",
    "don't": "do not",
    "dont": "do not",
    "doesn't": "does not",
    "doesnt": "does not",
    "didn't": "did not",
    "didnt": "did not",
    "isn't": "is not",
    "isnt": "is not",
    "aren't": "are not",
    "arent": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "can't": "cannot",
    "cant": "cannot",
    "couldn't": "could not",
    "won't": "will not",
    "wont": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
}

# Informal forms whose meaning is unambiguous.  Anything ambiguous ("id",
# "ur" for "your" or "you are") is left untouched and reported as ambiguity.
INFORMAL = {
    "u": "you",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "tnx": "thanks",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "have to",
    "cuz": "because",
    "bc": "because",
    "b4": "before",
    "kinda": "kind of",
    "sorta": "sort of",
    "dunno": "do not know",
    "lemme": "let me",
    "gimme": "give me",
    "ya": "you",
    "imma": "i am going to",
}

# Spelling vocabulary for conservative edit-distance correction.  Only words
# in this list are ever proposed as replacements, so a correction cannot invent
# an entity name that was not already in the message.
TYPO_VOCABULARY = frozenset("""
a about above across actually add after again against all almost along already
also although always am among an and another any anyone anything are around as
ask asked at author away back bad based be because been before being believe
best better between big both brother but buy by call called came can cannot car
career change city class color colour come company could country course cousin
current currently dad daily daughter day dear did different do does dog doing
done down drink drinks during each early eat eats eight either else email end
engineer engineering enough even event ever every everyone everything exactly
example family father favorite favourite feel feeling few field film find fine
first five food for friend friends from full get gets getting girl give go going
good got great group had half happen happened has have he hear heard help her
here high him his history home hope hotel house how however husband idea if important
in information instead interest interested into is it its job just keep kind know
known language large last late later learn learned least leave left less let
life like likely line list little live lived lives living local long look lost
lot love loved loves made make makes making man many may maybe me mean meaning
medical meet member might mind minute miss mom money month more morning most
mother move moved movie much music must my name named native near need never
new next night nine no not nothing now number of off office often oh okay old on
once one only open or order other others our out over own parent parents part
partner past people person phone place plan please point possible prefer
preferred pretty probably problem product program project put question quite
rather read ready real really reason recent recently remember right role room
run said same saw say says school science second see seem seen self sense sent
serious set seven several she should show side since sister six size small so
social some someone something sometimes son soon sorry speak speaking speaks
spent stand start started state stay still stop story street strong student
studied studies study studying such sure system take talk talking team tell ten
than thank thanks that the their them then there these they thing things think
this those though thought three through time to today together told too took top
town try trying turn two under understand until up updated upon us use used
uses using very video want wanted wants was watch way we week well went were what
when where whether which while who whole why wife will with within without woman
word work worked working works world would write year years yes yet you young
your
academic allergic allergy answer attribute bag base bicycle bike birthday book
budget camera camping cat cats child children coffee cold college commute
commuting concert cooking cuisine currency degree desktop device diet dinner
dish doctor dog dogs dream drink drinks employer engineer evening exercise
fitness flat floor friend game garden goal guitar habit hobby hospital hotel
income instrument job kitchen laptop language lecture lesson library lunch
manager market meal meeting mobile monitor mountain movie museum neighbour
neighborhood notebook office order outdoor outfit pet phone photo piano pizza
plant podcast project reading recipe rent restaurant salary school science
scooter screen service shirt shoe shoes sister skill software song sport sports
station street study suit team technology tennis ticket tool town traffic train
travel trip truck university vacation vehicle wallet watch weather website
weekend wifi window wine work laptop phone car dog cat book music film movie
author artist singer band teacher doctor nurse engineer developer designer
manager director student professor researcher scientist writer painter farmer
driver pilot lawyer accountant chef waiter barista cashier clerk guard
red orange yellow green blue purple pink brown black white gray grey silver gold
teal cyan magenta violet indigo maroon beige coral crimson lavender turquoise
emerald sapphire ruby jade rust amber ivory pearl bronze copper navy olive mint
peach plum salmon tan khaki aqua scarlet azure
english spanish french german italian portuguese dutch russian arabic hebrew
hindi bengali urdu turkish swahili amharic korean japanese chinese mandarin
cantonese greek polish ukrainian romanian hungarian czech swedish norwegian
danish finnish icelandic irish welsh persian farsi vietnamese thai indonesian
malay tagalog
microbiology optics geology dynamics turbulence physics chemistry biology
mathematics economics history philosophy psychology sociology anthropology
linguistics neuroscience genetics ecology astronomy astrophysics engineering
medicine law architecture journalism design accounting finance marketing
management literature poetry sculpture painting
account address advice age ago agree air airport amount angry animal apartment
apple april area arm arrive arrived art august aunt autumn baby banana bank
basketball bath beach beautiful became become becomes becoming bed began begin
begins begun bird blood boat body boring boss bought box boy brain bread break
breakfast bridge bring broke broken brought build built business busy calling
calls card carried carries carrying case cause center chair chance changed
changes cheap choose chose chosen clean cleaning clock close clothes club coat
comes coming community computer continued continues control cook cooked cool
cost costs couple court cup data date deal death december decide decided
decision described describes detail difficult dirty door draw dress drew drive
driven drives drove earth east easy eaten eating edge education effect energy
expensive eye face fact fall false farm fast fear february feet finding finds
finish finished fire flew flower fly foot football force form found free friday
front fruit given gives giving glass goes gone grass grew ground grow growing
grown hair hand happens happy hard having head healthy heart held helped hill
hold holds horse hot hour hungry image impossible inch include included
includes interesting island item january juice july june keeps kept key kid
king knew knows lady lake land learning leaves leg letter level light liked
listen looking looks lose low machine main map march mark matter means meant
meat meets message met meter method middle mile milk moment monday moon mouth
moving nature nearly neck needed needs network news noise north note notice
november object ocean october offer oil opened opening page paid pain paint
pair paper park party path patient pay pays peace pen picture piece plane
played playing police pool position power president price property puts queen
race radio rain ran rarely reach remains remembered report rest result return
rice ride river road rock rode roof rope rule running sad salt sand sang sat
saturday save saved sea season seat secret seems sell send sends september
shape ship shop shopping short showed showing shown shows sick sign simple sing
sit site sits skin sky sleep slept slow smile snow sold sound soup space speed
spend spring star starts stayed step stone stood stopped stops store storm
summer sun sunday swim swimming table taken takes taking tall task tax tea
teach teaching telling tells test text thinks thirsty thursday tired title
tomorrow toy traveled traveling tree tried true truth tuesday turns twice type
ugly uncle unit usually valley value view village visited voice wait waited
waiting wake walk walking wall war warm wash watching water weak wednesday
weight west wheel whom whose win wind wing winter wish woke won wood writes
writing written wrong wrote yesterday
""".split())

_INFLECTION_SUFFIXES = ("s", "es", "ed", "ing", "er", "est", "ly")


def is_vocabulary_word(token: str) -> bool:
    """True when the folded token is a known vocabulary word."""
    return token.casefold() in TYPO_VOCABULARY


def is_vocabulary_inflection(token: str) -> bool:
    """True when the token is a known word or a regular inflection of one.

    Used to keep typo repair from rewriting valid forms that the vocabulary
    list happens to lack (``roads``, ``commutes``, ``worked``).
    """
    lower = token.casefold()
    if lower in TYPO_VOCABULARY:
        return True
    for suffix in _INFLECTION_SUFFIXES:
        # The token has to actually end with the suffix.  Slicing by length
        # alone turns "teachfes" into "teach" for the three-letter suffixes and
        # classifies a typo as a valid inflection, which then blocks repair.
        if not lower.endswith(suffix):
            continue
        stem = lower[:-len(suffix)]
        if len(stem) >= 2 and stem in TYPO_VOCABULARY:
            return True
    if lower.endswith("ies") and lower[:-3] + "y" in TYPO_VOCABULARY:
        return True
    if lower.endswith("ied") and lower[:-3] + "y" in TYPO_VOCABULARY:
        return True
    return False


# Common content words.  The typo repair below rewrites a token only when no
# reading of it is a known word; without this list an ordinary noun that the
# vocabulary happens to lack (a colour, an instrument, a vehicle) would be
# "repaired" into an unrelated one ("wheat" → "what", "tram" → "team").
CONTENT_WORDS = frozenset("""
accordion amp bassoon banjo bike boat boots bracelet bread burger butter
cable cake camera canoe carpet carrot cello chair charger cherry chess
chicken chocolate clarinet clock coat coffee cola console cookie curtain
cymbals dancing desk drum drums earrings egg eggs ferry fish flour flute
fridge gloves golf grape guitar harmonica harp hat headphones hiking honey
hockey jacket jeans jug juice kettle keyboard kite lamp laptop lens lime
lemon lentils lemonade lute mail mandolin mango melon metro microphone milk
mirror motorbike motorcycle mouse necklace noodles notebook oat oats oil
onion oven pancake pasta peach pear pepper piano pillow pizza plane player
plum pork potato printer projector radio raspberry record rice ring router
rugby salad salt sandwich saxophone scarf scanner scooter sewing shirt
shrimp skateboard skirt skis snowboard socks sofa soup speakers spinach
steak stings strawberry suitcase sunglasses surfing sweater table tablet
tambourine taxi tea telescope tennis tent tofu tomato toaster towel tram
trombone trousers trumpet tuba turkey ukulele van vacuum vase vinegar violin
wallet watch watermelon wheat wine yogurt
bay commuter soy studio studios
""".split())


# Common inflection and keyboard slips whose edit distance exceeds the
# single-edit rule but whose meaning is unambiguous.
COMMON_MISSPELLINGS = {
    "studys": "studies",
    "studing": "studying",
    "studed": "studied",
    "favourit": "favorite",
    "favourits": "favorites",
    "fav": "favorite",
    "favrt": "favorite",
    "colr": "color",
    "colur": "color",
    "cwlor": "color",
    "langauge": "language",
    "langage": "language",
    "alergic": "allergic",
    "alllergic": "allergic",
    "recieve": "receive",
    "seperate": "separate",
    "definately": "definitely",
    "wich": "which",
    "becuase": "because",
    "beacuse": "because",
    "bown": "born",
}

# Structural words: the verbs and attribute names the extraction rules match.
# Stage 1 may recover a misspelled token against this closed list because a
# wrong reading changes how the sentence parses.  Values, places, organisations
# and names are deliberately absent: recovering those would invent content.
STRUCTURAL_WORDS = frozenset("""
lives live lived living reside resides resided residing
works work worked working employ employed employs
studies study studied studying research researches researched researching
plays play played playing reads read reading speaks speak speaking
teaches teach taught teaching owns own owned drinks drink drank
eats eat eaten born allergic commute commutes commuted commuting
volunteer volunteers volunteered volunteering called named name
likes like loves love hates hate uses use used move moved relocated relocate
favorite favourite color colour author book film movie food music song sport
team language native job title role salary rent age aged degree birthday hometown
phone email address fluent instruct instructs enjoys enjoy prefers prefer
earns earn pays pay got
""".split())


# Words that look like typos but are themselves meaningful; never replace.
TYPO_PROTECTED = frozenset("""
id ok tv pc ai ml us uk eu nyc la sf dc mr mrs ms dr st jr sr vs etc ie eg
""".split())

# Place names that are one edit away from an ordinary word: "rome" is not a
# slip for "role", and rewriting a city into a different word would put a
# value in the fact that the speaker never wrote.  The typo corrector repairs
# only toward a closed vocabulary and cannot tell a city from a slip, so the
# names it must leave alone are listed explicitly.  A capitalised place is
# never rewritten anyway; this covers the lowercase running speech that most
# real messages use.
COMMON_PLACE_NAMES = frozenset("""
addis ababa nairobi mombasa kampala kigali dodoma khartoum cairo alexandria
casablanca marrakesh tunis algiers tripoli lagos accra dakar bamako niamey
abidjan monrovia freetown banjul lome pretoria durban johannesburg
windhoek gaborone harare lusaka luanda maputo
london manchester birmingham liverpool leeds sheffield bristol cardiff
edinburgh glasgow belfast dublin cork galway hayes camden soho
paris lyon marseille toulouse nice bordeaux lille nantes strasbourg
berlin munich hamburg cologne frankfurt stuttgart dresden leipzig bonn
bern zurich geneva basel lausanne luzern vienna prague bratislava budapest
warsaw krakow rome milan turin naples florence venice bologna genoa palermo
siena pisa bari padua madrid barcelona valencia seville granada bilbao
malaga lisbon porto amsterdam rotterdam utrecht brussels antwerp ghent
copenhagen stockholm oslo bergen helsinki reykjavik tallinn vilnius riga
moscow kyiv kiev minsk chisinau bucharest sofia skopje tirana zagreb
belgrade sarajevo ljubljana istanbul ankara izmir athens thessaloniki
nicosia valletta york kent yale dale gale vale wale
boston chicago houston phoenix dallas austin seattle denver atlanta miami
toronto vancouver montreal ottawa calgary edmonton bogota quito caracas
santiago montevideo asuncion brasilia rio tokyo osaka kyoto seoul beijing
shanghai shenzhen guangzhou taipei singapore jakarta manila bangkok hanoi
delhi mumbai kolkata chennai bangalore hyderabad karachi lahore dhaka
colombo kathmandu yangon sydney melbourne brisbane perth adelaide canberra
auckland wellington dubai doha riyadh jeddah muscat kuwait manama tehran
baghdad amman beirut damascus jerusalem gaza acre
""".split())

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

PLACE_ALIASES = {
    "addis": "Addis Ababa",
    "nyc": "New York City",
    "sf": "San Francisco",
    "la": "Los Angeles",
    "dc": "Washington DC",
    "uk": "United Kingdom",
    "usa": "United States",
    "us": "United States",
}

RELATION_NOUNS = frozenset("""
wife husband partner girlfriend boyfriend fiance fiancee son daughter mother
father mom dad sister brother friend best_friend boss manager colleague
coworker teammate child kid cousin uncle aunt nephew niece roommate spouse
grandmother grandfather grandparent grandson granddaughter sibling
neighbour neighbor landlord landlady flatmate housemate
stepmother stepfather stepbrother stepsister
""".split())

RELATION_GENDER = {
    "wife": "female", "girlfriend": "female", "fiancee": "female",
    "daughter": "female", "mother": "female", "mom": "female",
    "sister": "female", "aunt": "female", "niece": "female",
    "husband": "male", "boyfriend": "male", "fiance": "male",
    "son": "male", "father": "male", "dad": "male", "brother": "male",
    "uncle": "male", "nephew": "male",
    "grandmother": "female", "granddaughter": "female", "landlady": "female",
    "stepmother": "female", "stepsister": "female",
    "grandfather": "male", "grandson": "male", "stepfather": "male",
    "stepbrother": "male",
}

# ── relation normalization ───────────────────────────────────────────────── #

PREDICATE_SYNONYMS = {
    "live_in": "lives_in",
    "lives_at": "lives_in",
    "reside_in": "lives_in",
    "resides_in": "lives_in",
    "based_in": "lives_in",
    "moved_to": "lives_in",
    "fluent_in": "speaks",
    "speak": "speaks",
    "knows_language": "speaks",
    "study": "studies",
    "majoring_in": "studies",
    "majored_in": "studies",
    "learning": "studies",
    "work_as": "works_as",
    "work_at": "works_at",
    "employed_by": "works_at",
    "is_allergic_to": "allergic_to",
    "allergic": "allergic_to",
    "favourite": "favorite",
}

ATTRIBUTE_PREDICATES = {
    "favorite color": "favorite_color",
    "favorite colour": "favorite_color",
    "favorite film": "favorite_film",
    # "movie" and "film" name the same attribute; one canonical keeps a later
    # restatement comparable with an earlier one instead of looking distinct.
    "favorite movie": "favorite_film",
    "favorite author": "favorite_author",
    "favorite book": "favorite_book",
    "favorite food": "favorite_food",
    "favorite drink": "favorite_drink",
    "favorite music": "favorite_music",
    "favorite song": "favorite_song",
    "favorite sport": "favorite_sport",
    "favorite team": "favorite_team",
    "favorite language": "favorite_language",
    "favourite color": "favorite_color",
    "favourite colour": "favorite_color",
    "favourite film": "favorite_film",
    "favourite movie": "favorite_film",
    "favourite author": "favorite_author",
    "favourite book": "favorite_book",
    "favourite food": "favorite_food",
    "favourite drink": "favorite_drink",
    "favourite music": "favorite_music",
    "favourite song": "favorite_song",
    "favourite sport": "favorite_sport",
    "favourite team": "favorite_team",
    "favourite language": "favorite_language",
    "preferred author": "favorite_author",
    "preferred film": "favorite_film",
    "preferred movie": "favorite_film",
    "native language": "native_language",
    "job": "job",
    "job title": "role",
    "role": "role",
    "salary": "salary",
    "rent": "rent",
    "age": "age",
    "phone number": "phone_number",
    "email address": "email_address",
    "e-mail": "email_address",
    "degree": "degree",
    "birthday": "birthday",
    "hometown": "hometown",
}

# ── temporal / discourse markers ─────────────────────────────────────────── #

TEMPORAL_PAST = (
    "used to", "previously", "formerly", "back in", "in the past",
    "when i was", "years ago", "last year", "last month", "last week",
)

TEMPORAL_NOW = (
    "now", "currently", "these days", "at the moment", "right now",
    "anymore", "still", "no longer", "no more",
)

TEMPORAL_FUTURE = ("will", "going to", "plan to", "next year", "soon")

CORRECTION_MARKERS = (
    "actually", "correction", "i mean", "rather", "instead", "to be clear",
    "update", "sorry", "scratch that",
)

NEGATIONS = (
    "not", "never", "no longer", "isn't", "aren't", "don't", "doesn't",
    "didn't", "cannot", "can't", "without", "none", "neither", "nor",
)

HYPOTHETICAL = ("if", "would", "could", "might", "maybe", "perhaps",
                "suppose", "imagine")

STOPWORDS = frozenset("""
a an the and or but if of to in on at for with by from as is are was were be
been being am do does did have has had i you he she it we they me him her us
them my your his its our their this that these those what which who whom whose
where when why how not no yes so than then there here about into over after
before between during without within up down out off again further once only
own same very just than too also can could would should will shall may might
must
""".split())

# ── language identification ──────────────────────────────────────────────── #

LANGUAGE_STOPWORDS = {
    "en": frozenset("""
a an the and is are am was were be been being of to in on at for with by
from as it this that these those i you he she we they my your his her our
their what which who how when where why not no do does did have has had will
would can could should there their then than so if but or about into over
after before just very also more most some any all
""".split()),
    "es": frozenset("""
el la los las un una unos unas y o pero de del en con por para que como es
son estoy está están mi tu su mis tus sus yo tú él ella nosotros ellos esto
eso este esta muy también porque cuando donde quien hay no sí se le les lo
hola gracias soy eres estas estamos trabajo vivo hablo estudio
""".split()),
    "fr": frozenset("""
le la les un une des et ou mais de du en avec pour que qui comme est sont je
tu il elle nous vous ils elles mon ton son ma ta sa mes tes ses ce cette ces
pas ne plus très aussi parce quand où quoi sur dans au aux bonjour salut
merci suis habite parle travaille
""".split()),
    "de": frozenset("""
der die das ein eine einer eines und oder aber von zu im in mit für auf ist
sind ich du er sie es wir ihr mein dein sein ihre nicht kein keine auch sehr
wie wo wer was warum dass wenn als bei aus nach über unter
""".split()),
    "it": frozenset("""
il lo la i gli le un una di del della e o ma che chi come è sono io tu lui
lei noi voi loro mio tuo suo mia tua sua non più anche molto perché quando
dove con per in su da al alla
""".split()),
    "pt": frozenset("""
o a os as um uma uns umas e ou mas de do da em com para que quem como é são
eu tu ele ela nós vós eles elas meu tua seu minha sua não mais também muito
porque quando onde isso este esta
""".split()),
    "nl": frozenset("""
de het een en of maar van in op met voor dat die is zijn ik jij hij zij wij
jullie mijn jouw niet geen ook heel erg hoe waar waarom wanneer omdat als
bij uit naar over onder
""".split()),
    "tr": frozenset("""
ve bir bu şu o da de için ile ama veya ben sen o biz siz onlar benim senin
değil çok daha nasıl neden nerede ne zaman gibi olarak var yok
""".split()),
    "sw": frozenset("""
na ya wa kwa ni katika za la cha vya mimi wewe yeye sisi nyinyi wao yangu
yako yake yetu yao si sana pia lakini au kama nini wapi lini kwa nini
""".split()),
    "id": frozenset("""
dan yang di ke dari untuk dengan ini itu saya kamu dia kami kita mereka
tidak bukan juga sangat apa bagaimana mengapa kapan dimana karena atau
sudah belum akan
""".split()),
    "vi": frozenset("""
và của là có không tôi bạn anh chị em chúng ta họ này đó một những cho với
được trong khi nhưng hoặc rất cũng tại sao như thế nào ở đâu khi nào
""".split()),
}

SCRIPT_LANGUAGE = {
    "ethiopic": "am",
    "arabic": "ar",
    "hebrew": "he",
    "cyrillic": "ru",
    "greek": "el",
    "devanagari": "hi",
    "han": "zh",
    "hangul": "ko",
    "thai": "th",
}

SCRIPT_RANGES = {
    "latin": ((0x0041, 0x024F), (0x1E00, 0x1EFF)),
    "cyrillic": ((0x0400, 0x04FF),),
    "greek": ((0x0370, 0x03FF),),
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F)),
    "hebrew": ((0x0590, 0x05FF),),
    "devanagari": ((0x0900, 0x097F),),
    "ethiopic": ((0x1200, 0x137F),),
    "han": ((0x4E00, 0x9FFF),),
    "hangul": ((0xAC00, 0xD7AF),),
    "thai": ((0x0E00, 0x0E7F),),
}

URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Hyphenated and slashed digit groups stay one token: "555-0134" and
# "2024-05-06" are values, and re-spacing them would alter the number.  A digit
# group followed by "-word" ("30-something") is not a number, so the match must
# not stop at the digits and protect only half of the word.
NUMBER = re.compile(
    r"(?<!\w)(?:[$€£]\s?\d[\d,.]*|\d[\d,.]*)(?:[-/]\d+)*%?(?![\w-])")
# The full date forms come first: the year branch would otherwise match the
# leading "2024" of "2024-05-06" and leave the rest of the date exposed.
DATE = re.compile(
    r"\b(?:19|20)\d{2}-\d{1,2}-\d{1,2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    r"|\b(?:19|20)\d{2}\b")
TOKEN = re.compile(
    # A hyphenated word is one token: "sci-fi" and "well-known" are single
    # words, and splitting them would re-space the value the author wrote.
    r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*"
    r"(?:-[^\W\d_]+(?:['\u2019][^\W\d_]+)*)*"
    r"|\d+(?:-\d+)*(?:-[^\W\d_]+(?:-[^\W\d_]+)*)*"
    r"|[^\s]",
    re.UNICODE)
# Script-specific letters that separate closely related Cyrillic languages.
CYRILLIC_MARKERS = {
    "uk": set("іїєґІЇЄҐ"),
    "be": set("ўЎ"),
    "bg": set("ъ"),
    "sr": set("ђћџЂЋЏ"),
    "mk": set("ѓќѕЃЌЅ"),
}

__all__ = [
    "ATTRIBUTE_PREDICATES",
    "COMMON_MISSPELLINGS",
    "COMMON_PLACE_NAMES",
    "CONTRACTIONS",
    "CONTENT_WORDS",
    "CORRECTION_MARKERS",
    "CYRILLIC_MARKERS",
    "DATE",
    "EMAIL",
    "HYPOTHETICAL",
    "INFORMAL",
    "LANGUAGE_STOPWORDS",
    "NEGATIONS",
    "NUMBER",
    "NUMBER_WORDS",
    "PLACE_ALIASES",
    "PREDICATE_SYNONYMS",
    "RELATION_GENDER",
    "RELATION_NOUNS",
    "SCRIPT_LANGUAGE",
    "SCRIPT_RANGES",
    "STOPWORDS",
    "STRUCTURAL_WORDS",
    "TEMPORAL_FUTURE",
    "TEMPORAL_NOW",
    "TEMPORAL_PAST",
    "TOKEN",
    "TYPO_PROTECTED",
    "TYPO_VOCABULARY",
    "URL",
    "is_vocabulary_inflection",
    "is_vocabulary_word",
]
