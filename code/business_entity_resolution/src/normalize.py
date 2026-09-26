"""Text normalization for business names and addresses.

Handles the noise patterns described in the challenge: abbreviations, legal
suffixes, punctuation, and domain-name-style business names. Cross-script
names (e.g. an Indian entity written in Tamil/Hindi/Kannada in source2/3 but
English in source1) are NOT transliterated here -- there is no reliable,
license-clean way to do that locally, and address fields (street numbers,
city/state tokens) turn out to stay far more consistent across scripts than
names do. The matching model is trained to lean on address similarity for
those cases instead.

Performance/memory note: this runs over 10M+ rows on a 16GB-RAM machine.
Only plain normalized *strings* are kept as persistent DataFrame columns.
Token/digit sets are cheap to derive from a string (`str.split`, a regex
findall) and are computed transiently wherever they're needed (blocking's
explode step, feature Jaccard scores) instead of being materialized as
`frozenset` columns across the whole dataset -- frozensets carry heavy
per-object Python overhead and storing ~30M of them was blowing past
available memory and thrashing the disk.
"""
import re
import string
import unicodedata

LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "pvt", "private", "limited",
    "ltd", "llp", "llc", "inc", "incorporated", "corp", "corporation",
    "co", "company", "enterprises", "enterprise",
]
_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b\.?"
)

_DOMAIN_RE = re.compile(r"\.(com|net|org|co|in|biz|info)\b")
_DIGIT_RE = re.compile(r"\d+")

_EXTRA_PUNCT = "‘’“”–—"  # curly quotes, en/em dash
_PUNCT_TABLE = {ord(c): " " for c in string.punctuation + _EXTRA_PUNCT}

ADDRESS_ABBREVIATIONS = {
    "rd": "road", "st": "street", "ave": "avenue", "ln": "lane",
    "dr": "drive", "blvd": "boulevard", "apt": "apartment", "ste": "suite",
    "no": "number", "ph": "phone",
}


def basic_clean(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return text.lower()


def normalize_name(raw_name):
    """Returns the normalized name string only (no suffix/legal-form words,
    no punctuation, no domain extension)."""
    text = basic_clean(raw_name)
    if not text:
        return ""
    text = _DOMAIN_RE.sub("", text)
    text = text.translate(_PUNCT_TABLE)
    text = _SUFFIX_RE.sub(" ", text)
    return " ".join(text.split())


def normalize_address(raw_address):
    text = basic_clean(raw_address)
    if not text:
        return ""
    text = text.translate(_PUNCT_TABLE)
    tokens = [ADDRESS_ABBREVIATIONS.get(t, t) for t in text.split()]
    return " ".join(tokens)


def compact(norm_text):
    """Alnum/no-space form for comparing against domain-style names."""
    return norm_text.replace(" ", "")


def tokens(norm_text):
    return frozenset(norm_text.split()) if norm_text else frozenset()


def digit_tokens(norm_text):
    if not norm_text:
        return frozenset()
    return frozenset(t for t in _DIGIT_RE.findall(norm_text) if len(t) >= 2)


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0
