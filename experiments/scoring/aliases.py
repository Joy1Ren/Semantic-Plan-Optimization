"""Match a plan's name for an entity against a stored list of that entity's accepted names.

Both sides are normalized the same way -- accents, case, "&", punctuation, and legal forms such as
"Inc." or "S.p.A." -- and compared exactly. Whatever variants the ground truth accepts are listed
in its alias file; nothing is fuzzy-matched, so a near-miss name is wrong unless it is listed.
"""
from __future__ import annotations

import html
import re
import unicodedata

# Legal forms and corporate words dropped from the end of a name. Multi-word forms are matched
# after punctuation is gone ("s p a", "public company limited").
LEGAL_SUFFIXES = sorted({
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies", "ltd", "limited",
    "plc", "llc", "llp", "lp", "l p", "sa", "s a", "spa", "s p a", "sarl", "sas", "ag", "se", "nv",
    "n v", "bv", "b v", "ab", "ab publ", "publ", "as", "a s", "asa", "oyj", "oy", "kgaa",
    "se co kgaa", "gmbh", "holdings", "holding", "group", "groupe", "gruppo", "berhad", "bhd",
    "pjsc", "jsc", "tbk", "pt", "pte", "pty", "pty ltd", "public company limited", "co ltd",
    "kk", "k k", "srl", "s r l", "sab de cv", "de cv", "nl", "the", "and co", "and company",
}, key=lambda s: -len(s))
_SUFFIX = re.compile(r"(?:\s+(?:%s))+$" % "|".join(re.escape(s) for s in LEGAL_SUFFIXES))
_PREFIX = re.compile(r"^(?:the|pt)\s+")


def normalize_name(name) -> str:
    """Lowercase ASCII words with the legal form removed: "Enpro Industries, Inc." -> "enpro
    industries", "Center-invest Bank" -> "center invest bank", "Domino's Pizza" -> "dominos pizza"."""
    text = unicodedata.normalize("NFKD", html.unescape(str(name or ""))).encode("ascii", "ignore").decode()
    text = re.sub(r"['\u2019]", "", text.lower()).replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    text = _PREFIX.sub("", text)
    stripped = _SUFFIX.sub("", text).strip()
    return stripped or text


def name_matches(predicted, aliases: list[str]) -> bool:
    """Whether the plan's name, normalized, equals one of the accepted names, normalized. A name
    with a parenthetical ("Saudi Aramco (Saudi Arabian Oil Company)") also matches on either part."""
    text = str(predicted or "")
    candidates = {normalize_name(part) for part in [text, *re.split(r"[()]", text)]} - {""}
    return bool(candidates & {normalize_name(a) for a in aliases})
