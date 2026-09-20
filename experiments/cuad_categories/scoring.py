"""Score one CUAD category: did the plan give the right answer, and did it quote the right text.

Ground truth is ground_truth/cuad_gt.csv, built by cuad/clean_cuad_gt.py. A contract scores 1 or
0: the question is whether the plan got the information out, not how it phrased it.

  1  the answer means what the ground truth says -- a date part the contract leaves blank matches
     anything, "6 months" matches "180 days", "Beijing, China" contains the ground-truth place
     "Beijing", and for Parties the count is right;
  1  or, failing that, the text it quoted overlaps CUAD's highlighted text by more than
     SPAN_THRESHOLD (token-set Jaccard, the word-set measure docetl's CUAD scorer uses): the right
     clause was found even though the value came out wrong;
  0  otherwise.

Where the ground-truth answer is "unspecified" the contract states no value to get right, so only
the quoted text is judged.

A contract whose `unusable` column names this category is skipped: its ground truth was found not
to be gradable (see clean_cuad_gt.py). For Parties, "number_parties" disqualifies only the count,
so those contracts are judged on their spans.
"""
from __future__ import annotations

import json
import re

import pandas as pd

# Answers that are a word rather than a value. Each is a real answer: "perpetual" (no end),
# "relative" (tied to an event), "unspecified" (the clause states no length), "redacted"
# (blacked out in the filing).
KEYWORDS = {"perpetual", "relative", "unspecified", "redacted"}
UNSPECIFIED = "unspecified"
# Answers that say the contract states no usable value. Getting the word right does not show the
# plan found the clause, so these also need the quoted text to line up.
SPAN_BACKED = {"unspecified", "redacted", "relative"}
# How much of CUAD's highlighted text a plan must have quoted for the clause to count as found.
SPAN_THRESHOLD = 0.5

# Place names a plan may reasonably use for the same jurisdiction. The ground truth holds the most
# specific place the contract names (see clean_cuad_gt.py); anything here is accepted for it.
PLACE_ALIASES = {
    "United States": ["usa", "us", "u.s.", "u.s.a.", "united states of america", "american"],
    "District of Columbia": ["dc", "d.c.", "washington dc", "washington, d.c."],
    "China": ["prc", "p.r.c.", "peoples republic of china", "mainland china"],
    "Hong Kong": ["hksar", "hong kong sar", "hong kong special administrative region"],
    "Taiwan": ["republic of china", "roc"],
    "United Kingdom": ["uk", "u.k.", "great britain", "britain"],
    "Netherlands": ["holland", "the netherlands"],
    "South Africa": ["rsa"],
    "Papua New Guinea": ["png"],
    "British Columbia": ["bc", "b.c."],
}
# Two-letter postal codes, so "NY" is accepted for New York.
_STATE_CODES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
# The country a region sits in. A plan that names both ("Beijing, China") is not charged for the
# country, but naming only the country is not the place the contract chose.
PLACE_PARENT = {
    **{state: "United States" for state in _STATE_CODES}, "District of Columbia": "United States",
    "England": "United Kingdom", "Wales": "United Kingdom", "Ontario": "Canada",
    "British Columbia": "Canada", "Nova Scotia": "Canada", "Victoria": "Australia",
    "Beijing": "China", "Hong Kong": "China",
}
COUNTRIES = {
    "United States", "United Kingdom", "Canada", "Australia", "China", "Germany", "Kazakhstan",
    "South Africa", "Israel", "Japan", "Netherlands", "India", "Switzerland", "Taiwan", "Italy",
    "Spain", "Colombia", "Papua New Guinea", "Singapore", "Belgium", "Hong Kong",
}
# Everything that counts as naming a jurisdiction. A phrase that resolves to none of these is
# wording, not a place ("without regard to conflicts of law"), and is not charged as a wrong one.
KNOWN_PLACES = set(PLACE_PARENT) | set(PLACE_ALIASES) | COUNTRIES | {"redacted"}
_ALIAS_TO_PLACE = {alias: place for place, aliases in PLACE_ALIASES.items() for alias in aliases}
_ALIAS_TO_PLACE.update({code.lower(): state for state, code in _STATE_CODES.items()})
_PLACE_PREFIX = re.compile(
    r"^(the )?(laws? of )?(the )?((people's|federal) )?"
    r"(republic|commonwealth|state|province|city|municipality) of ", re.I)
# A governing law that is not a place at all; graded on spans only.
NON_PLACE = "non-place"
DAYS = {"days": 1, "weeks": 7, "months": 30, "years": 365}
_DURATION = r"([\d.]+)\s*(day|week|month|year)s?"
_COUNTED = re.compile(rf"(unlimited|\d+)\s*[x×]\s*{_DURATION}", re.I)
_PLAIN = re.compile(_DURATION, re.I)
_DATE = re.compile(r"([\d?]{1,4})\s*[/-]\s*([\d?]{1,4})\s*[/-]\s*([\d?]{2,4})")
_NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
_EMPTY = {"", "none", "null", "n/a", "na", "-", "[]", "not present", "not stated", "nan"}


def token_set(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9\s]", " ", str(text).lower()).split())


def span_jaccard(gt_spans: list[str], predicted: str) -> float:
    gold, pred = token_set(" ".join(gt_spans)), token_set(predicted)
    if not gold and not pred:
        return 1.0
    union = gold | pred
    return len(gold & pred) / len(union) if union else 1.0


def _year(text: str) -> str:
    """A two-digit year as CUAD writes it: 90-99 are 1990s, 00-45 are 2000s."""
    return ("19" if int(text) >= 50 else "20") + text if re.fullmatch(r"\d{2}", text) else text


def parse_answer(text: str) -> list:
    """A plan's answer as comparable items, in the same shapes cuad_gt.csv uses. Accepts the
    formats the task prompt asks for, and JSON, since a plan may echo the ground-truth shape."""
    text = "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text).strip()
    if text.lower() in _EMPTY:
        return []
    try:                                    # a plan that answered in the ground truth's own JSON
        loaded = json.loads(text)
        if isinstance(loaded, (list, dict)):
            items = loaded if isinstance(loaded, list) else [loaded]
            return [i for i in items if i not in ("", None)]
    except (json.JSONDecodeError, TypeError):
        pass

    items: list = []
    for part in (p.strip() for p in re.split(r";|\n|\band\b", text) if p.strip()):
        low = part.lower()
        keyword = next((k for k in KEYWORDS if k in low), None)
        counted, date, plain = _COUNTED.search(part), _DATE.search(part), _PLAIN.search(part)
        if counted:
            renewals = counted[1].lower()
            items.append({"renewals": "unlimited" if renewals == "unlimited" else int(renewals),
                          "length": float(counted[2]), "unit": counted[3].lower() + "s"})
        elif date:
            month, day, year = (g.zfill(2) if g.isdigit() else "??" for g in date.group(1, 2, 3))
            items.append({"month": month, "day": day, "year": _year(date[3]) if date[3].isdigit() else "????"})
        elif plain:
            items.append({"length": float(plain[1]), "unit": plain[2].lower() + "s"})
        elif keyword:
            items.append(keyword)
        else:
            items.append(part)              # a place, for Governing Law
    return items


def _date_part_matches(gold: str, predicted) -> bool:
    predicted = "" if predicted is None else str(predicted)
    if "?" in gold:                         # unknown in the contract: only "unknown" is right
        return not predicted or "?" in predicted or predicted.strip("0").lower() in ("", "x", "xx", "unknown")
    return gold == predicted


def _in_days(item: dict) -> float | None:
    scale = DAYS.get(str(item.get("unit")))
    length = item.get("length")
    return scale * length if scale and isinstance(length, (int, float)) else None


def _as_keyword(item):
    """A value whose length is blacked out is the answer "redacted", whatever else it carries:
    {"renewals": "unlimited", "length": "redacted", "unit": "years"} is answered "redacted"."""
    if isinstance(item, dict) and any(v == "redacted" for v in item.values()):
        return "redacted"
    return item


def _match(gold, predicted) -> float:
    """How well one predicted item matches one ground-truth item, 0 to 1."""
    gold, predicted = _as_keyword(gold), _as_keyword(predicted)
    if isinstance(gold, str) or isinstance(predicted, str):
        return float(str(gold).lower() == str(predicted).lower())
    if "year" in gold:
        # A date part the contract leaves blank has to come back unknown, not guessed: the task
        # prompt asks for "?" there, so a plan that invents a day is wrong, not lucky.
        if "year" not in predicted:
            return 0.0
        return float(all(_date_part_matches(gold[k], predicted.get(k)) for k in ("month", "day", "year")))
    if "length" not in gold or "length" not in predicted or "year" in predicted:
        return 0.0
    gold_days, predicted_days = _in_days(gold), _in_days(predicted)
    if gold_days is None or predicted_days is None or abs(gold_days - predicted_days) > 0.5:
        return 0.0
    if "renewals" not in gold:
        return 1.0
    # A renewal answer is the count and the length together: the right length for the wrong number
    # of renewals is a different renewal term, and earns nothing.
    return float(str(gold.get("renewals")) == str(predicted.get("renewals")))


def _flatten(text: str) -> str:
    """Lowercased, punctuation stripped, single-spaced -- so "U.S.", "the State of New York," and
    "new york" all compare the same way."""
    flat = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
    # "u s a" is "usa": a dotted abbreviation loses its dots, not its letters
    flat = re.sub(r"\b(\w)\s+(?=\w\b)", r"\1", flat)
    return f" {flat} "


def _names_for(place: str) -> list[str]:
    """Everything a plan may write for this place: the ground truth's own name and its aliases."""
    names = [place, *PLACE_ALIASES.get(place, [])]
    if place in _STATE_CODES:
        names.append(_STATE_CODES[place])
    return names


# Words that surround a place name without being part of it, so a short code can still be seen
# standing alone in "U.S. federal law".
_FILLER = {"the", "law", "laws", "federal", "state", "of", "internal", "substantive", "governing",
           "court", "courts", "jurisdiction"}


def _without_filler(flattened: str) -> str:
    return " " + " ".join(w for w in flattened.split() if w not in _FILLER) + " "


def _mentions(place: str, flattened: str) -> bool:
    """Whether the answer names this place anywhere in it. Extra words around the name do not
    matter ("the laws of the State of New York" names New York); short codes have to stand alone,
    so "IN" for Indiana does not fire on the word "in"."""
    for name in _names_for(place):
        flat = _flatten(name)
        if len(flat.strip()) <= 3:
            if flat in (flattened, _without_filler(flattened)):
                return True
        elif flat in flattened:
            return True
    return False


def canonical_place(text: str) -> str:
    """One place name as the ground truth writes it, for deciding whether a place the answer names
    is one nobody asked for."""
    flattened = _flatten(text)
    for place in KNOWN_PLACES:
        if _mentions(place, flattened):
            return place
    return _ALIAS_TO_PLACE.get(flattened.strip(), flattened.strip())


def governing_law_f1(gold: list, predicted_text: str) -> float | None:
    """F1 over the places named. Recall asks whether each ground-truth place (or an alias of it)
    appears in the answer at all, so wording around it is free; precision charges for a place the
    answer names that the contract's law is not, except the country of a place it got right
    ("Beijing, China")."""
    wanted = [g for g in gold if g != NON_PLACE]
    flattened = _flatten(predicted_text)
    if not wanted:
        return None if gold else float(not flattened.strip())
    if not flattened.strip():
        return 0.0
    hits = [place for place in wanted if _mentions(place, flattened)]
    parents = {PLACE_PARENT.get(place) for place in hits}
    extra = 0
    for piece in re.split(r";|,|&|\band\b|\bor\b|\n", str(predicted_text or "")):
        if not piece.strip() or any(_mentions(place, _flatten(piece)) for place in hits):
            continue
        named = canonical_place(piece)
        if named not in KNOWN_PLACES or named in parents:
            continue                        # wording, or the country of a place it got right
        extra += 1
    precision = len(hits) / (len(hits) + extra) if hits or extra else 0.0
    recall = len(hits) / len(wanted)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def answer_score(category: str, gold: list, predicted_text: str) -> float | None:
    """1 when the plan's answer means what the ground truth says, 0 when it does not, None when
    this contract has nothing to score for the category."""
    predicted = parse_answer(predicted_text)
    if not gold:
        return float(not predicted)
    if not predicted:
        return 0.0
    return max(_match(g, p) for g in gold for p in predicted)


def _party_count(predicted_text) -> int | None:
    """The one count an answer gives. The task prompt asks for a single integer, so an answer
    holding anything else that could be a count gives none."""
    text = str(predicted_text if predicted_text is not None else "").strip().lower()
    if text in _NUMBER_WORDS:
        return _NUMBER_WORDS[text]
    numbers = re.findall(r"\d+", text)
    return int(numbers[0]) if len(numbers) == 1 else None


def party_count_score(gold: int | None, predicted_text) -> float | None:
    if gold is None:
        return None
    return float(_party_count(predicted_text) == gold)


def party_count_closeness(gold: int | None, predicted_text) -> float | None:
    """1 / (1 + |gold - predicted| / gold): 1 for the right count, and 0 for an answer that gives
    no single count."""
    if not gold:
        return None
    predicted = _party_count(predicted_text)
    return 0.0 if predicted is None else 1 / (1 + abs(gold - predicted) / gold)


def _best_match(gold: list, predicted: list):
    """The ground-truth item the plan's answer matches, and how well."""
    best, best_item = 0.0, None
    for item in gold:
        for candidate in predicted:
            score = _match(item, candidate)
            if score > best:
                best, best_item = score, _as_keyword(item)
    return best, best_item


def score_document(category: str, gold_answer, gold_span: list[str] | None,
                   answer_text, spans_text) -> float:
    """This contract's quality, 1 or 0 (a fraction only for Governing Law's F1).

    `gold_answer` and `gold_span` come from the query's own ground-truth file (Q{id}_gt.csv):
    the answer, and the text CUAD highlighted where the answer alone cannot show the plan read the
    right clause -- None where that file says no span is needed.

    The answer decides it. The quoted text is consulted only where the answer is a word that says
    the contract states no usable value -- "unspecified", "redacted", "relative".
    """
    if category == "Governing Law":         # a list of places, scored by F1; spans are not used
        return governing_law_f1(gold_answer or [], answer_text)
    if category == "Parties":               # a single number; spans are not used
        return party_count_score(gold_answer, answer_text)

    gold = gold_answer or []
    predicted = parse_answer(answer_text or "")
    if not gold:                            # the contract has no such clause: silence is the answer
        return float(not predicted)
    if not predicted:
        return 0.0
    matched, item = _best_match(gold, predicted)
    if matched != 1.0:
        return 0.0
    if isinstance(item, str) and item in SPAN_BACKED and gold_span is not None:
        return float(span_jaccard(gold_span, spans_text or "") > SPAN_THRESHOLD)
    return 1.0
