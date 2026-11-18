"""Word-set overlap between a plan's quoted text and the ground truth's.

Token-set Jaccard is the measure docetl's CUAD scorer uses: lowercase, punctuation to spaces, and
compare the sets of words.
"""
from __future__ import annotations

import re

# How much of the ground truth's text a quote must share for it to count as the right passage.
SPAN_THRESHOLD = 0.5


def token_set(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9\s]", " ", str(text).lower()).split())


def span_jaccard(gt_spans: list[str], predicted: str) -> float:
    """Jaccard against the UNION of the gold spans, as CUAD grades a category whose highlighted
    text comes in several pieces that together make one answer."""
    gold, pred = token_set(" ".join(gt_spans)), token_set(predicted)
    if not gold and not pred:
        return 1.0
    union = gold | pred
    return len(gold & pred) / len(union) if union else 1.0


def best_span_jaccard(gt_spans: list[str], predicted: str) -> float:
    """Jaccard against the closest single gold span, for documents where each gold span is a
    separate, equally valid quote (a report that states the same fact in several places)."""
    return max((span_jaccard([span], predicted) for span in gt_spans), default=0.0)
