"""Helpers for the UDFs in agent-written plan code (pipeline.map / pipeline.filter lambdas), which
the plan sandbox makes available by name."""
from __future__ import annotations


def number_items(items: list) -> str:
    """The items as numbered lines ('1. first\n2. second'), so an LLM can refer to items by number."""
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items or [], 1))


def keys_by_items(keys: list, lists: list) -> dict:
    """For each item, the keys whose list contains it: keys [7, 9] and lists
    [["Jack", "Lily"], ["Lily"]] give {"Jack": [7], "Lily": [7, 9]}."""
    out: dict = {}
    for key, items in zip(keys, lists):
        for item in items or []:
            out.setdefault(item, []).append(key)
    return out


def row_id_str(value) -> str:
    """A row id as it names files on disk (<image_subdir>/<id>.jpg). An integer id that reaches a
    UDF as a float (50194.0) gives '50194'."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
