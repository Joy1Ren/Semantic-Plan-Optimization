"""Row and query embeddings, and the cache that keeps them from being paid for twice.

Row embeddings dominate the sampler's cost -- an existing on-disk record shows ~$30 for a
single use case's rows -- so the cache is the load-bearing part of this module. It is keyed by
(use case via `cache_dir`, embedding model via the filename, column set via a fingerprint) and
shared across every query of that use case. It only ever grows: an id already embedded for a
given column set is never embedded again, even by a different query.

Column sets matter because a row's vector depends on which columns were concatenated to make
it. The previous sampler keyed only on the model, so two queries whose CSVs had different
columns silently shared one set of vectors. Fingerprinting the column list fixes that and lets
the agent target a subset of columns without invalidating what is already cached.
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from agent_cost_model.sampling.constants import DEFAULT_EMBED_WORKERS, _RAG_EMBEDDING_BASE_URL
from agent_cost_model.sampling.text import column_fingerprint

__all__ = ["EmbedCost", "EmbeddingClient", "EmbeddingCache", "model_slug"]


def model_slug(model: str) -> str:
    """Filesystem-safe slug for an embedding model id, so caches never mix across models."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


@dataclass
class EmbedCost:
    cost: float = 0.0
    latency: float = 0.0
    n_calls: int = 0

    def add(self, cost: float, latency: float) -> None:
        self.cost += cost
        self.latency += latency
        self.n_calls += 1


class EmbeddingClient:
    """One embedding call, over raw HTTP.

    Not the OpenAI SDK's embeddings helper: it defaults to base64 encoding plus a post-parser
    that hides provider errors behind a cryptic "No embedding data received". A direct call
    with encoding_format="float" is broadly compatible and surfaces the real response body.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = _RAG_EMBEDDING_BASE_URL,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        import httpx

        self.model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self._http = httpx.Client(timeout=timeout)

    def _call(self, input_value: Any) -> tuple[np.ndarray, float, float]:
        body = {
            "model": self.model,
            "input": input_value,
            "encoding_format": "float",
            "usage": {"include": True},  # ask OpenRouter to report dollar cost
        }
        t0 = time.time()
        resp = self._http.post(
            f"{self._base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        latency = time.time() - t0
        if resp.status_code != 200:
            raise RuntimeError(
                f"Embedding request failed ({resp.status_code}) for model "
                f"{self.model!r}: {resp.text[:800]}"
            )
        payload = resp.json()
        data = payload.get("data")
        if not data:
            raise RuntimeError(
                f"No embedding data from {self.model!r}. Response: {json.dumps(payload)[:800]}"
            )
        vec = np.asarray(data[0]["embedding"], dtype=np.float32)
        usage = payload.get("usage") or {}
        cost = 0.0
        for key in ("cost", "total_cost", "estimated_cost"):
            value = usage.get(key)
            if value is not None:
                try:
                    cost = float(value)
                    break
                except (TypeError, ValueError):
                    continue
        return vec, cost, latency

    def embed_text(self, text: str) -> tuple[np.ndarray, float, float]:
        return self._call(text or " ")

    def embed_image(self, path: pathlib.Path) -> tuple[np.ndarray, float, float]:
        """Embed an image as a base64 data URL. Requires a multimodal embedding model."""
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return self._call(f"data:image/jpeg;base64,{b64}")


@dataclass
class _Group:
    """Row vectors for one ordered column set."""

    fingerprint: str
    columns: list[str]
    vecs: dict[str, np.ndarray] = field(default_factory=dict)


class EmbeddingCache:
    """Loads, grows, and persists row embeddings.

    On-disk layout (unchanged filenames, so an existing cache is picked up as-is):

        {cache_dir}/{slug}_embeddings.npz    ids/text (legacy), image_ids/image, ids__{fp}/vec__{fp}
        {cache_dir}/{slug}_embed_costs.json  per-row cost/latency + the column-set registry
    """

    def __init__(
        self,
        cache_dir: pathlib.Path,
        *,
        client: EmbeddingClient,
        verbose: bool = True,
        max_workers: int = DEFAULT_EMBED_WORKERS,
    ) -> None:
        self.dir = pathlib.Path(cache_dir)
        self.client = client
        self.verbose = verbose
        self.max_workers = max(1, int(max_workers))
        self.slug = model_slug(client.model)
        self.emb_path = self.dir / f"{self.slug}_embeddings.npz"
        self.cost_path = self.dir / f"{self.slug}_embed_costs.json"

        self._groups: dict[str, _Group] = {}
        self._images: dict[str, np.ndarray] = {}
        self._row_costs: dict[str, dict] = {}
        self._column_sets: dict[str, list[str]] = {}
        self._legacy_fp: str | None = None
        # Verbatim legacy ids/text as read from disk, so a run that does not adopt them can
        # still write them back rather than dropping paid-for vectors.
        self._loaded_legacy: dict[str, Any] = {}
        self._query_cache: dict[str, np.ndarray] = {}
        self._dirty = False
        self._loaded = False

        self.row_cost = EmbedCost()
        self.query_cost = EmbedCost()
        self.n_reused = 0
        self.sunk_cost = 0.0
        # Wall time the reused vectors took when they were first computed. The dollar analogue
        # of this (`sunk_cost`) was already reported; without the seconds, a warm-cache run
        # looks fast without any record of how much work it skipped.
        self.sunk_latency = 0.0
        self.embed_wall = 0.0
        self._counted_reuse: set[tuple] = set()
        self._embedded_this_run: set[tuple] = set()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[sampling/embeddings] {msg}")

    # ------------------------------------------------------------------ load
    def load(self, *, all_text_columns: Sequence[str]) -> None:
        """Read the cache, adopting a legacy all-text-columns array where one is present."""
        if self._loaded:
            return
        self._loaded = True
        fp_all = column_fingerprint(all_text_columns)

        if self.cost_path.exists():
            try:
                with open(self.cost_path) as fh:
                    raw = json.load(fh)
                rows = raw.get("rows", raw if "embedding_model" not in raw else {})
                self._row_costs = {str(k): v for k, v in rows.items()}
                self._column_sets = {str(k): list(v) for k, v in (raw.get("column_sets") or {}).items()}
                self._legacy_fp = raw.get("legacy_text_fingerprint")
            except Exception as e:
                self._log(f"could not read {self.cost_path.name} ({e}); starting cost record fresh")

        if not self.emb_path.exists():
            return
        try:
            data = np.load(self.emb_path, allow_pickle=True)
        except Exception as e:
            self._log(f"could not read {self.emb_path.name} ({e}); starting cache fresh")
            return

        files = set(data.files)
        if "ids" in files and "text" in files:
            self._loaded_legacy = {"ids": data["ids"], "text": data["text"]}
        for name in files:
            if not name.startswith("vec__"):
                continue
            fp = name[len("vec__"):]
            id_key = f"ids__{fp}"
            if id_key not in files:
                continue
            ids = [str(x) for x in data[id_key].tolist()]
            vecs = data[name]
            self._groups[fp] = _Group(
                fingerprint=fp,
                columns=self._column_sets.get(fp, []),
                vecs={ids[i]: vecs[i] for i in range(len(ids))},
            )

        if "image_ids" in files and "image" in files:
            iids = [str(x) for x in data["image_ids"].tolist()]
            image = data["image"]
            self._images = {iids[i]: image[i] for i in range(len(iids))}

        # Legacy adoption: the previous sampler wrote one all-text-columns array under
        # ids/text with no record of which columns produced it. Adopt it as this column set's
        # group -- that is what makes the embeddings already paid for still usable -- unless a
        # previously recorded fingerprint proves it was built from different columns.
        # `all_text_columns` empty means an image-only query: there is no text column set for
        # the legacy vectors to belong to. Adopting them anyway files 2,435 real text vectors
        # under fingerprint(()) -- the SHA of the empty string -- which both mislabels them and
        # makes the next genuine text run fail the fingerprint check and re-embed the table at
        # full cost. An image-only run must leave the text cache untouched.
        if all_text_columns and fp_all not in self._groups and "ids" in files and "text" in files:
            if self._legacy_fp is not None and self._legacy_fp != fp_all:
                self._log(
                    f"legacy ids/text array was built from a different column set "
                    f"({self._legacy_fp} != {fp_all}); NOT adopting it. Those rows will be "
                    "re-embedded for the current columns."
                )
            else:
                ids = [str(x) for x in data["ids"].tolist()]
                text = data["text"]
                self._groups[fp_all] = _Group(
                    fingerprint=fp_all,
                    columns=list(all_text_columns),
                    vecs={ids[i]: text[i] for i in range(len(ids))},
                )
                for rid in ids:
                    entry = self._row_costs.setdefault(rid, {})
                    by_fp = entry.setdefault("by_fp", {})
                    by_fp.setdefault(
                        fp_all,
                        {"cost": entry.get("text_cost", 0.0), "latency": entry.get("text_latency", 0.0)},
                    )
                self._legacy_fp = fp_all
                self._dirty = True
                self._log(f"adopted {len(ids)} legacy text vectors as column set {fp_all}")

    # --------------------------------------------------------------- embedding
    def _count_reuse(self, key: tuple, sunk: Any, sunk_latency: Any = None) -> None:
        """Count one vector this run got for free because an EARLIER run had already paid.

        Two things are deliberately not counted. A vector embedded during this run and then
        re-read by a later agentic round is not reuse -- this run paid for it, and it is
        already in `embedding_rows_usd`; counting it again as sunk would report the same
        dollars twice. And the same id seen across several rounds counts once, not once per
        round. Both matter because `embedding_rows_usd_sunk` is the number used to argue the
        sampler is cheap on a warm cache.
        """
        if key in self._embedded_this_run or key in self._counted_reuse:
            return
        self._counted_reuse.add(key)
        self.n_reused += 1
        for value, attr in ((sunk, "sunk_cost"), (sunk_latency, "sunk_latency")):
            if value is None:
                continue
            try:
                setattr(self, attr, getattr(self, attr) + float(value))
            except (TypeError, ValueError):
                pass

    def _embed_batch(
        self, embed: Any, items: Sequence[tuple[str, Any]]
    ) -> Iterable[tuple[str, tuple[np.ndarray, float, float]]]:
        """Embed `items` (id, payload) concurrently, yielding results in the calling thread.

        Only the network call is parallel. Every mutation of the cache happens in the consuming
        loop, single-threaded, so no lock is needed and the accounting cannot lose an update.
        Embedding is network-bound and the GIL is released for the duration of the request, so
        threads -- not processes -- are what make this faster.

        Completion order is not input order, which is fine: `matrix()` indexes by the caller's
        id list and selection sorts by score, so nothing downstream depends on it.
        """
        t0 = time.time()
        try:
            if self.max_workers <= 1 or len(items) <= 1:
                for rid, payload in items:
                    yield rid, embed(payload)
                return
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(embed, payload): rid for rid, payload in items}
                for future in as_completed(futures):
                    yield futures[future], future.result()
        finally:
            # Wall clock, as distinct from the summed per-call service time in `row_cost`.
            # With N workers the two diverge by roughly N, so a latency number is only
            # interpretable next to the worker count that produced it.
            self.embed_wall += time.time() - t0

    def ensure_text(
        self, df: pd.DataFrame, columns: Sequence[str], id_col: str, ids: Iterable[str]
    ) -> str:
        """Embed any of `ids` missing from `columns`' group. Returns the group fingerprint."""
        from agent_cost_model.sampling.text import norm_id, row_text

        fp = column_fingerprint(columns)
        group = self._groups.get(fp)
        if group is None:
            group = _Group(fingerprint=fp, columns=list(columns))
            self._groups[fp] = group
        group.columns = list(columns)
        self._column_sets[fp] = list(columns)

        wanted = list(dict.fromkeys(ids))
        missing = [rid for rid in wanted if rid not in group.vecs]
        for rid in wanted:
            if rid in group.vecs:
                by_fp = (self._row_costs.get(rid) or {}).get("by_fp", {}).get(fp, {})
                self._count_reuse(
                    ("text", fp, rid), by_fp.get("cost"), by_fp.get("latency")
                )

        if not missing:
            return fp
        if not columns:
            return fp

        self._log(
            f"embedding {len(missing)} rows for column set {fp} ({', '.join(columns)}) "
            f"with {self.max_workers} worker(s)"
        )
        by_id = {norm_id(row[id_col]): row for _, row in df.iterrows()}
        todo = [(rid, row_text(by_id[rid], columns)) for rid in missing if rid in by_id]
        for rid, (vec, cost, latency) in self._embed_batch(self.client.embed_text, todo):
            group.vecs[rid] = vec
            self._embedded_this_run.add(("text", fp, rid))
            self.row_cost.add(cost, latency)
            entry = self._row_costs.setdefault(rid, {})
            entry.setdefault("by_fp", {})[fp] = {"cost": cost, "latency": latency}
            self._dirty = True
        return fp

    def ensure_images(self, ids: Iterable[str], image_dir: str | pathlib.Path) -> None:
        """Embed any of `ids` whose image is not cached and whose `{id}.jpg` exists."""
        directory = pathlib.Path(image_dir)
        wanted = list(dict.fromkeys(ids))
        missing = [rid for rid in wanted if rid not in self._images]
        for rid in wanted:
            if rid in self._images:
                entry = self._row_costs.get(rid) or {}
                self._count_reuse(
                    ("image", rid), entry.get("image_cost"), entry.get("image_latency")
                )
        if not missing:
            return
        found = [(rid, directory / f"{rid}.jpg") for rid in missing]
        found = [(rid, p) for rid, p in found if p.exists()]
        if not found:
            return
        self._log(f"embedding {len(found)} images with {self.max_workers} worker(s)")
        for rid, (vec, cost, latency) in self._embed_batch(self.client.embed_image, found):
            self._images[rid] = vec
            self._embedded_this_run.add(("image", rid))
            self.row_cost.add(cost, latency)
            entry = self._row_costs.setdefault(rid, {})
            entry["image_cost"], entry["image_latency"] = cost, latency
            self._dirty = True

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query string once per run; repeats across rounds are free."""
        if text in self._query_cache:
            return self._query_cache[text]
        t0 = time.time()
        vec, cost, latency = self.client.embed_text(text)
        self.embed_wall += time.time() - t0
        self.query_cost.add(cost, latency)
        self._query_cache[text] = vec
        return vec

    # ----------------------------------------------------------------- access
    def matrix(self, fingerprint: str, ids: Sequence[str]) -> np.ndarray:
        """(len(ids), D) matrix of row vectors; a missing id contributes a zero row."""
        group = self._groups.get(fingerprint)
        if group is None or not group.vecs:
            return np.zeros((len(ids), 0), dtype=np.float32)
        dim = len(next(iter(group.vecs.values())))
        out = np.zeros((len(ids), dim), dtype=np.float32)
        for i, rid in enumerate(ids):
            vec = group.vecs.get(rid)
            if vec is not None:
                out[i] = vec
        return out

    def image_matrix(self, ids: Sequence[str]) -> np.ndarray:
        if not self._images:
            return np.zeros((len(ids), 0), dtype=np.float32)
        dim = len(next(iter(self._images.values())))
        out = np.zeros((len(ids), dim), dtype=np.float32)
        for i, rid in enumerate(ids):
            vec = self._images.get(rid)
            if vec is not None:
                out[i] = vec
        return out

    def has_image(self, rid: str) -> bool:
        return rid in self._images

    def has_text(self, fingerprint: str, rid: str) -> bool:
        group = self._groups.get(fingerprint)
        return bool(group and rid in group.vecs)

    def group_count(self, fingerprint: str) -> int:
        """How many rows are cached for one column set. 0 when the set is unknown."""
        group = self._groups.get(fingerprint)
        return len(group.vecs) if group else 0

    # ------------------------------------------------------------------- save
    def save(self, *, legacy_fingerprint: str | None = None) -> None:
        """Persist the grown cache, re-merging anything a concurrent run added first."""
        if not self._dirty:
            return
        self.dir.mkdir(parents=True, exist_ok=True)

        # A sibling query of the same use case may have written new vectors since we loaded.
        # Re-read and keep whatever we do not have, so neither run's work is lost.
        if self.emb_path.exists():
            try:
                disk = np.load(self.emb_path, allow_pickle=True)
                for name in disk.files:
                    if not name.startswith("vec__"):
                        continue
                    fp = name[len("vec__"):]
                    id_key = f"ids__{fp}"
                    if id_key not in disk.files:
                        continue
                    ids = [str(x) for x in disk[id_key].tolist()]
                    vecs = disk[name]
                    group = self._groups.setdefault(
                        fp, _Group(fingerprint=fp, columns=self._column_sets.get(fp, []))
                    )
                    for i, rid in enumerate(ids):
                        group.vecs.setdefault(rid, vecs[i])
                if "image_ids" in disk.files and "image" in disk.files:
                    iids = [str(x) for x in disk["image_ids"].tolist()]
                    image = disk["image"]
                    for i, rid in enumerate(iids):
                        self._images.setdefault(rid, image[i])
            except Exception as e:
                self._log(f"could not re-merge {self.emb_path.name} before saving ({e})")

        save_kwargs: dict[str, Any] = {}
        for fp, group in self._groups.items():
            if not group.vecs:
                continue
            ids = list(group.vecs.keys())
            save_kwargs[f"ids__{fp}"] = np.asarray(ids)
            save_kwargs[f"vec__{fp}"] = np.asarray([group.vecs[i] for i in ids])

        # The all-text-columns group is ALSO written under the legacy ids/text names, so a cache
        # written here stays readable by any older tooling that expects those array names.
        # `legacy_fingerprint` names which group is the all-text one; None means there is no
        # such group (an image-only run), not "use the empty column set".
        legacy_fp = legacy_fingerprint or self._legacy_fp
        legacy_group = self._groups.get(legacy_fp) if legacy_fp else None
        if legacy_group and legacy_group.vecs:
            ids = list(legacy_group.vecs.keys())
            save_kwargs["ids"] = np.asarray(ids)
            save_kwargs["text"] = np.asarray([legacy_group.vecs[i] for i in ids])
        elif "ids" in self._loaded_legacy and "text" in self._loaded_legacy:
            # We loaded legacy arrays but did not adopt them -- a fingerprint mismatch, or an
            # image-only run that has no text columns. Write them back untouched: they are
            # embeddings someone already paid for, and this save would otherwise be the moment
            # they are silently dropped.
            save_kwargs["ids"] = self._loaded_legacy["ids"]
            save_kwargs["text"] = self._loaded_legacy["text"]

        if self._images:
            iids = list(self._images.keys())
            save_kwargs["image_ids"] = np.asarray(iids)
            save_kwargs["image"] = np.asarray([self._images[i] for i in iids])

        if not save_kwargs:
            return

        # The temp name must itself end in .npz: np.savez appends that extension when the
        # given path lacks it, which would leave os.replace pointing at a file that was never
        # written.
        tmp = self.emb_path.parent / f"{self.emb_path.stem}.tmp.{os.getpid()}.npz"
        np.savez(tmp, **save_kwargs)
        os.replace(tmp, self.emb_path)

        payload = {
            "embedding_model": self.client.model,
            "column_sets": self._column_sets,
            "legacy_text_fingerprint": legacy_fp,
            "rows": self._row_costs,
        }
        cost_tmp = self.cost_path.with_suffix(f".json.tmp.{os.getpid()}")
        with open(cost_tmp, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(cost_tmp, self.cost_path)
        self._dirty = False
