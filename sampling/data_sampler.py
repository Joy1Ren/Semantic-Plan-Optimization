"""`DataSampler`: build the optimization subset for one query.

Three modes share one scoring core (see `scoring.py` for the formula); a fourth bypasses it:

  baseline  the raw query text against every column, scored by BM25 + embedding cosine. No LLM
            call at all.
  one-shot  one LLM call writes the retrieval spec, which is then scored once.
  agentic   the same, plus up to two revisions driven by a report of what was retrieved.
  random    no LLM call, no scoring: a seeded uniform draw of `sample_size` rows, straight out
            of `_random_ids`. The no-signal floor the other three are measured against.

`sample()` never raises. Any failure -- transport, parse, scoring -- is logged and falls back
to that same seeded uniform draw rather than garbage, and the subset CSV and results record are
still written. The plan search downstream depends on that file existing.
"""
from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from agent_cost_model.opt_agent.errors import ParseError
from agent_cost_model.opt_agent.step_parsing import _parse_step
from agent_cost_model.sampling import prompts
from agent_cost_model.sampling.constants import (
    _RAG_EMBEDDING_BASE_URL,
    DEFAULT_ALPHA,
    DEFAULT_EMBED_WORKERS,
    FEEDBACK_CHAR_CEILING,
    FEEDBACK_ROW_CHARS,
    DEFAULT_MAX_ROUNDS,
    DEFAULT_RAG_EMBEDDING_MODEL,
    EXPLORE_AUTHORIZED_IMPORTS,
    EXPLORE_MAX_IMAGES,
    EXPLORE_OUTPUT_CHARS,
    IMAGE_DESCRIBER_MODEL,
    IMAGE_FIELD,
    MAX_CELL_CHARS,
    MODES,
    RESULTS_SCHEMA_VERSION,
    SUBSET_SEED,
)
from agent_cost_model.sampling.embeddings import EmbeddingCache, EmbeddingClient
from agent_cost_model.sampling.feedback import build_feedback, describe_images
from agent_cost_model.sampling.filters import compile_filter, filter_mask
from agent_cost_model.sampling.results import (
    CostLedger,
    RoundRecord,
    SamplingResult,
    StratumRecord,
    merge_results_json,
    write_rounds_trace,
    write_scores_csv,
    write_subset,
)
from agent_cost_model.sampling.scoring import (
    ScoredPool,
    bm25_scores,
    combine,
    cosine_scores,
    dead_bm25_terms,
    keyphrase_hits,
    select_topk,
    summarize,
)
from agent_cost_model.sampling.spec import SamplingSpec, Stratum, validate_plan
from agent_cost_model.sampling.text import (
    all_text_columns,
    column_fingerprint,
    norm_id,
    norm_id_series,
    row_keyphrase_text,
    row_text,
    truncate_middle,
)

__all__ = ["DataSampler"]


@dataclass
class StratumScore:
    """One stratum after scoring: what it asked for, how the pool ranked, what it took."""

    stratum: Stratum
    pool: ScoredPool
    filter_info: dict
    chosen: list[str]
    outside: list[str]
    # This stratum's own top-m, ignoring rows earlier strata already claimed. `chosen` is what
    # it contributed to the subset; `display` is what its spec actually ranks highest, and that
    # is what the agent has to see to judge whether the spec works. They differ whenever strata
    # overlap: a stratum whose top rows were all taken contributes rows from further down.
    display: list[str] = field(default_factory=list)

# One reprompt: the contract is a single fenced block, and a model that misses it once usually
# gets it when told. Beyond that, fall back rather than keep paying for retries.
_PARSE_ATTEMPTS = 2


class DataSampler:
    def __init__(
        self,
        query_id: int | str,
        use_case: str,
        query_text: str,
        df: pd.DataFrame,
        *,
        cache_dir: str,
        subset_out_path: str,
        results_path: str,
        sample_size: int,
        mode: str = "agentic",
        llm_client: Any = None,
        embedding_model: str = DEFAULT_RAG_EMBEDDING_MODEL,
        id_col: str = "idx",
        image_dir: str | None = None,
        text_cols: list[str] | None = None,
        alpha: float = DEFAULT_ALPHA,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        feedback_row_chars: int = FEEDBACK_ROW_CHARS,
        feedback_char_ceiling: int = FEEDBACK_CHAR_CEILING,
        max_cell_chars: int = MAX_CELL_CHARS,
        image_describer_model: str | None = IMAGE_DESCRIBER_MODEL,
        no_image_emb: bool = False,
        llm_model: str | None = None,
        explore: bool = True,
        idx_specification: bool = False,
        embed_workers: int = DEFAULT_EMBED_WORKERS,
        seed: int = SUBSET_SEED,
        base_url: str = _RAG_EMBEDDING_BASE_URL,
        api_key: str | None = None,
        verbose: bool = True,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode not in ("baseline", "random") and llm_client is None:
            raise ValueError(
                f"mode={mode!r} needs an llm_client; only 'baseline' and 'random' run without one"
            )
        if id_col not in df.columns:
            raise ValueError(f"id_col {id_col!r} is not a column of the given DataFrame")

        self.query_id = query_id
        self.use_case = use_case
        self.query_text = query_text
        self.mode = mode
        self.llm_client = llm_client
        self.id_col = id_col
        self.sample_size = sample_size
        self.alpha = alpha
        self.max_rounds = max(1, max_rounds)
        self.feedback_row_chars = feedback_row_chars
        self.feedback_char_ceiling = feedback_char_ceiling
        self.max_cell_chars = max_cell_chars
        self.seed = seed
        # Agentic only: one optional data-exploration call before the first spec.
        self.explore = explore and mode == "agentic"
        # Off by default; see prompts.PRESELECT_BLOCK for why naming rows is opt-in.
        self.idx_specification = idx_specification
        # What the agent asked the image describer at exploration time. Kept for the rest of
        # the run so the per-round report answers the same question the agent posed, instead of
        # reverting to a generic description once the round loop starts.
        self._image_question = ""
        self.verbose = verbose

        # Never mutate the caller's frame; ids are normalized once, here, so every downstream
        # lookup (cache keys included) agrees on what an id string looks like.
        self.df = df.copy().reset_index(drop=True)
        self.df[id_col] = norm_id_series(self.df[id_col])
        if self.df[id_col].duplicated().any():
            n_dupes = int(self.df[id_col].duplicated().sum())
            self._log(f"dropping {n_dupes} row(s) with duplicate {id_col} values")
            self.df = self.df.drop_duplicates(subset=id_col, keep="first").reset_index(drop=True)

        self.image_dir = image_dir
        self.no_image_emb = no_image_emb
        self.has_images = bool(image_dir) and not no_image_emb
        # `text_cols=[]` means an image-only query: the table carries no scorable text.
        self.text_cols = (
            all_text_columns(self.df, id_col) if text_cols is None else list(text_cols)
        )

        self.cache_dir = pathlib.Path(cache_dir)
        self.subset_out_path = pathlib.Path(subset_out_path)
        self.results_path = pathlib.Path(results_path)
        self.trace_path = (
            self.results_path.parent / "rounds" / f"Q{query_id}_{mode}.json"
        )
        self.scores_path = (
            self.results_path.parent / "scores" / f"Q{query_id}_{mode}_scores.csv"
        )

        # Read off the client so the record cannot disagree with what actually ran; the
        # explicit argument is for clients that do not expose a model id (a test double, or a
        # wrapper around several).
        self.llm_model = llm_model or getattr(llm_client, "model", None)
        self.embedding_model = embedding_model
        self.embed_workers = max(1, int(embed_workers))
        # random never scores, so it never embeds -- and both clients read their API key at
        # construction, which would demand OPENROUTER_API_KEY for a mode that makes no request.
        self._client = None
        self.cache = None
        if mode != "random":
            self._client = EmbeddingClient(embedding_model, base_url=base_url, api_key=api_key)
            self.cache = EmbeddingCache(
                self.cache_dir / "cache", client=self._client, verbose=verbose,
                max_workers=self.embed_workers,
            )

        self._image_describer = None
        if image_describer_model and self.has_images and mode != "random":
            from agent_cost_model.opt_agent.llm_client import OpenRouterClient

            self._image_describer = OpenRouterClient(image_describer_model, api_key=api_key)

        self.cost = CostLedger()
        self._messages: list[dict] = []
        self._trace: list[dict] = []
        self._score_frames: list[pd.DataFrame] = []

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[sampling] {msg}")

    def _log_plan(self, strata: list[Stratum]) -> None:
        """Print every stratum of the plan, with the share of the subset each one claims."""
        if self.mode == "baseline":
            return
        if len(strata) == 1:
            self._log_spec(strata[0].spec)
            return
        for i, st in enumerate(strata, 1):
            name = f" {st.label}" if st.label else ""
            self._log(f"  stratum {i}{name}: {st.m} of {self.sample_size} rows")
            self._log_spec(st.spec)

    def _log_spec(self, spec: SamplingSpec) -> None:
        """Print the retrieval spec a round will score with.

        Only for the LLM modes: baseline's spec is always the raw query over every column, so
        echoing it says nothing the mode name has not already said. Components the spec leaves
        empty are shown as `(unused)` rather than omitted, so the log distinguishes "the agent
        chose not to use BM25" from "the log forgot to mention it".
        """
        if self.mode == "baseline":
            return
        on = set(spec.scored_components())

        def line(label: str, value: str, cols: Sequence[str], name: str) -> str:
            if name not in on:
                return f"    {label:<11}= (unused)"
            return f"    {label:<11}= {value}  over {list(cols)}"

        self._log(line("dense", repr(spec.dense_query), spec.cols.dense, "dense"))
        self._log(line("BM25", repr(spec.bm25_query), spec.cols.bm25, "bm25"))
        self._log(line("keyphrases", str(list(spec.keyphrases)), spec.cols.keyphrase, "keyphrase"))
        self._log(f"    {'alpha':<11}= {spec.alpha}")
        if spec.filter:
            self._log(f"    {'filter':<11}= {spec.filter}")

    # ------------------------------------------------------------------ LLM
    def _llm(self, system: str, messages: list[dict]) -> tuple[str, float, float]:
        t0 = time.time()
        result = self.llm_client.generate(system, messages)
        latency = time.time() - t0
        content, meta = result, {}
        if isinstance(result, tuple):
            content = result[0] if result else ""
            if len(result) >= 3 and isinstance(result[2], dict):
                meta = result[2]
        cost = float((meta or {}).get("cost_usd", 0.0) or 0.0)
        self.cost.llm += cost
        self.cost.llm_latency += latency
        return str(content or ""), cost, latency

    def _ask_for_object(self, system: str, user: str, *, label: str) -> tuple[dict | None, int, float, float]:
        """One bounded ask for a single fenced JSON object. Returns (payload, attempts, cost, latency).

        Reprompts only for structural failures -- a missing fence or malformed JSON. Anything
        the object merely gets *wrong* is repaired by `validate_spec` instead, since a repair
        costs nothing and another round trip does.
        """
        self._messages.append({"role": "user", "content": user})
        total_cost = 0.0
        total_latency = 0.0
        for attempt in range(1, _PARSE_ATTEMPTS + 1):
            raw, cost, latency = self._llm(system, self._messages)
            total_cost += cost
            total_latency += latency
            self._messages.append({"role": "assistant", "content": raw})
            self._trace.append({"label": label, "attempt": attempt, "system": system,
                                "user": user if attempt == 1 else None, "reply": raw})
            try:
                step = _parse_step(raw)
            except ParseError as e:
                detail = e.detail
            else:
                if step.code is not None or not isinstance(step.result, dict):
                    detail = "expected a ```json``` block holding an object"
                else:
                    return step.result, attempt, total_cost, total_latency
            self._log(f"{label}: attempt {attempt} unparseable — {detail}")
            self._messages.append({"role": "user", "content": prompts.REPROMPT.format(detail=detail)})
        return None, _PARSE_ATTEMPTS, total_cost, total_latency

    # ---------------------------------------------------------- exploration
    def _explore(self) -> str:
        """One optional look at the data before the first spec is written. Agentic only.

        Returns a rendered note for the spec prompt, or "" if the agent declined or the attempt
        failed. Writing a retrieval spec blind means guessing at how values are actually spelled
        and how common they are; one cheap read of the frame turns that guess into a
        measurement. Capped at a single attempt -- this is orientation, not an analysis loop.

        The snippet runs in `LocalPythonExecutor`, the same sandbox the optimizer already uses
        for LLM-written code, with a narrow import list and only `df` in scope. Any failure is
        reported to the agent as text rather than raised: exploration is an optimization, and
        losing it must not cost the run its subset.

        On an image dataset the agent may also ask for a handful of images, described in words
        by the cheap vision model. A snippet cannot read pixels, so without this the agent
        writes the `dense_query` that the image embeddings are ranked against having never seen
        what the pictures contain.
        """
        payload, attempts, cost, latency = self._ask_for_object(
            prompts.explore_system(self.has_images),
            prompts.build_explore_user(
                query_text=self.query_text,
                df=self.df,
                id_col=self.id_col,
                has_images=self.has_images,
                sample_size=self.sample_size,
            ),
            label="explore",
        )
        self.cost.llm_by_round.append(round(cost, 8))
        if payload is None:
            self._log("exploration reply could not be parsed; continuing without it")
            return ""
        if not payload.get("explore"):
            reason = str(payload.get("reason") or "").strip()
            self._log(f"exploration skipped by the agent{f': {reason}' if reason else ''}")
            self._trace.append({"label": "explore", "skipped": True, "reason": reason})
            return ""

        image_note, image_record = self._explore_images(payload)
        code = str(payload.get("code") or "").strip()
        if not code:
            if not image_note:
                self._log("exploration requested but nothing to run or look at; continuing without it")
                return ""
            self._trace.append({"label": "explore", "attempts": attempts,
                                "latency_s": round(latency, 4), **image_record})
            return image_note

        self._log("Exploration")
        for line in code.splitlines()[:20]:
            self._log(f"    | {line}")
        output, error = self._run_explore_code(code)
        if error:
            self._log(f"    exploration failed: {error}")
        rendered = prompts.render_explore_result(code, output, error)
        if image_note:
            rendered = f"{rendered}\n\n{image_note}"
        self._trace.append(
            {"label": "explore", "code": code, "output": output, "error": error,
             "attempts": attempts, "latency_s": round(latency, 4), **image_record}
        )
        return rendered

    def _explore_images(self, payload: dict) -> tuple[str, dict]:
        """Describe a random handful of the dataset's images. Returns (rendered note, trace).

        Deliberately a random draw and not the agent's pick: this look happens before it has
        written a spec, so it has nothing to select rows on but the preview ids, and choosing
        them there would let it steer the sample it is about to be shown. Seeded with the run's
        own seed, so a re-run explores the same rows.
        """
        if not payload.get("see_images") or not self.has_images or self._image_describer is None:
            return "", {}

        ids = list(
            self.df[self.id_col].sample(
                n=min(EXPLORE_MAX_IMAGES, len(self.df)), random_state=self.seed
            )
        )
        question = str(payload.get("image_question") or "").strip()
        self._image_question = question
        self._log(f"Exploration: describing {len(ids)} image(s)"
                  + (f" — asked: {question}" if question else ""))
        t0 = time.time()
        descriptions, cost = describe_images(
            ids,
            self.image_dir,
            describer=self._image_describer,
            system=prompts.image_describe_system(question),
        )
        self.cost.image_describe += cost
        # An LLM round trip like any other; omitting it would make the reported latency
        # smaller than the wall clock.
        self.cost.llm_latency += time.time() - t0
        missing = [rid for rid in ids if rid not in descriptions]
        note = prompts.render_explore_images(descriptions, question, missing)
        return note, {"image_ids": ids, "image_question": question,
                      "image_descriptions": descriptions}

    def _run_explore_code(self, code: str) -> tuple[str, str]:
        """Execute one exploration snippet. Returns (output, error); never raises."""
        t0 = time.time()
        try:
            from agent_cost_model.opt_agent.local_python_executor import LocalPythonExecutor

            executor = LocalPythonExecutor(
                additional_authorized_imports=list(EXPLORE_AUTHORIZED_IMPORTS)
            )
            # A copy, so an in-place mutation in the snippet cannot reach the frame the rest of
            # the run scores against.
            executor.send_variables({"df": self.df.copy(), "id_col": self.id_col})
            executor.send_tools({})
            result = executor(code)
            printed = (result.logs or "").strip()
            value = "" if result.output is None else str(result.output).strip()
            parts = [p for p in (printed, value) if p]
            joined = "\n".join(parts) or "(the snippet produced no output)"
            return truncate_middle(joined, EXPLORE_OUTPUT_CHARS), ""
        except Exception as e:
            return "", f"{type(e).__name__}: {e}"
        finally:
            self.cost.scoring_latency += time.time() - t0

    # -------------------------------------------------------------- scoring
    def _score(self, spec: SamplingSpec) -> tuple[ScoredPool, dict, list[str]]:
        """Score one spec. Returns (pool, filter_info, ids_outside_filter)."""
        t0 = time.time()
        # Embedding happens inside this method, so its wall time has to come back out of the
        # scoring figure -- otherwise it lands in both `embedding_s` and `scoring_s` and
        # `total_s` reports roughly twice the real duration.
        embed_wall_before = self.cache.embed_wall
        flt, compile_error = compile_filter(spec.filter)
        if compile_error:
            self._log(f"filter {spec.filter!r} did not compile: {compile_error}")
        mask = filter_mask(self.df, flt)
        pool_df = self.df[mask]
        outside = [str(x) for x in self.df.loc[~mask, self.id_col]]
        if pool_df.empty:
            # Nothing survived, so there is nothing to rank. Keep the filter recorded and let
            # the caller fill the whole subset from outside it.
            pool_df = self.df.iloc[0:0]
        filter_info = {
            "expr": spec.filter,
            "applied": flt is not None and not compile_error,
            "compile_error": compile_error,
            "n_pool": int(len(pool_df)),
            "n_total": int(len(self.df)),
            "underfilled": bool(len(pool_df) < self.sample_size),
        }

        ids = [str(x) for x in pool_df[self.id_col]]
        live: list[str] = []
        configured: list[str] = []

        # --- sparse
        bm25_cols = [c for c in spec.cols.bm25 if c != IMAGE_FIELD]
        docs: list[str] = []
        if spec.bm25_query and bm25_cols and ids:
            configured.append("bm25")
            docs = [row_text(row, bm25_cols) for _, row in pool_df.iterrows()]
            bm25_raw = bm25_scores(spec.bm25_query, docs)
            if np.any(bm25_raw != 0.0):
                live.append("bm25")
        else:
            bm25_raw = np.zeros(len(ids))

        kp_cols = [c for c in spec.cols.keyphrase if c != IMAGE_FIELD]
        if spec.keyphrases and kp_cols and ids:
            configured.append("keyphrase")
            padded = [row_keyphrase_text(row, kp_cols) for _, row in pool_df.iterrows()]
            hits, hit_names = keyphrase_hits(spec.keyphrases, padded)
            if np.any(hits > 0):
                live.append("keyphrase")
        else:
            hits, hit_names = np.zeros(len(ids)), [[] for _ in ids]

        # --- dense
        dense_text_cols = [c for c in spec.cols.dense if c != IMAGE_FIELD]
        use_image = IMAGE_FIELD in spec.cols.dense and self.has_images
        dense_parts: list[np.ndarray] = []
        if spec.dense_query and ids and (dense_text_cols or use_image):
            configured.append("dense")
            self.cache.load(all_text_columns=self.text_cols)
            query_vec = self.cache.embed_query(spec.dense_query)
            if dense_text_cols:
                fp = self.cache.ensure_text(pool_df, dense_text_cols, self.id_col, ids)
                dense_parts.append(cosine_scores(query_vec, self.cache.matrix(fp, ids)))
            if use_image:
                self.cache.ensure_images(ids, self.image_dir)
                dense_parts.append(cosine_scores(query_vec, self.cache.image_matrix(ids)))
        # Mean, not sum: a row with no image should not be pushed to the bottom by a phantom
        # zero simply because the other rows have one.
        if dense_parts:
            dense_raw = np.mean(np.vstack(dense_parts), axis=0)
            if np.any(dense_raw != 0.0):
                live.append("dense")
        else:
            dense_raw = np.zeros(len(ids))

        zero_hit = [p for p in spec.keyphrases if not any(p in names for names in hit_names)]
        pool = combine(
            ids=ids,
            bm25_raw=bm25_raw,
            hits=hits,
            hit_names=hit_names,
            dense_raw=dense_raw,
            alpha=spec.alpha,
            live=live,
            configured=configured,
            dead_terms=dead_bm25_terms(spec.bm25_query, docs) if docs else [],
            zero_hit_keyphrases=zero_hit,
        )
        self.cost.scoring_latency += (
            (time.time() - t0) - (self.cache.embed_wall - embed_wall_before)
        )
        return pool, filter_info, outside

    # ------------------------------------------------------------- selection
    def _score_plan(
        self, strata: list[Stratum], preselected: Sequence[str] = ()
    ) -> tuple[list[StratumScore], list[str], list[str]]:
        """Score every stratum and draw its allocation. Returns (per-stratum, chosen, filled).

        Strata draw in order, each skipping rows an earlier one already claimed, so the subset
        holds `sum(m)` distinct rows rather than letting two overlapping ideas both spend their
        allocation on the same row. A stratum that cannot fill its m from what is left simply
        contributes fewer, and the shortfall is made up once at the end by `_fill`.

        `preselected` rows are claimed before any stratum draws, which is all it takes to keep
        them: the strata already exclude what is claimed, and their allocations were computed
        against the reduced size, so the subset still totals `sample_size`.
        """
        scored: list[StratumScore] = []
        chosen: list[str] = list(dict.fromkeys(preselected))
        for st in strata:
            pool, filter_info, outside = self._score(st.spec)
            picked = select_topk(pool, st.m, exclude=chosen)
            chosen.extend(picked)
            scored.append(
                StratumScore(stratum=st, pool=pool, filter_info=filter_info,
                             chosen=picked, outside=list(outside),
                             display=select_topk(pool, st.m))
            )
        # Only rows outside EVERY stratum's filter are candidates for topping up: a row one
        # stratum filtered out may be exactly what another was ranking.
        outside_all = [
            rid for rid in dict.fromkeys(r for ss in scored for r in ss.outside)
            if all(rid in set(ss.outside) for ss in scored)
        ]
        n_ranked = len(chosen)
        topped_up = self._fill(chosen, outside_all)
        return scored, topped_up, topped_up[n_ranked:]

    def _fill(self, chosen: list[str], outside: Sequence[str]) -> list[str]:
        """Top the selection up to `sample_size` with a seeded draw from the unscored rows.

        Reached when a filter leaves fewer than k rows. The filter is kept rather than
        discarded -- its survivors are exactly the rows it was written to find -- and the
        remaining slots are filled uniformly so the subset is never short. A short subset
        would silently change what downstream quality scoring compares.
        """
        if len(chosen) >= self.sample_size:
            return chosen[: self.sample_size]
        picked = set(chosen)
        remaining = [rid for rid in outside if rid not in picked]
        if not remaining:
            remaining = [str(x) for x in self.df[self.id_col] if str(x) not in picked]
        if not remaining:
            return chosen
        rng = np.random.default_rng(self.seed)
        take = min(self.sample_size - len(chosen), len(remaining))
        extra = rng.choice(remaining, size=take, replace=False)
        return chosen + [str(x) for x in extra]

    def _random_ids(self, keep: Sequence[str] = ()) -> list[str]:
        """A uniform draw of `sample_size` rows, `keep` first.

        `keep` is the preselection: rows the agent named outright. They survive the fallback
        to a uniform draw, because "no scorer could rank these rows" says nothing about the
        rows the agent picked by hand.
        """
        kept = [rid for rid in dict.fromkeys(keep)][: self.sample_size]
        rng = np.random.default_rng(self.seed)
        pool = [str(x) for x in self.df[self.id_col] if str(x) not in set(kept)]
        take = min(self.sample_size - len(kept), len(pool))
        return kept + [str(x) for x in rng.choice(pool, size=take, replace=False)]

    def _preselected(self, payload: Any) -> tuple[list[str], list[str]]:
        """Ids the agent named in `preselected_idx`. Returns (ids, warnings).

        Only read under `--idx-specification`. Ids that are not in the table are dropped and
        reported rather than carried: a hallucinated id would otherwise take a slot from the
        ranked rows and then contribute nothing, silently shrinking the subset.
        """
        if not self.idx_specification or not isinstance(payload, dict):
            return [], []
        raw = payload.get("preselected_idx") or []
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        asked = list(dict.fromkeys(norm_id(r) for r in raw))
        known = set(self.df[self.id_col])
        ids = [rid for rid in asked if rid in known]
        warnings = []
        if len(ids) != len(asked):
            unknown = [rid for rid in asked if rid not in known]
            warnings.append(
                f"preselected_idx: {unknown} are not ids in this table; dropped"
            )
        if len(ids) > self.sample_size:
            warnings.append(
                f"preselected_idx: {len(ids)} ids given but the subset holds only "
                f"{self.sample_size}; kept the first {self.sample_size}"
            )
            ids = ids[: self.sample_size]
        return ids, warnings

    # ------------------------------------------------------------------ run
    def sample(self) -> SamplingResult:
        # Start-to-end wall clock for the whole run. Measured rather than summed from the
        # component timers: those miss cache load/save, retries, image description, filtering
        # and frame work, so their sum is a lower bound on how long this actually took.
        run_start = time.time()
        rounds: list[RoundRecord] = []
        sampled: list[str] = []
        accepted_round = 1
        stop_reason = "accepted"

        try:
            sampled, rounds, accepted_round, stop_reason = self._run(rounds)
        except Exception as e:
            stop_reason = f"error:{type(e).__name__}"
            # Printed regardless of `verbose`: the run still produces a subset, but a random
            # one, and a comparison that silently comes from the wrong sampler is worse than
            # a loud failure. `rounds` keeps whatever completed before the error.
            print(
                f"[sampling] WARNING: Q{self.query_id} [{self.mode}] failed "
                f"({type(e).__name__}: {e}) — falling back to a uniform random subset."
            )
            sampled = self._random_ids()

        if not sampled:
            sampled = self._random_ids()

        # random never constructs a cache (see __init__): nothing was embedded, so the cost/
        # latency fields stay at CostLedger's zero defaults, which is the accurate answer.
        if self.cache is not None:
            try:
                # No text columns means an image-only query, which has no all-text group to
                # name. Passing fingerprint(()) here would record the SHA of the empty string
                # as the cache's legacy text fingerprint and make the next real text run disown
                # its vectors -- the same mislabeling the adoption guard in load() prevents.
                self.cache.save(
                    legacy_fingerprint=column_fingerprint(self.text_cols) if self.text_cols else None
                )
            except Exception as e:
                # Printed regardless of `verbose`: a cache that silently fails to persist means
                # every later query re-pays for embeddings it should have reused.
                print(
                    f"[sampling] WARNING: could not persist the embedding cache at "
                    f"{self.cache.emb_path} ({type(e).__name__}: {e}). The next run will re-embed."
                )

            self.cost.embedding_query = self.cache.query_cost.cost
            self.cost.embedding_rows = self.cache.row_cost.cost
            self.cost.embedding_rows_sunk = self.cache.sunk_cost
            self.cost.embedding_sunk_latency = self.cache.sunk_latency
            self.cost.embedding_latency = self.cache.query_cost.latency + self.cache.row_cost.latency
            self.cost.embedding_wall = self.cache.embed_wall
            self.cost.n_row_embeddings_computed = self.cache.row_cost.n_calls
            self.cost.n_row_embeddings_reused = self.cache.n_reused
        self.cost.embed_workers = self.embed_workers
        self.cost.total_wall = time.time() - run_start

        record = {
            "query_id": str(self.query_id),
            "mode": self.mode,
            "sampler": "DataSampler",
            "schema_version": RESULTS_SCHEMA_VERSION,
            "use_case": self.use_case,
            "embedding_model": self.embedding_model,
            "id_col": self.id_col,
            "sample_size": self.sample_size,
            "seed": self.seed,
            "no_image_emb": self.no_image_emb,
            "idx_specification": self.idx_specification,
            "embed_workers": self.embed_workers,
            "n_rows_total": int(len(self.df)),
            "n_rounds": len(rounds),
            "stop_reason": stop_reason,
            "sampled_ids": sampled,
            "cost": self.cost.to_json(self.mode),
            "latency": self.cost.latency_json(self.mode),
            # Recorded so a later run_opt (or an analysis notebook) can find what this run
            # produced without re-deriving the layout from benchmark.yaml.
            "artifacts": {
                "subset": str(self.subset_out_path),
                "scores_csv": str(self.scores_path),
            },
            "rounds": [r.to_json(self.mode) for r in rounds],
        }
        # Only agentic can end on a round other than its last, so elsewhere the field would
        # just restate n_rounds.
        # Which model planned the retrieval. Baseline has none, so the key would be a null
        # standing in for a step that never happened.
        if self.mode != "baseline" and self.llm_model:
            record["sampler_llm_model"] = self.llm_model
        # A third model, billed separately under image_describe_usd, so it is named wherever it
        # actually ran rather than left as the one paid-for step with no attribution.
        if self._image_describer is not None and self.cost.image_describe > 0:
            record["image_describer_model"] = getattr(self._image_describer, "model", None)
        if self.mode == "agentic":
            record["accepted_round"] = accepted_round
        if self._trace:
            record["artifacts"]["trace"] = str(self.trace_path)

        write_subset(self.df, self.id_col, sampled, self.subset_out_path)
        merge_results_json(self.results_path, str(self.query_id), self.mode, record)
        write_scores_csv(self.scores_path, self._score_frames)
        if self._trace:
            write_rounds_trace(
                self.trace_path,
                {"query_id": str(self.query_id), "mode": self.mode, "steps": self._trace},
            )

        self._log(
            f"Q{self.query_id} [{self.mode}] selected {len(sampled)} rows "
            f"({stop_reason}, {len(rounds)} round(s)) -> {self.subset_out_path}"
        )
        return SamplingResult(
            sampled_ids=sampled,
            subset_path=self.subset_out_path,
            results_path=self.results_path,
            scores_path=self.scores_path,
            mode=self.mode,
            rounds=rounds,
            accepted_round=accepted_round,
            stop_reason=stop_reason,
            cost=self.cost,
            record=record,
        )

    def _run(self, rounds: list[RoundRecord]) -> tuple[list[str], list[RoundRecord], int, str]:
        self._log(f"Starting sampling on mode {self.mode}")
        if self.mode == "random":
            # No LLM, no BM25/embedding scoring, no strata: literally a seeded uniform draw.
            # `rounds` stays empty -- nothing here is a round of anything.
            return self._random_ids(), rounds, 1, "accepted"
        default_plan = [
            Stratum(
                spec=SamplingSpec.default(
                    df=self.df,
                    id_col=self.id_col,
                    query_text=self.query_text,
                    alpha=self.alpha,
                    has_images=self.has_images,
                    text_columns=self.text_cols,
                ),
                m=self.sample_size,
            )
        ]

        strata = default_plan
        spec_raw: dict = {}
        warnings: list[str] = []
        preselected: list[str] = []
        attempts = 1
        round_cost = 0.0
        round_latency = 0.0
        stop_reason = "accepted"

        if self.mode != "baseline":
            exploration = self._explore() if self.explore else ""
            user = prompts.build_spec_user(
                query_text=self.query_text,
                df=self.df,
                id_col=self.id_col,
                has_images=self.has_images,
                sample_size=self.sample_size,
                exploration=exploration,
            )
            payload, attempts, round_cost, round_latency = self._ask_for_object(
                prompts.spec_system(self.sample_size, self.idx_specification), user, label="spec"
            )
            if payload is None:
                self._log("could not get a usable plan; falling back to the baseline spec")
                stop_reason = "parse_failed"
            else:
                spec_raw = payload
                preselected, pre_warnings = self._preselected(payload)
                strata, warnings = validate_plan(
                    payload,
                    df=self.df,
                    id_col=self.id_col,
                    query_text=self.query_text,
                    sample_size=self.sample_size - len(preselected),
                    default_alpha=self.alpha,
                    has_images=self.has_images,
                    text_columns=self.text_cols,
                )
                warnings = pre_warnings + warnings
                for w in warnings:
                    self._log(f"plan repaired: {w}")
                if preselected:
                    self._log(f"    preselected {len(preselected)} row(s) by id: {preselected}")

        self._log("Round 1")
        self._log_plan(strata)

        if not strata:
            chosen = self._log_and_draw_uniform(preselected)
            rounds.append(
                RoundRecord(
                    round=1, spec_raw=spec_raw, warnings=warnings, attempts=attempts,
                    chosen_ids=chosen, llm_cost_usd=round(round_cost, 8),
                    llm_latency_s=round(round_latency, 4),
                )
            )
            self.cost.llm_by_round.append(round(round_cost, 8))
            return chosen, rounds, 1, self._no_plan_reason(preselected)

        # Fixed after round 1 across every stratum: re-embedding the table is the one change a
        # revision cannot make cheaply.
        pinned_dense = strata[0].spec.cols.dense
        scored, chosen, filled = self._score_plan(strata, preselected)
        rounds.append(
            self._record(1, scored, spec_raw, warnings, attempts, chosen, filled,
                         round_cost, round_latency)
        )
        self._score_frames.append(self._scores_frame(1, scored))
        self.cost.llm_by_round.append(round(round_cost, 8))
        accepted_round = 1

        if self.mode != "agentic" or stop_reason == "parse_failed":
            return chosen, rounds, accepted_round, stop_reason

        for attempt in range(2, self.max_rounds + 1):
            report = self._build_report(scored, chosen, warnings)
            payload, n_attempts, cost, latency = self._ask_for_object(
                prompts.revise_system(self.sample_size, self.idx_specification),
                prompts.build_revise_user(report, attempt=attempt - 1, max_attempts=self.max_rounds - 1),
                label=f"revise-{attempt}",
            )
            self.cost.llm_by_round.append(round(cost, 8))
            if payload is None:
                rounds[-1].action = "accept"
                rounds[-1].reason = "revision reply could not be parsed; kept the previous rows"
                return chosen, rounds, accepted_round, "parse_failed"

            reason = str(payload.get("reason") or "").strip()
            if payload.get("accept") is True:
                rounds[-1].action = "accept"
                rounds[-1].reason = reason
                self._log(f"Round {attempt - 1} accepted: {reason or '(no reason given)'}")
                return chosen, rounds, accepted_round, "accepted"

            rounds[-1].action = "resample"
            rounds[-1].reason = reason
            spec_raw = payload
            preselected, pre_warnings = self._preselected(payload)
            strata, warnings = validate_plan(
                payload,
                df=self.df,
                id_col=self.id_col,
                query_text=self.query_text,
                sample_size=self.sample_size - len(preselected),
                default_alpha=self.alpha,
                has_images=self.has_images,
                pinned_dense=pinned_dense,
                text_columns=self.text_cols,
            )
            warnings = pre_warnings + warnings
            for w in warnings:
                self._log(f"plan repaired: {w}")
            if preselected:
                self._log(f"    preselected {len(preselected)} row(s) by id: {preselected}")
            self._log(f"Round {attempt}")
            if reason:
                self._log(f"    resampling because: {reason}")
            self._log_plan(strata)

            if not strata:
                chosen = self._log_and_draw_uniform(preselected)
                rounds.append(
                    RoundRecord(
                        round=attempt, spec_raw=spec_raw, warnings=warnings,
                        attempts=n_attempts, chosen_ids=chosen,
                        llm_cost_usd=round(cost, 8), llm_latency_s=round(latency, 4),
                    )
                )
                return chosen, rounds, attempt, self._no_plan_reason(preselected)

            scored, chosen, filled = self._score_plan(strata, preselected)
            accepted_round = attempt
            rounds.append(
                self._record(attempt, scored, spec_raw, warnings, n_attempts, chosen, filled,
                             cost, latency)
            )
            self._score_frames.append(self._scores_frame(attempt, scored))

        return chosen, rounds, accepted_round, "max_rounds"

    def _log_and_draw_uniform(self, keep: Sequence[str] = ()) -> list[str]:
        """Uniform draw for a plan with nothing to rank by, plus the line that explains it."""
        if len(keep) >= self.sample_size:
            self._log("    the preselected rows fill the subset; nothing left to rank")
        else:
            self._log(
                "    no scorer requested — every row is equally relevant, "
                "sampling uniformly instead"
            )
        return self._random_ids(keep)

    def _no_plan_reason(self, preselected: Sequence[str]) -> str:
        """Why a round ended with no strata to score.

        Distinguished because the two mean opposite things downstream: `random_no_target` is an
        unguided subset, while a preselection that fills k is the most deliberate subset the
        agent can produce. Recording both as `random_no_target` would make a hand-picked run
        read as a random one in any later comparison.
        """
        return "preselected_full" if len(preselected) >= self.sample_size else "random_no_target"

    def _build_report(
        self, scored: list["StratumScore"], chosen: list[str], warnings: list[str]
    ) -> str:
        """The report the agent judges a round by: every selected row, grouped by stratum."""
        descriptions: dict[str, str] = {}
        wants_images = any(
            IMAGE_FIELD in ss.stratum.spec.cols.dense for ss in scored
        )
        if self.has_images and wants_images and self._image_describer:
            t0 = time.time()
            descriptions, cost = describe_images(
                list(dict.fromkeys([r for ss in scored for r in ss.display] + chosen)),
                self.image_dir,
                describer=self._image_describer,
                system=prompts.image_describe_system(self._image_question),
            )
            self.cost.image_describe += cost
            # Billed to llm_latency: it is an LLM round trip, and leaving it out would make
            # the reported total latency smaller than the wall clock.
            self.cost.llm_latency += time.time() - t0
        return build_feedback(
            sections=[
                (ss.stratum.label, ss.stratum.spec, ss.pool, ss.filter_info, ss.chosen)
                for ss in scored
            ],
            own_tops=[ss.display for ss in scored],
            df=self.df,
            id_col=self.id_col,
            chosen=chosen,
            # Scales with the subset so each row keeps its space as k grows, capped so a wide
            # table with a large k cannot run away.
            budget=min(self.feedback_char_ceiling,
                       self.feedback_row_chars * max(1, len(chosen))),
            max_cell=self.max_cell_chars,
            image_descriptions=descriptions,
            warnings=warnings,
        )

    def _scores_frame(
        self, index: int, plan: list["StratumScore"]
    ) -> pd.DataFrame:
        """Every scored row of one round, per stratum, with raw and normalized components.

        One block per stratum, since each scores the same pool differently -- a single pooled
        table would have no way to say which spec produced a given score. Ordered by stratum
        then rank so the file opens on the rows that were actually selected. The spec is
        repeated on each row rather than referenced, so the CSV stands alone in a notebook.
        """
        frames: list[pd.DataFrame] = []
        for si, ss in enumerate(plan, 1):
            pool, spec, chosen = ss.pool, ss.stratum.spec, ss.chosen
            selected = set(chosen)
            order = np.argsort(-pool.total, kind="stable")
            rank = np.empty(len(pool.ids), dtype=int)
            rank[order] = np.arange(1, len(pool.ids) + 1)
            frames.append(pd.DataFrame({
                "round": index,
                "stratum": si,
                "stratum_label": ss.stratum.label,
                "stratum_m": ss.stratum.m,
                self.id_col: pool.ids,
                "rank": rank,
                "bm25_raw": pool.bm25_raw,
                "bm25_norm": pool.bm25_norm,
                "keyphrase_hits": pool.hits.astype(int),
                "keyphrase_matched": [";".join(n) for n in pool.hit_names],
                "cosine_raw": pool.dense_raw,
                "dense_norm": pool.dense_norm,
                "sparse": pool.sparse,
                "score": pool.total,
                "selected": [rid in selected for rid in pool.ids],
                "mode": self.mode,
                "alpha": spec.alpha,
                "sample_size": self.sample_size,
                "embedding_model": self.embedding_model,
                "filter": spec.filter or "",
                "bm25_query": spec.bm25_query,
                "dense_query": spec.dense_query,
                "keyphrases": ";".join(spec.keyphrases),
                "bm25_columns": ";".join(spec.cols.bm25),
                "keyphrase_columns": ";".join(spec.cols.keyphrase),
                "dense_columns": ";".join(spec.cols.dense),
            }).sort_values("rank", ignore_index=True))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _chosen_scores(self, pool: ScoredPool, chosen: Sequence[str]) -> list[dict]:
        by_id = {rid: i for i, rid in enumerate(pool.ids)}
        configured = set(pool.configured)
        out: list[dict] = []
        for rid in chosen:
            i = by_id.get(rid)
            if i is None:
                out.append({"id": rid, "source": "fill"})
                continue
            # Same rule as the summary: a subscore from a component that never ran is a
            # structural zero, not a measurement of this row.
            entry = {"id": rid, "total": round(float(pool.total[i]), 6)}
            if "bm25" in configured or "keyphrase" in configured:
                entry["sparse"] = round(float(pool.sparse[i]), 6)
            if "bm25" in configured:
                entry["bm25_norm"] = round(float(pool.bm25_norm[i]), 6)
            if "keyphrase" in configured:
                entry["keyphrase_hits"] = int(pool.hits[i])
            if "dense" in configured:
                entry["dense_norm"] = round(float(pool.dense_norm[i]), 6)
            out.append(entry)
        return out

    def _record(
        self,
        index: int,
        plan: list["StratumScore"],
        spec_raw: dict,
        warnings: list[str],
        attempts: int,
        chosen: list[str],
        fill_ids: list[str],
        cost: float,
        latency: float,
    ) -> RoundRecord:
        return RoundRecord(
            round=index,
            spec_raw=spec_raw,
            strata=[
                StratumRecord(
                    m=ss.stratum.m,
                    label=ss.stratum.label,
                    spec=ss.stratum.spec.to_json(ss.pool.configured),
                    filter_info=ss.filter_info,
                    scores=summarize(ss.pool),
                    chosen_ids=ss.chosen,
                    chosen_scores=self._chosen_scores(ss.pool, ss.chosen),
                )
                for ss in plan
            ],
            warnings=warnings,
            attempts=attempts,
            chosen_ids=chosen,
            fill_ids=fill_ids,
            llm_cost_usd=round(cost, 8),
            llm_latency_s=round(latency, 4),
        )
