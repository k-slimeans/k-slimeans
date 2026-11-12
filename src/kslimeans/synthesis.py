"""The synthesis (issue #50): summary quality, centroid representation, clustering outcome and cost, per cluster.

One row per (dataset, method, seed, summary step, cluster). The row joins

- **summary quality** (#34, `summary_quality`): coverage, genericity, unsupported content and fluency proxies;
- **centroid representation** (#33, `diagnostic` / `replay`): how far the summary moved the centroid, whether it is
  still nearest to its own cluster, members lost and gained at the replacement and over the Lloyd block;
- **clustering outcome**: the cluster's purity and majority intent before and after the step, from the LOGGED
  partitions, and each run's final ACC / NMI / ARI;
- **cost** (#35, `costcheck`): the input and output tokens of the cluster's prompt and summary. A cluster the loop
  did not send (a memo-cache hit) costs 0 and is flagged `sent == False`.

Indexing. #34 numbers summary steps from 0, #33 from 1 (step 1 is the first summary). The joined frame uses #33's
numbering in `step` and keeps #34's in `quality_step`. Runs are matched on `run_config` (the config name that #33
calls `run`; #34's `run` is `<config name>/<timestamp>`), never on the method, whose #34 label differs from #33's raw name.

What is exact. At step 1 the pre-replacement state is replayed exactly (#33 section 2), so the replacement-time
columns (`drift`, `cosine`, `own_nearest`, `empty_replacement`, `size_replacement`, `n_out_replacement`,
`n_in_replacement`) are exact with the GPU encode and exact up to encoder noise with the CPU fallback. From step 2 on
they rest on replayed previous blocks, which reproduce the logged partitions when the summary embeddings are the GPU
encode (`summary_embeddings_cuda.npz`) but, for the 4B, often not with the CPU fallback
(`summary_embeddings_cpu.npz`): `step1` marks the step-1 rows,
`pre_matches_source` says whether the replayed pre-replacement cluster has the members the summary was written from
(`size_pre == n_members`), and `end_matches_logged` whether the replayed end-of-block size equals the logged one
(`size_end == n_after`). The end-of-block columns of #33 (`size_end`, `n_out_end`, `n_in_end`, `jaccard_end_vs_pre`,
`to_end_radii`, `end_vs_pre_radii`) are replayed at every step; the outcome columns below come from the logged partitions
at every step and are exact.

Statistics. Clusters of one seed are not independent samples, so every interval here treats the SEED as the unit: a
statistic is computed within each seed, and the interval is the Student-t interval of the per-seed values
(`seed_interval`). With five seeds the interval has four degrees of freedom and is wide on purpose.

Old-format runs (BANKING77 before #30) have no per-step assignments: they carry the last summary step scored against the
FINAL partition (`approx_partition`), no #33 columns, no outcome columns and approximate costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .runlog import members_by_cluster

KEYS = ["run_config", "seed", "step", "cluster"]
"""The join key of the per-cluster files, with `step` numbered from 1."""

RUN_KEYS = ["run", "seed", "step", "cluster"]

HELP_DELTA = 0.1
"""A cluster is `helped` when its members' purity (`d_member_purity`) rose by at least this much over the step,
`harmed` when it fell by as much."""

MIN_PER_SEED = 3
"""Clusters a seed needs in a stratum (or in each group of a contrast) to contribute its per-seed value."""

MIN_SEEDS = 3
"""Seeds a table cell needs for its mean and interval to be reported (`summarise_seeds`); fewer give NaN."""

# ---------------------------------------------------------------------------------------------------- join


class JoinError(AssertionError):
    """A join that was meant to be one-to-one was not."""


def one_to_one_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: list[str],
    name: str,
    *,
    left_only_ok: bool = False,
    right_only_ok: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Join `right` onto `left` on `on`, asserting that each key occurs at most once on either side.

    Rows of `left` without a partner raise `JoinError` unless `left_only_ok` (they are kept with NaN); rows of `right`
    without a partner raise unless `right_only_ok` (they are dropped). Nothing is dropped silently: the second return
    value counts `n_left`, `n_right`, `matched`, `left_only` and `right_only`.
    """
    for side, frame in (("left", left), ("right", right)):
        dup = frame.duplicated(on, keep=False)
        if dup.any():
            raise JoinError(f"{name}: {int(dup.sum())} rows of the {side} frame share a key on {on}")
    clash = sorted((set(left.columns) & set(right.columns)) - set(on))
    if clash:
        raise JoinError(f"{name}: both frames have the non-key column(s) {clash}")
    merged = left.merge(right, on=on, how="outer", indicator=True)
    counts = merged["_merge"].value_counts()
    report = {
        "join": name,
        "n_left": len(left),
        "n_right": len(right),
        "matched": int(counts.get("both", 0)),
        "left_only": int(counts.get("left_only", 0)),
        "right_only": int(counts.get("right_only", 0)),
    }
    if report["left_only"] and not left_only_ok:
        raise JoinError(f"{name}: {report['left_only']} left rows have no partner ({report})")
    if report["right_only"] and not right_only_ok:
        raise JoinError(f"{name}: {report['right_only']} right rows have no partner ({report})")
    merged = merged[merged["_merge"] != "right_only"].drop(columns="_merge")
    if len(merged) != len(left):  # pragma: no cover - guarded by the duplicate checks above
        raise JoinError(f"{name}: the join changed the number of rows ({len(left)} -> {len(merged)})")
    return merged.reset_index(drop=True), report


def prepare_quality(quality: pd.DataFrame) -> pd.DataFrame:
    """#34's per-cluster frame with `run_config`, `quality_step` (0-indexed, as logged) and `step` (1-indexed)."""
    out = quality.copy()
    out["run_config"] = out["run"].str.split("/").str[0]
    out["quality_step"] = out["step"]
    out["step"] = out["quality_step"] + 1
    return out


DIAG_DROP = ("method", "dup", "meta", "flagged")
"""#33 columns not carried: the method name (a different spelling than #34's) and the #33 surface flags, which #34's
`flag_duplicate` / `flag_mentions` supersede."""


def join_quality_diagnostic(
    quality: pd.DataFrame, diag: pd.DataFrame, stale: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, list[dict]]:
    """#34 rows with their #33 columns, run by run.

    `quality` is `prepare_quality`'s output. Runs present in `diag` must match one to one in both directions (every
    scored cluster has its #33 row and vice versa); runs absent from `diag` (old-format runs) keep their rows with
    `has_diag == False` and NaN #33 columns. `stale` (optional, #33's `_stale.csv`, same key as `diag`) is joined to
    `diag` first. A #33 run that #34 never scored raises. Returns the frame and one report row per run.
    """
    diag = diag.rename(columns={"run": "run_config"}).drop(columns=[c for c in DIAG_DROP if c in diag])
    reports: list[dict] = []
    if stale is not None:
        stale = stale.rename(columns={"run": "run_config"}).drop(columns=[c for c in DIAG_DROP if c in stale])
        diag, rep = one_to_one_join(diag, stale, KEYS, "diagnostic x stale")
        reports.append(rep)
    unscored = sorted(set(diag["run_config"]) - set(quality["run_config"]))
    if unscored:
        raise JoinError(f"#33 runs with no #34 scores: {unscored}")
    parts = []
    for run_config, rows in quality.groupby("run_config", sort=False):
        part = diag[diag["run_config"] == run_config]
        if len(part) == 0:
            rows = rows.assign(has_diag=False)
            reports.append(
                {"join": f"quality x diagnostic [{run_config}]", "n_left": len(rows), "n_right": 0, "matched": 0,
                 "left_only": len(rows), "right_only": 0}
            )
        else:
            rows, rep = one_to_one_join(rows, part, KEYS, f"quality x diagnostic [{run_config}]")
            rows = rows.assign(has_diag=True)
            reports.append(rep)
        parts.append(rows)
    return pd.concat(parts, ignore_index=True), reports


def attach_by_key(
    frame: pd.DataFrame, extra: pd.DataFrame, on: list[str], name: str, *, extra_unmatched_ok: bool = False
) -> tuple[pd.DataFrame, dict]:
    """Add the columns of `extra` (one row per key) to `frame`; every row of `frame` must find its partner.

    `extra` may have rows `frame` lacks only when `extra_unmatched_ok` (the cost frame covers empty clusters too).
    """
    return one_to_one_join(frame, extra, on, name, right_only_ok=extra_unmatched_ok)


def attach_run_level(frame: pd.DataFrame, run_level: pd.DataFrame, on: list[str], name: str) -> tuple[pd.DataFrame, dict]:
    """Add per-(run, seed) columns to every cluster row (many to one); a row with no partner raises."""
    if run_level.duplicated(on).any():
        raise JoinError(f"{name}: the run-level frame has duplicate keys on {on}")
    merged = frame.merge(run_level, on=on, how="left", indicator=True, validate="many_to_one")
    n_missing = int((merged["_merge"] != "both").sum())
    report = {"join": name, "n_left": len(frame), "n_right": len(run_level), "matched": len(frame) - n_missing,
              "left_only": n_missing, "right_only": 0}
    if n_missing:
        raise JoinError(f"{name}: {n_missing} cluster rows have no run-level partner")
    return merged.drop(columns="_merge"), report


# ----------------------------------------------------------------------------------- per-run extras (outcome, cost)


def after_overlap(source: np.ndarray, after: np.ndarray, gold: np.ndarray, k: int) -> pd.DataFrame:
    """Per cluster, how the step changed its majority gold intent.

    `source` is the partition the summary was written from, `after` the partition after the step's Lloyd block.
    `majority_before` is the lowest-id most common gold intent of the source cluster (#34's `majority_intent`),
    `majority_after` the same for the after cluster, `agree_majority` whether they are equal, `major_share_after`
    the share of the after cluster holding the BEFORE majority intent, and `jaccard_major_after` the Jaccard overlap
    of the after cluster with that intent's members (compare #34's `jaccard_majority`, the same quantity before).
    `member_agree_after` follows the MEMBERS instead of the cluster id: the share of the source cluster's members whose
    gold intent is the majority intent of the cluster they end in (before the step this share is the purity, so the
    difference is the cluster's contribution to the change in overall purity; a cluster that lost everyone is still
    scored, whereas `purity_after` then describes a different set of documents under the same id).
    Clusters empty before or after give NaN (`majority_*` None).
    """
    gold = np.asarray(gold)
    n_gold = int(gold.max()) + 1
    class_sizes = np.bincount(gold, minlength=n_gold)
    before = members_by_cluster(source, k)
    post = members_by_cluster(after, k)
    majority_after = np.array(
        [np.bincount(gold[m], minlength=n_gold).argmax() if len(m) else -1 for m in post], dtype=np.int64
    )
    rows = []
    for c in range(k):
        row = {"cluster": c, "majority_before": None, "majority_after": None, "agree_majority": np.nan,
               "major_share_after": np.nan, "jaccard_major_after": np.nan, "member_agree_after": np.nan}
        if len(before[c]):
            row["member_agree_after"] = float(np.mean(gold[before[c]] == majority_after[after[before[c]]]))
        if len(before[c]):
            row["majority_before"] = int(np.bincount(gold[before[c]], minlength=n_gold).argmax())
        if len(post[c]):
            counts = np.bincount(gold[post[c]], minlength=n_gold)
            row["majority_after"] = int(counts.argmax())
            if row["majority_before"] is not None:
                g = row["majority_before"]
                row["agree_majority"] = float(row["majority_after"] == g)
                row["major_share_after"] = counts[g] / len(post[c])
                row["jaccard_major_after"] = counts[g] / (len(post[c]) + class_sizes[g] - counts[g])
        rows.append(row)
    return pd.DataFrame(rows)


def cost_rows(k: int, sent: list[int], input_tokens: dict[int, int], output_tokens: dict[int, int]) -> pd.DataFrame:
    """One row per cluster of a step: `sent`, `input_tokens`, `output_tokens`.

    `input_tokens[c]` / `output_tokens[c]` hold the counts of the sent clusters; a cluster not in `sent` (empty, or a
    memo-cache hit whose summary was reused) costs 0 and has `sent == False`.
    """
    sent_set = set(sent)
    return pd.DataFrame(
        {
            "cluster": range(k),
            "sent": [c in sent_set for c in range(k)],
            "input_tokens": [input_tokens.get(c, 0) if c in sent_set else 0 for c in range(k)],
            "output_tokens": [output_tokens.get(c, 0) if c in sent_set else 0 for c in range(k)],
        }
    )


def record_cost_rows(record, texts, X, force_context_length: int, prompt_tokens, output_tokens) -> pd.DataFrame:
    """Per summary step and cluster of a record that logged its assignments: `sent`, `input_tokens`, `output_tokens`.

    `sent_clusters` and `step_inputs` (the live member selection and memo rule) say which clusters were sent and with
    which texts; `prompt_tokens(texts)` and `output_tokens(summary)` count tokens (the script passes tokenizer-based
    counters, tests pass fakes). Steps are numbered from 1; a cluster that was not sent costs 0.
    """
    from .runlog import sent_clusters, step_inputs

    parts = []
    for j in range(record.n_summary_steps):
        summaries = record.summaries_evolution[j]
        sent = sent_clusters(record, j, texts, X, force_context_length)
        inputs = step_inputs(record, j, texts, X, force_context_length)
        n_in = {c: prompt_tokens(inputs[c].texts) for c in sent}
        n_out = {c: output_tokens(summaries[c]) for c in sent}
        parts.append(cost_rows(len(summaries), sent, n_in, n_out).assign(step=j + 1, cost_approx=False))
    return pd.concat(parts, ignore_index=True)


def member_agreement(source: np.ndarray, new: np.ndarray, gold: np.ndarray, k: int) -> np.ndarray:
    """Per source cluster, the share of its members whose gold intent is the majority intent of their cluster in `new`.

    `new = source` gives the purity. The majority of a cluster is its lowest-id most common intent; NaN for an empty
    source cluster.
    """
    gold = np.asarray(gold)
    counts = np.zeros((k, int(gold.max()) + 1))
    np.add.at(counts, (new, gold), 1)
    agree = (gold == counts.argmax(axis=1)[new]).astype(float)
    size = np.bincount(source, minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.bincount(source, weights=agree, minlength=k) / np.where(size == 0, np.nan, size)


def shuffle_null_labels(source: np.ndarray, lost: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """`source` with, for every cluster `c`, `lost[c]` randomly chosen members sent to a uniformly random other cluster."""
    k = len(lost)
    new = source.copy()
    for c in range(k):
        idx = np.flatnonzero(source == c)
        n = int(min(lost[c], len(idx)))
        if n == 0 or k < 2:
            continue
        moved = rng.choice(idx, size=n, replace=False)
        dest = rng.integers(0, k - 1, size=n)
        new[moved] = dest + (dest >= c)  # skip the cluster itself
    return new


def reference_rows(
    source: np.ndarray, after: np.ndarray, counterfactual: np.ndarray, gold: np.ndarray, k: int, n_draws: int = 20, seed: int = 0
) -> pd.DataFrame:
    """The two references for one summary step (used at step 1), one row per cluster.

    `counterfactual` is the partition plain Lloyd reaches from the pre-replacement state (no replacement). The random
    null moves each cluster's members at the cluster's own logged loss (`after`) to random destinations, averaged over
    `n_draws` draws. Both are scored like the outcome: `cf_lost_share`, `cf_member_agree_after`,
    `null_member_agree_after` (compare `member_agree_after`; before the step the same quantity is the purity).
    """
    size = np.bincount(source, minlength=k).astype(float)
    size[size == 0] = np.nan
    lost = np.array([np.sum(after[source == c] != c) for c in range(k)])
    cf_lost = np.array([np.sum(counterfactual[source == c] != c) for c in range(k)]) / size
    rng = np.random.default_rng(seed)
    null = np.mean([member_agreement(source, shuffle_null_labels(source, lost, rng), gold, k) for _ in range(n_draws)], axis=0)
    return pd.DataFrame(
        {"cluster": range(k), "cf_lost_share": cf_lost, "cf_member_agree_after": member_agreement(source, counterfactual, gold, k),
         "null_member_agree_after": null}
    )  # fmt: skip


def counterfactual_labels(X: np.ndarray, pre_centers: np.ndarray, n_iter: int) -> np.ndarray:
    """Labels after `n_iter` plain Lloyd iterations from `pre_centers` (#33's no-replacement counterfactual)."""
    from .replay import LloydReplayer

    return LloydReplayer(X).run_block(pre_centers, n_iter, stop=False).end.labels


def run_extras(run_dir, cache_dir, load_tokenizer=None, cf_cache: dict | None = None) -> dict[str, pd.DataFrame]:
    """Outcome, cost, reference and run-level frames of one run directory (CPU; tokenizer only, no model, no GPU).

    Returns `after`, `cost` and `ref` (one row per run, seed, step, cluster; steps numbered from 1; `ref` only for step
    1, see `reference_rows`) and `runs` (one row per (run, seed): the final metrics, the k-means baseline's, the run
    totals). `load_tokenizer(model_id, trust_remote_code)` supplies the tokenizer of an `hf` summarizer. Old-format runs
    get no `after` or `ref` rows, and their costs are rebuilt from the FINAL partition (`cost_approx`; every non-empty
    cluster is assumed sent). The counterfactual is the same for every method of a seed (the step-0 state is shared),
    so `cf_cache` lets several runs reuse one k-means++ fit.
    """
    from .core import _select_cluster
    from .costcheck import PromptSettings, count_tokens, rebuild_prompt
    from .replay import fit_initial
    from .runlog import load_run, load_run_dataset
    from .summarizers.hf import supports_system_role

    cf_cache = {} if cf_cache is None else cf_cache
    runlog = load_run(run_dir)
    cfg = runlog.config
    run = f"{cfg['name']}/{runlog.path.name}"
    ds, X = load_run_dataset(runlog, cache_dir)
    scfg = cfg["summarizer"]
    tokenizer = settings = None
    fold_system = False
    if scfg.get("type") == "hf":
        tokenizer = load_tokenizer(scfg["model_id"], bool(scfg.get("trust_remote_code", False)))
        settings = PromptSettings.from_run(scfg, ds.prompt, ds.text_type)
        fold_system = not supports_system_role(tokenizer, settings.chat_template_kwargs)
    fcl = cfg.get("force_context_length", 0)

    def prompt_tokens(texts: list[str]) -> int:
        return rebuild_prompt(tokenizer, settings, texts, fold_system).input_tokens if tokenizer else 0

    def output_tokens(summary: str) -> int:
        return count_tokens(tokenizer, summary) if tokenizer else 0

    after_parts, cost_parts, ref_parts, run_rows = [], [], [], []
    baselines = {r.seed: r.metrics for r in runlog.records.values() if not r.summaries_evolution}
    for method in runlog.methods:
        for record in runlog.method_records(method):
            if not record.summaries_evolution:
                continue
            key = {"run": run, "seed": record.seed}
            if record.assignments_evolution is not None and record.summary_source_steps is not None:
                cost_parts.append(record_cost_rows(record, ds.texts, X, fcl, prompt_tokens, output_tokens).assign(**key))
                for j in range(record.n_summary_steps):
                    try:
                        post = record.partition_after(j)
                    except IndexError:  # a final step that stopped on tol has no block after it
                        continue
                    k = len(record.summaries_evolution[j])
                    source = record.source_partition(j)
                    after_parts.append(after_overlap(source, post, ds.labels, k).assign(**key, step=j + 1))
                    if j == 0:
                        by_step = {row["step"]: row for row in record.trace}
                        block = by_step[record.summary_source_steps[0] + 1]
                        ckey = (ds.name, record.seed, k, by_step[0]["last_max_iter"], block["last_n_iter"])
                        if ckey not in cf_cache:
                            pre, _ = fit_initial(X, k, by_step[0]["last_max_iter"], record.seed)
                            if not np.array_equal(pre.labels, source):
                                raise ValueError(f"{run} seed {record.seed}: the step-0 fit does not replay the logged partition")
                            cf_cache[ckey] = counterfactual_labels(X, pre.centers, block["last_n_iter"])
                        ref_parts.append(
                            reference_rows(source, post, cf_cache[ckey], ds.labels, k, seed=record.seed).assign(**key, step=1)
                        )
            elif record.assignments is not None:
                j = record.n_summary_steps - 1
                k = len(record.summaries_evolution[j])
                inputs = [_select_cluster(c, record.assignments, X, ds.texts, fcl, record.seed, None) for c in range(k)]
                sent = [c for c in range(k) if inputs[c].doc_ids]
                n_in = {c: prompt_tokens(inputs[c].texts) for c in sent}
                n_out = {c: output_tokens(record.summaries_evolution[j][c]) for c in sent}
                cost_parts.append(cost_rows(k, sent, n_in, n_out).assign(**key, step=j + 1, cost_approx=True))
            metrics = record.metrics
            base = baselines.get(record.seed, {})
            run_rows.append(
                {
                    **key,
                    "method_raw": method,
                    **{f"final_{m}": metrics.get(m) for m in ("acc", "nmi", "ari")},
                    **{f"kmeans_{m}": base.get(m) for m in ("acc", "nmi", "ari")},
                    **{m: metrics.get(m) for m in ("runtime_s", "summarizer_seconds", "n_calls")},
                    "total_input_tokens": metrics.get("input_tokens"),
                    "total_output_tokens": metrics.get("output_tokens"),
                }
            )
    out = {"cost": pd.concat(cost_parts, ignore_index=True), "runs": pd.DataFrame(run_rows)}
    out["after"] = pd.concat(after_parts, ignore_index=True) if after_parts else pd.DataFrame(columns=[*RUN_KEYS, "majority_before"])
    out["ref"] = pd.concat(ref_parts, ignore_index=True) if ref_parts else pd.DataFrame(columns=RUN_KEYS)
    return out


# ------------------------------------------------------------------------------------------------ derived columns

FLU_COLUMNS = ("flu_empty", "flu_no_question", "flu_echo", "flu_copy", "flu_script", "flu_markup")
BOOL_COLUMNS = (
    "flag_low_coverage", "flag_duplicate", "flag_near_duplicate", "flag_mentions", "flag_low_margin", "flag_generic",
    "flag_unsupported", "flag_not_fluent", "label_top1", "own_nearest", "empty_replacement", "approx_partition",
    *FLU_COLUMNS,
)  # fmt: skip


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """The columns the analysis reads, computed from the joined ones (see the module docstring for the exact flags)."""
    out = df.copy()
    for col in BOOL_COLUMNS:
        if col in out:
            out[col] = out[col].map(lambda v: np.nan if pd.isna(v) else float(bool(v)))
    out["step1"] = out["step"] == 1
    n = out["n_members"]
    out["d_purity"] = out["purity_after"] - out["purity"]
    out["lost_share"] = out["n_lost"] / n
    out["gained_share"] = out["n_gained"] / n
    out["emptied"] = np.where(out["n_after"].isna(), np.nan, (out["n_after"] == 0).astype(float))
    out["d_member_purity"] = out["member_agree_after"] - out["purity"]
    out["helped"] = np.where(out["d_member_purity"].isna(), np.nan, (out["d_member_purity"] >= HELP_DELTA).astype(float))
    out["harmed"] = np.where(out["d_member_purity"].isna(), np.nan, (out["d_member_purity"] <= -HELP_DELTA).astype(float))
    if "cf_member_agree_after" in out:
        out["cf_d_member_purity"] = out["cf_member_agree_after"] - out["purity"]
        out["null_d_member_purity"] = out["null_member_agree_after"] - out["purity"]
        out["d_member_purity_vs_cf"] = out["d_member_purity"] - out["cf_d_member_purity"]
        out["d_member_purity_vs_null"] = out["d_member_purity"] - out["null_d_member_purity"]
        out["lost_share_vs_cf"] = out["lost_share"] - out["cf_lost_share"] if "cf_lost_share" in out else np.nan
        for name in ("helped", "harmed"):
            sign = 1 if name == "helped" else -1
            hit = sign * out["cf_d_member_purity"] >= HELP_DELTA
            out[f"cf_{name}"] = np.where(out["cf_d_member_purity"].isna(), np.nan, hit.astype(float))
    if "jaccard_major_after" in out:
        out["d_jaccard_major"] = out["jaccard_major_after"] - out["jaccard_majority"]
    out["log_n_members"] = np.log(n)
    out["n_flu_flags"] = out[[c for c in FLU_COLUMNS if c in out]].sum(axis=1)
    out["cost_tokens"] = out["input_tokens"] + out["output_tokens"]
    if "size_pre" in out:
        out["pre_matches_source"] = np.where(out["size_pre"].isna(), np.nan, (out["size_pre"] == n).astype(float))
        out["end_matches_logged"] = np.where(out["size_end"].isna(), np.nan, (out["size_end"] == out["n_after"]).astype(float))
    labels = out["method"]
    ambiguous = out.groupby(["dataset", "method"])["run_config"].transform("nunique") > 1
    out["method_id"] = np.where(ambiguous, labels + " [" + out["run_config"] + "]", labels)
    return out


# ------------------------------------------------------------------------------------------------- seed statistics


def seed_interval(values, level: float = 0.95) -> dict:
    """Mean of per-seed `values` and its two-sided Student-t interval (`n` values, `n - 1` degrees of freedom).

    NaNs are dropped. Fewer than two values give NaN bounds; no variation gives a zero-width interval.
    """
    v = np.asarray(pd.Series(values, dtype=float).dropna())
    n = len(v)
    if n == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n_seeds": 0}
    mean = float(v.mean())
    if n < 2:
        return {"mean": mean, "lo": np.nan, "hi": np.nan, "n_seeds": n}
    half = float(stats.t.ppf(0.5 + level / 2, n - 1) * v.std(ddof=1) / np.sqrt(n))
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "n_seeds": n}


def summarise_seeds(values, level: float = 0.95) -> dict:
    """`seed_interval`, or all-NaN (with the seed count kept) when fewer than `MIN_SEEDS` seeds contributed."""
    iv = seed_interval(values, level)
    if iv["n_seeds"] < MIN_SEEDS:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n_seeds": iv["n_seeds"]}
    return iv


def _rank_residuals(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(y)), controls])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def spearman(x, y, controls: pd.DataFrame | None = None) -> float:
    """Spearman correlation of `x` and `y`, partialled on `controls` (ranks regressed out of both ranks) if given.

    Rows with a NaN in any argument are dropped; NaN if fewer than 3 rows remain or either side is constant.
    """
    data = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)})
    if controls is not None and controls.shape[1]:
        data = pd.concat([data, controls.reset_index(drop=True).astype(float)], axis=1)
    data = data.dropna()
    if len(data) < 3 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return np.nan
    ranks = data.rank().to_numpy()
    rx, ry = ranks[:, 0], ranks[:, 1]
    if ranks.shape[1] > 2:
        rx, ry = _rank_residuals(rx, ranks[:, 2:]), _rank_residuals(ry, ranks[:, 2:])
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def correlation_table(
    df: pd.DataFrame,
    props: list[str],
    outcomes: list[str],
    controls: list[str] | None = None,
    group: str = "method_id",
) -> pd.DataFrame:
    """Per group, property and outcome: the per-seed Spearman correlation, summarised over seeds.

    `controls` are partialled out within each seed (`spearman`). One row per (group, property, outcome) with the mean
    over seeds, its interval, the number of seeds with a defined value and the number of clusters.
    """
    rows = []
    for gval, gdf in df.groupby(group, sort=False):
        for prop in props:
            for outcome in outcomes:
                per_seed = [
                    spearman(s[prop], s[outcome], s[controls] if controls else None)
                    for _, s in gdf.groupby("seed")
                ]
                n = int(gdf[[prop, outcome]].dropna().shape[0])
                iv = summarise_seeds(per_seed)
                rows.append(
                    {group: gval, "property": prop, "outcome": outcome, "controls": ",".join(controls or []) or "none",
                     "n_clusters": n, "rho": iv["mean"], "lo": iv["lo"], "hi": iv["hi"], "n_seeds": iv["n_seeds"]}
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------- strata


@dataclass(frozen=True)
class BinSpec:
    column: str
    edges: tuple[float, ...]
    labels: tuple[str, ...]
    right: bool = True


BIN_SPECS = {
    "coverage_nearest": BinSpec("coverage_nearest", (-np.inf, 0, 0.25, 0.5, 0.75, np.inf), ("0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1]")),
    "specificity_margin": BinSpec("specificity_margin", (-np.inf, 0, 0.1, 0.2, np.inf), ("<=0", "(0,.1]", "(.1,.2]", ">.2")),
    "unsupported_lemma_share": BinSpec(
        "unsupported_lemma_share", (-np.inf, 1e-9, 0.25, 0.5, np.inf), ("0", "(0,.25)", "[.25,.5)", ">=.5"), right=False
    ),
    "purity": BinSpec("purity", (-np.inf, 0.4, 0.6, 0.8, 0.9, np.inf), ("<=.4", "(.4,.6]", "(.6,.8]", "(.8,.9]", ">.9")),
    "size": BinSpec("n_members", (-np.inf, 150, np.inf), ("<=150", ">150")),
    "drift": BinSpec("drift", (-np.inf, 0.5, 1.0, 1.5, np.inf), ("<=.5", "(.5,1]", "(1,1.5]", ">1.5")),
}  # fmt: skip
FLAG_STRATA = ("flag_generic", "flag_duplicate", "flag_mentions", "flag_unsupported", "flag_low_coverage", "flag_not_fluent",
               "own_nearest", "label_top1", "empty_replacement")  # fmt: skip


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    """`bin_<property>` string columns for the numeric specs and the 0/1 flags (`no` / `yes`)."""
    out = df.copy()
    for name, spec in BIN_SPECS.items():
        if spec.column in out:
            out[f"bin_{name}"] = pd.cut(out[spec.column], list(spec.edges), labels=list(spec.labels), right=spec.right).astype(object)
    for flag in FLAG_STRATA:
        if flag in out:
            out[f"bin_{flag}"] = out[flag].map({0.0: "no", 1.0: "yes"})
    if "bin_purity" in out and "bin_size" in out:
        both = out["bin_purity"].notna() & out["bin_size"].notna()
        out["bin_purity_x_size"] = (out["bin_purity"] + " / " + out["bin_size"]).where(both)
    return out


def stratified_table(df: pd.DataFrame, strata: str, outcomes: list[str], group: str = "method_id") -> pd.DataFrame:
    """Per group and stratum of the column `strata`: cluster counts and the outcome means, seeds as the unit.

    For each outcome the per-seed value is the mean over the seed's clusters in the stratum (seeds with fewer than
    `MIN_PER_SEED` clusters there are left out), summarised with `seed_interval`. Columns: `n_clusters`, `n_seeds`
    (seeds that contributed), then `<outcome>`, `<outcome>_lo`, `<outcome>_hi`. Rows follow the order of first
    appearance of the strata in `df` (so bin labels stay in bin order if the frame is sorted that way).
    """
    rows = []
    for gval, gdf in df.dropna(subset=[strata]).groupby(group, sort=False):
        for stratum, sdf in gdf.groupby(strata, sort=False):
            row = {group: gval, "stratum": stratum, "n_clusters": len(sdf)}
            n_seeds = 0
            for outcome in outcomes:
                per_seed = [
                    s[outcome].mean() for _, s in sdf.groupby("seed") if s[outcome].notna().sum() >= MIN_PER_SEED
                ]
                iv = summarise_seeds(per_seed)
                n_seeds = max(n_seeds, iv["n_seeds"])
                row.update({outcome: iv["mean"], f"{outcome}_lo": iv["lo"], f"{outcome}_hi": iv["hi"]})
            row["n_seeds"] = n_seeds
            rows.append(row)
    return pd.DataFrame(rows)


def stratified_difference(
    df: pd.DataFrame, flag: str, outcome: str, strata: list[str] | None = None, seed: str = "seed"
) -> dict:
    """Mean outcome of flagged minus unflagged clusters, within strata, seeds as the unit.

    Per seed, for every stratum (a combination of the `strata` columns; none means one stratum) with at least
    `MIN_PER_SEED` clusters in each of the flag groups, the difference of the group means; the seed's value is the
    average of those differences weighted by the stratum's cluster count. `seed_interval` then summarises the seeds.
    Also returns `n_strata_used`, the mean number of strata a seed contributed.
    """
    values, used = [], []
    for _, sdf in df.groupby(seed):
        sdf = sdf.dropna(subset=[flag, outcome, *(strata or [])])
        cells = sdf.groupby(strata) if strata else [(None, sdf)]
        diffs, weights = [], []
        for _, cell in cells:
            on, off = cell.loc[cell[flag] == 1.0, outcome], cell.loc[cell[flag] == 0.0, outcome]
            if len(on) >= MIN_PER_SEED and len(off) >= MIN_PER_SEED:
                diffs.append(on.mean() - off.mean())
                weights.append(len(cell))
        if diffs:
            values.append(float(np.average(diffs, weights=weights)))
            used.append(len(diffs))
    return {**summarise_seeds(values), "n_strata_used": float(np.mean(used)) if used else 0.0}


# ----------------------------------------------------------------------------------------------- method level

QUALITY_MEANS = ("coverage_nearest", "specificity_margin", "unsupported_lemma_share", "flag_generic", "flag_not_fluent", "flag_low_coverage")


def method_table(df: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Per method: summary quality, final ACC / NMI / ARI, the gain over k-means and the cost, seeds as the unit.

    `df` is the joined frame (any steps; quality means pool the clusters of each seed first), `runs` the
    `run_extras` run-level frame. ACC / NMI / ARI and costs are per-run values over seeds; `d_<metric>` is the
    paired difference to the k-means run of the same seed with its interval.
    """
    keys = df[["run", "method_id"]].drop_duplicates()
    seed_quality = df.groupby(["run", "seed"])[list(QUALITY_MEANS)].mean().reset_index()
    per_seed = runs.merge(keys, on="run", validate="many_to_one").merge(seed_quality, on=["run", "seed"], validate="one_to_one")
    rows = []
    for mid, g in per_seed.groupby("method_id", sort=False):
        row = {"method_id": mid, "n_seeds": len(g)}
        for col in (*QUALITY_MEANS, "final_acc", "final_nmi", "final_ari", "runtime_s", "summarizer_seconds", "n_calls", "total_input_tokens", "total_output_tokens"):
            vals = pd.to_numeric(g[col], errors="coerce")
            iv = seed_interval(vals)
            row[col] = iv["mean"]
            row[f"{col}_sd"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
        for m in ("acc", "nmi", "ari"):
            iv = seed_interval(pd.to_numeric(g[f"final_{m}"], errors="coerce") - pd.to_numeric(g[f"kmeans_{m}"], errors="coerce"))
            row.update({f"d_{m}": iv["mean"], f"d_{m}_lo": iv["lo"], f"d_{m}_hi": iv["hi"]})
        row["kmeans_acc"] = pd.to_numeric(g["kmeans_acc"], errors="coerce").mean()
        row["kmeans_nmi"] = pd.to_numeric(g["kmeans_nmi"], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------ markdown


def format_cell(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    if pd.isna(mean):
        return "n/a"
    if pd.isna(lo):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    align = ["---"] + ["---:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------- the analysis

QUALITY_PROPS = ("coverage_nearest", "coverage_intents", "specificity_margin", "max_summary_cos", "unsupported_lemma_share")
GENERIC_PROPS = ("flag_generic", "flag_duplicate", "flag_mentions", "flag_unsupported", "flag_low_coverage")
REPR_PROPS = ("drift", "cosine", "own_nearest", "empty_replacement")
OUTCOMES = ("d_member_purity", "d_purity", "lost_share")
STRATA_OUTCOMES = ("d_member_purity", "d_purity", "lost_share", "helped", "harmed")
HARM_STRATA = (
    "coverage_nearest", "specificity_margin", "unsupported_lemma_share", "flag_generic", "flag_duplicate", "flag_mentions",
    "drift", "own_nearest",
)  # fmt: skip
HELP_STRATA = ("all", "purity", "label_top1", "coverage_nearest")
REF_STRATA = ("all", "purity", "purity_x_size")
REF_OUTCOMES = (
    "d_member_purity", "cf_d_member_purity", "null_d_member_purity", "d_member_purity_vs_cf", "d_member_purity_vs_null",
    "helped", "cf_helped", "harmed", "cf_harmed", "lost_share", "cf_lost_share",
)  # fmt: skip
FLUENCY_CONTROLS = ["coverage_nearest", "specificity_margin", "unsupported_lemma_share", "purity", "log_n_members"]
SIZE_CONTROLS = ["purity", "log_n_members"]


def scopes(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The row sets analysed: the exact step 1, and every step with #33 columns (steps 2 on rest on replayed blocks)."""
    have = df[df["has_diag"]]
    return {"step 1": have[have["step1"]], "all steps": have}


def _in_bin_order(table: pd.DataFrame, order: list[str], group: str = "method_id") -> pd.DataFrame:
    """Sort a stratified table by method (first appearance), then by the stratum's position in `order`."""
    methods = {m: i for i, m in enumerate(dict.fromkeys(table[group]))}
    rank = table["stratum"].map({lab: i for i, lab in enumerate(order)})
    return table.assign(_m=table[group].map(methods), _s=rank).sort_values(["_m", "_s"]).drop(columns=["_m", "_s"])


def _stratified(df: pd.DataFrame, props: tuple[str, ...], outcomes: tuple[str, ...]) -> pd.DataFrame:
    df = add_bins(df).assign(bin_all="all")
    parts = []
    for prop in props:
        if f"bin_{prop}" in df and df[f"bin_{prop}"].notna().any():
            if prop == "purity_x_size":
                order = [f"{p} / {z}" for p in BIN_SPECS["purity"].labels for z in BIN_SPECS["size"].labels]
            else:
                order = list(BIN_SPECS[prop].labels) if prop in BIN_SPECS else ["no", "yes", "all"]
            parts.append(_in_bin_order(stratified_table(df, f"bin_{prop}", list(outcomes)), order).assign(strata=prop))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fluency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Does failing the fluency proxies predict the outcome once coverage and genericity are accounted for?

    Per method and outcome: the difference between clusters that fail a proxy and those that do not, raw, within
    strata of nearest-summary share, and within strata of nearest-summary share x generic x unsupported. Methods with
    too few clusters in either fluency group in every seed give NaN.
    """
    df = add_bins(df)
    adjustments = {
        "raw": None,
        "within coverage": ["bin_coverage_nearest"],
        "within coverage x generic x unsupported": ["bin_coverage_nearest", "bin_flag_generic", "bin_flag_unsupported"],
    }
    rows = []
    for mid, g in df.groupby("method_id", sort=False):
        for outcome in ("d_member_purity", "lost_share"):
            for name, strata in adjustments.items():
                rows.append({"method_id": mid, "outcome": outcome, "adjustment": name,
                             "n_not_fluent": int((g["flag_not_fluent"] == 1.0).sum()), "n_clusters": len(g),
                             **stratified_difference(g, "flag_not_fluent", outcome, strata)})
    return pd.DataFrame(rows)


def analyse_with_outcomes(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All tables for the rows that have #33 columns and logged outcomes (CLINC150): one frame per question."""
    out: dict[str, list[pd.DataFrame]] = {k: [] for k in ("harm_strata", "help_strata", "help_reference", "correlations", "size_corr", "fluency", "fluency_partial")}
    for scope, rows in scopes(df).items():
        if rows.empty:
            continue
        out["harm_strata"].append(_stratified(rows, HARM_STRATA, STRATA_OUTCOMES).assign(scope=scope))
        out["help_strata"].append(_stratified(rows, HELP_STRATA, STRATA_OUTCOMES).assign(scope=scope))
        props = [*QUALITY_PROPS, *GENERIC_PROPS, "n_flu_flags", "flag_not_fluent", *REPR_PROPS]
        for controls in (None, SIZE_CONTROLS):
            out["correlations"].append(correlation_table(rows, props, list(OUTCOMES), controls).assign(scope=scope))
        out["fluency"].append(fluency_table(rows).assign(scope=scope))
        for controls in (None, ["purity"]):
            out["size_corr"].append(
                correlation_table(rows, ["n_members"], ["d_member_purity", "lost_share"], controls).assign(scope=scope)
            )
        if scope == "step 1" and "cf_member_agree_after" in rows:
            out["help_reference"].append(_stratified(rows, REF_STRATA, REF_OUTCOMES).assign(scope=scope))
        for controls in (None, FLUENCY_CONTROLS):
            out["fluency_partial"].append(
                correlation_table(rows, ["n_flu_flags"], ["d_member_purity", "lost_share"], controls).assign(scope=scope)
            )
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items() if v}


def cross_section(df: pd.DataFrame, outcome: str = "purity") -> pd.DataFrame:
    """Rank correlation of each summary property with the cluster's purity, per method (no step needed).

    For runs without per-step assignments (BANKING77) this is all there is: the last summaries against the FINAL
    partition, so it is a cross-sectional association with an approximate partition, not an effect of the step.
    """
    props = [*QUALITY_PROPS, *GENERIC_PROPS, "n_flu_flags", "flag_not_fluent"]
    parts = [correlation_table(df, props, [outcome], controls) for controls in (None, ["log_n_members"])]
    return pd.concat(parts, ignore_index=True)
