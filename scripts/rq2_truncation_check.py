"""How much did dataset-order truncation touch the logged #15 (RQ2) runs? CPU-only, from the run logs (#28, #37 step 1).

The #15 runs ran with a summarizer that kept the longest leading prefix of a cluster's members (ascending dataset
index) when the prompt exceeded `max_input_tokens`; no run logged which prompts were cut. This script rebuilds the
prompts from what was logged and counts the cuts. No model is run: it needs the Qwen3 tokenizer (for token counts),
the cached CLINC150 embeddings and the logs.

- Summary step 1 is exact. Every schedule starts from the same k-means++ fit (`random_state` = seed) and summarizes all
  150 clusters after its first Lloyd block, so that partition is replayed with scikit-learn (checked against the
  logged step-0 inertia).
- Later steps are approximate. Their partitions were not logged before #30, so the run's final partition stands in for
  them, and the clusters summarized at a later step are the ones whose summary text changed. `--check` runs the same
  rebuild on the #31 Qwen3-0.6B run, where the true partitions and the logged truncation counts exist (member
  selection there is the #28 seeded random subset), to show how close the approximation is.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python scripts/rq2_truncation_check.py --results-dir /path/to/results \\
        --cache-dir /path/to/cache --out-dir docs/results
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

from kslimeans.runlog import RunRecord, load_run, load_run_dataset, members_by_cluster
from kslimeans.summarizers.hf import (
    DEFAULT_SYSTEM_PROMPT,
    TRUNCATION_SEED,
    build_messages,
    fit_texts_to_budget,
    render_conversation,
    supports_system_role,
)

MAX_INPUT_TOKENS = 4096
EVENT_COLUMNS = ["cluster", "size", "kept", "dropped", "n_intents", "lost_intents", "step", "exact", "seed", "method"]
BIG_CLUSTER = 200  # members; the smallest overflowing cluster found has 202
MODEL_ID = "Qwen/Qwen3-0.6B"
N_CLUSTERS = 150
PREFILTER = 0.85  # clusters whose additive token estimate is below this share of the budget cannot overflow


class PromptBudget:
    """Token counts of the prompts `HFSummarizer` builds for CLINC150 (same render path, tokenizer only)."""

    def __init__(self, texts: list[str], prompt: str, text_type: str, max_tokens: int = MAX_INPUT_TOKENS) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
        self.kwargs = {"enable_thinking": False}
        self.fold = not supports_system_role(self.tokenizer, self.kwargs)
        self.prompt, self.text_type, self.max_tokens = prompt, text_type, max_tokens
        self.texts = texts
        self.text_tokens = np.array([len(ids) for ids in self.tokenizer(texts, add_special_tokens=False)["input_ids"]])
        self.overhead = self.count([])

    def render(self, texts: list[str]) -> str:
        messages = build_messages(DEFAULT_SYSTEM_PROMPT, self.prompt, self.text_type, texts, self.fold)
        return render_conversation(self.tokenizer, messages, self.text_type, self.kwargs)

    def count_rendered(self, rendered: str) -> int:
        return len(self.tokenizer(rendered, add_special_tokens=False)["input_ids"])

    def count(self, texts: list[str]) -> int:
        return self.count_rendered(self.render(texts))

    def may_overflow(self, doc_ids: np.ndarray) -> bool:
        """Cheap screen to skip exact counting: each member costs about its own tokens plus a newline."""
        return self.overhead + int(self.text_tokens[doc_ids].sum()) + len(doc_ids) > PREFILTER * self.max_tokens

    def kept_prefix(self, doc_ids: np.ndarray) -> int:
        """How many leading members (ascending dataset index) the #15 summarizer kept: the old rule."""
        texts = [self.texts[i] for i in doc_ids]
        n = len(texts)
        if n <= 1 or self.count(texts) <= self.max_tokens:
            return n
        lo, hi, kept = 1, n - 1, 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.count(texts[:mid]) <= self.max_tokens:
                kept, lo = mid, mid + 1
            else:
                hi = mid - 1
        return kept

    def kept_random(self, doc_ids: np.ndarray) -> int:
        """How many members the current (#28) rule keeps: a seeded random subset."""
        texts = [self.texts[i] for i in doc_ids]
        fitted, _, _ = fit_texts_to_budget(texts, self.render, self.count_rendered, self.max_tokens)
        return len(fitted)


def _lost_intents(labels: np.ndarray, doc_ids: np.ndarray, kept: int, random: bool) -> tuple[int, int]:
    """(distinct intents in the cluster, intents with no kept member). The random rule's kept set is the
    first `kept` of the TRUNCATION_SEED permutation, as in `fit_texts_to_budget`."""
    members = labels[doc_ids]
    if random:
        sel = np.sort(np.random.default_rng(TRUNCATION_SEED).permutation(len(doc_ids))[:kept])
        survivors = members[sel]
    else:
        survivors = members[:kept]
    present = np.unique(members)
    return len(present), len(present) - len(np.unique(survivors))


def cluster_events(
    budget: PromptBudget, labels_gold: np.ndarray, partition: np.ndarray, clusters: list[int], rule: str
) -> list[dict]:
    """One row per summarized cluster whose prompt overflows under `rule` ("prefix" = #15, "random" = #28)."""
    members = members_by_cluster(partition, N_CLUSTERS)
    rows = []
    for c in clusters:
        doc_ids = members[c]
        if len(doc_ids) == 0 or not budget.may_overflow(doc_ids):
            continue
        kept = budget.kept_prefix(doc_ids) if rule == "prefix" else budget.kept_random(doc_ids)
        if kept >= len(doc_ids):
            continue
        n_intents, lost = _lost_intents(labels_gold, doc_ids, kept, rule == "random")
        rows.append({"cluster": c, "size": len(doc_ids), "kept": kept, "dropped": len(doc_ids) - kept,
                     "n_intents": n_intents, "lost_intents": lost, "doc_ids": tuple(int(i) for i in doc_ids)})
    return rows


def changed_clusters(evolution: list[list], j: int) -> list[int]:
    """Clusters whose summary text differs from the previous step's (step 0 of the evolution: all of them)."""
    now = evolution[j]
    if j == 0:
        return [c for c, s in enumerate(now) if s is not None]
    prev = evolution[j - 1]
    return [c for c, s in enumerate(now) if s is not None and s != prev[c]]


def first_partition(X: np.ndarray, seed: int, first_iters: int) -> tuple[np.ndarray, float]:
    km = KMeans(n_clusters=N_CLUSTERS, init="k-means++", max_iter=first_iters, random_state=seed)
    labels = km.fit_predict(X)
    return labels, float(km.inertia_)


def first_iters(cfg: dict) -> int:
    s = cfg["schedule"]
    d = s["n_summaries"] + 1
    return s["max_iter"] // d if s.get("initial_iter", True) else 1


def run_events(
    budget: PromptBudget, gold: np.ndarray, X: np.ndarray, cfg: dict, record: RunRecord, rule: str = "prefix"
) -> tuple[pd.DataFrame, dict]:
    """Overflowing prompts of one logged (method, seed), one row per (summary step, cluster), plus totals.

    Step 1 uses the replayed partition (`exact` True), later steps the final partition (`exact` False). The clusters
    summarized at a step are those whose summary text changed since the previous step; a re-summary that happened to
    return the same text, and a memo-cache hit, are therefore not counted. `members` totals the sizes of the
    summarized clusters (what the summarizer is offered before trimming); `n_big_step1` counts the step-1 clusters with
    at least `BIG_CLUSTER` members.
    """
    evolution = record.summaries_evolution
    p1, inertia = first_partition(X, record.seed, first_iters(cfg))
    logged = record.trace[0]["inertia"]
    rows = []
    totals = {"n_summarized": 0, "members": 0, "inertia_rel_err": abs(inertia - logged) / logged,
              "n_summarized_trace": sum(r["n_summarized"] for r in record.trace)}
    for j in range(len(evolution)):
        partition = p1 if j == 0 else record.assignments
        summarized = changed_clusters(evolution, j)
        sizes = np.bincount(partition, minlength=N_CLUSTERS)
        totals["n_summarized"] += len(summarized)
        totals["members"] += int(sizes[summarized].sum())
        if j == 0:
            totals["n_big_step1"] = int((sizes >= BIG_CLUSTER).sum())
        for ev in cluster_events(budget, gold, partition, summarized, rule):
            ev.pop("doc_ids")
            ev.update(step=j + 1, exact=(j == 0), seed=record.seed, method=record.method)
            rows.append(ev)
    return pd.DataFrame(rows, columns=EVENT_COLUMNS), totals


def rebuild_31(results: Path, cache_dir: str) -> pd.DataFrame:
    """Check of the rebuild on the #31 Qwen3-0.6B run, one row per seed. `*_logged` is what the run logged;
    `*_exact` rebuilds it from the true partitions and the memo cache under the #28 random-subset rule; `*_approx` is
    what the #15 procedure (replayed step 1, final partition later, changed summaries) gives for the same run."""
    run_dir = Path(sorted(glob.glob(str(results / "clinc150_qwen3-0.6b" / "*" / "summary.csv")))[-1]).parent
    runlog = load_run(run_dir)
    ds, X = load_run_dataset(runlog, cache_dir=cache_dir)
    gold = np.asarray(ds.labels)
    budget = PromptBudget(ds.texts, ds.prompt, ds.text_type)
    rows = []
    for rec in runlog.method_records("k-SLMmeans"):
        seen: set[tuple[int, ...]] = set()
        calls = trunc = dropped = 0
        for j in range(rec.n_summary_steps):
            partition = rec.source_partition(j)
            members = members_by_cluster(partition, N_CLUSTERS)
            todo = [c for c in range(N_CLUSTERS) if len(members[c]) and tuple(int(i) for i in members[c]) not in seen]
            events = cluster_events(budget, gold, partition, todo, "random")
            calls += len(todo)
            trunc += len(events)
            dropped += sum(e["dropped"] for e in events)
            seen.update(tuple(int(i) for i in m) for m in members if len(m))
        approx, totals = run_events(budget, gold, X, runlog.config, rec, rule="random")
        m = rec.metrics
        rows.append({"seed": rec.seed, "calls_logged": m["n_calls"], "calls_exact": calls,
                     "truncated_logged": m["n_truncated_clusters"], "truncated_exact": trunc,
                     "truncated_approx": len(approx), "dropped_logged": m["n_dropped_texts"], "dropped_exact": dropped,
                     "dropped_approx": int(approx["dropped"].sum()),
                     "step1_inertia_rel_err": totals["inertia_rel_err"]})
    return pd.DataFrame(rows)


def natural_experiment(results: Path, totals: pd.DataFrame) -> pd.DataFrame:
    """fixed-n5 on seeds 0-4 in #15 (prefix rule) against the #31 0.6B run (random-subset rule): same model, prompt,
    schedule, embeddings and seeds, so the two differ in the truncation rule, the code version and GPU noise. Per seed:
    ACC of both, and the number of summaries per step that are textually identical in the two runs."""
    def load(pattern: str, method: str) -> dict[int, dict]:
        out = {}
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    if row["method"] == method and row.get("summaries_evolution"):
                        out[row["seed"]] = row
        return out

    old = load(str(results / "rq2_fixed_n5" / "*" / "summaries.jsonl"), "fixed-n5")
    new_dir = Path(sorted(glob.glob(str(results / "clinc150_qwen3-0.6b" / "*" / "summary.csv")))[-1]).parent
    new = load(str(new_dir / "summaries.jsonl"), "k-SLMmeans")
    acc_new = {r.seed: r.metrics["acc"] for r in load_run(new_dir).method_records("k-SLMmeans")}
    mine = totals[(totals["run"] == "rq2_fixed_n5") & (totals["method"] == "fixed-n5")].set_index("seed")
    rows = []
    for seed in sorted(old.keys() & new.keys()):
        same = [sum(x == y for x, y in zip(a, b)) for a, b in zip(old[seed]["summaries_evolution"], new[seed]["summaries_evolution"])]
        rows.append({"seed": seed, "acc_15": mine.loc[seed, "acc"], "acc_31": acc_new[seed],
                     "acc_diff": acc_new[seed] - mine.loc[seed, "acc"], "truncated_15": mine.loc[seed, "n_trunc"],
                     "dropped_15": mine.loc[seed, "dropped"], **{f"same_step{j + 1}": n for j, n in enumerate(same)}})
    return pd.DataFrame(rows)


def collect(results: Path, cache_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overflowing prompts (one row each) and per-(run, method, seed) totals of the 60 `rq2_*` run directories."""
    dirs = sorted(d for d in results.glob("rq2_*") if d.is_dir() and not d.name.startswith("rq2r_"))
    runs = [(d, load_run(sorted(d.glob("*/summary.csv"))[-1].parent)) for d in dirs if list(d.glob("*/summary.csv"))]
    ds, X = load_run_dataset(runs[0][1], cache_dir=cache_dir)
    gold = np.asarray(ds.labels)
    budget = PromptBudget(ds.texts, ds.prompt, ds.text_type)
    events, totals = [], []
    for d, runlog in runs:
        for rec in runlog.records.values():
            if not rec.summaries_evolution:
                continue
            ev, tot = run_events(budget, gold, X, runlog.config, rec)
            ev["run"] = d.name
            tot.update(run=d.name, method=rec.method, seed=rec.seed, n_calls=rec.metrics["n_calls"],
                       acc=rec.metrics["acc"], n_trunc=len(ev), dropped=int(ev["dropped"].sum()),
                       n_trunc_step1=int((ev["step"] == 1).sum()),
                       dropped_step1=int(ev.loc[ev["step"] == 1, "dropped"].sum()))
            events.append(ev)
            totals.append(tot)
        print(f"{d.name}: {len(totals)} records", flush=True)
    return pd.concat(events, ignore_index=True), pd.DataFrame(totals)


HEADLINE = ["fixed-n1", "fixed-n2", "fixed-n5", "adaptive-d0.6-r0.7", "random-d0.6-r0.7"]


def schedule_table(totals: pd.DataFrame) -> pd.DataFrame:
    """Per method, means over its (run, seed) records: calls, overflowing prompts and dropped members."""
    g = totals.groupby("method").agg(
        records=("seed", "size"), calls=("n_calls", "mean"), summarized=("n_summarized", "mean"),
        truncated=("n_trunc", "mean"), truncated_min=("n_trunc", "min"), truncated_max=("n_trunc", "max"),
        step1=("n_trunc_step1", "mean"), dropped=("dropped", "mean"), members=("members", "mean"))
    g["truncated_per_100_summaries"] = 100 * g["truncated"] / g["summarized"]
    g["dropped_share"] = g["dropped"] / g["members"]
    return g


def gap_correlations(totals: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation across seeds between the paired ACC gap (adaptive minus a reference) and the paired
    difference in truncation exposure (overflowing prompts, dropped members) of the same two runs. Observational:
    7-10 seeds, and step-1 exposure is identical for the adaptive run, fixed-n5 and the control by construction."""
    adaptive = "adaptive-d0.6-r0.7"
    by = {m: totals[totals["method"] == m].set_index("seed") for m in HEADLINE}
    rows = []
    for label, seeds in (("0-9", range(10)), ("3-9", range(3, 10))):
        seeds = list(seeds)
        for ref in ("fixed-n5", "random-d0.6-r0.7", "fixed-n1", "fixed-n2"):
            gap = (by[adaptive]["acc"] - by[ref]["acc"]).loc[seeds]
            for col in ("n_trunc", "dropped"):
                diff = (by[adaptive][col] - by[ref][col]).loc[seeds]
                rho, pval = spearmanr(diff, gap) if diff.nunique() > 1 else (float("nan"), float("nan"))
                rows.append({"seeds": label, "reference": ref, "exposure": col, "n": len(seeds), "rho": rho, "p": pval})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-dir", default="docs/results")
    parser.add_argument("--check", action="store_true", help="only rebuild the #31 Qwen3-0.6B counts and print them")
    args = parser.parse_args()
    results, out = Path(args.results_dir), Path(args.out_dir)
    check = rebuild_31(results, args.cache_dir)
    print(check.to_string(index=False))
    if args.check:
        return
    events, totals = collect(results, args.cache_dir)
    check.to_csv(out / "rq2_truncation_check31.csv", index=False, float_format="%.6g")
    totals.to_csv(out / "rq2_truncation_per_run.csv", index=False, float_format="%.6g")
    events.to_csv(out / "rq2_truncation_prompts.csv", index=False)
    print(schedule_table(totals).round(3).to_string())
    print(gap_correlations(totals).round(2).to_string(index=False))
    print(natural_experiment(results, totals).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
