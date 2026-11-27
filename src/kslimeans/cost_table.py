"""Cost table and cost per unit of benefit (#82), from the logged runs and the #10 price list.

`scripts/cost_table.py` is the CLI. For each dataset (`DATASETS`) the run directories are read with
`aggregate.cost_table`, `aggregate.per_seed_table` and `aggregate.paired_differences`, and `build_rows`
turns them into one row per method (and prompt): seeds, summarizer calls, tokens and seconds, the timing
source with its caveat, ACC / NMI / ARI, and the paired difference against LSA and against k-means with
the 95% Student-t interval over seeds. `cost_per_benefit` applies the interval gate: extra seconds
(and dollars) per ACC point are reported only when the paired ACC interval excludes 0 on the positive side.

Seconds are the logged `summarizer_seconds` when the run has them, otherwise the logged `runtime_s` of the
whole method (BANKING77 and the arXiv runs predate the cost fields, #30). Nothing is re-timed here.

Prices (`load_prices`) come from a CSV of `item,unit,usd,source,checked` rows. Without it every dollar
column is empty (pending #10).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .aggregate import (
    METRICS,
    cost_table,
    method_label,
    paired_differences,
    per_seed_table,
)

LSA = "k-NLPmeans (LSA)"
KMEANS = "k-means"
REFERENCES = (LSA, KMEANS)

# A foreign-memory rise above the pre-run baseline beyond our own CUDA context (136 to 152 MiB in the
# idle fixed-workload runs, `cost_timing.md` section 1) marks a run as not clean.
CLEAN_MIB = 300.0

# Measured inflation of the #31 timings, seed 0 step 1 only, logged over idle-card fixed workload
# (`cost_timing.md` section 4). The whole-run inflation was not measured.
STEP1_INFLATION = {"0.6B": 1.76, "1.7B": 1.72, "4B": 1.15}
# Idle-card fixed workload (150 step-1 prompts of seed 0, #31 prompt) at each size's #31 batch size:
# (batch size, median seconds per call), `cost_timing.md` section 2.
IDLE_S_PER_CALL = {"0.6B": (8, 0.63), "1.7B": (4, 1.11), "4B": (1, 2.10)}

PRICE_COLUMNS = ("item", "unit", "usd", "source", "checked")
# CONFIRM(#10): item and unit names below are this script's convention; match them to
# docs/cost_assumptions.md once Nabin's price list lands (due Nov 26).
GPU_ITEM = "gpu"
GPU_UNIT = "hour"
API_UNIT = "1M tokens"
API_PREFIXES = {"api_input:": "input", "api_output:": "output"}


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset block of the table: its run directories under the results root and its caveats."""

    key: str
    title: str
    run_dirs: tuple[str, ...]
    default_prompt: str
    descriptive_only: bool = False
    row_notes: dict = field(default_factory=dict)  # method label -> (timing code, extra timing caveat)


_CLINC_OLD = (
    "clinc150_lsa/20261027-174631",
    "clinc150_qwen3-0.6b/20261027-190908",
    "clinc150_qwen3-1.7b/20261027-202532",
    "clinc150_qwen3-4b/20261027-221042",
)
_CLINC_V2 = (
    "clinc150_v2_qwen3-0.6b/20261112-200926",
    "clinc150_v2_qwen3-1.7b/20261112-204127",
    "clinc150_v2_qwen3-4b/20261112-181256",
)

DATASETS = (
    DatasetSpec("clinc150", "CLINC150 (intent)", _CLINC_OLD + _CLINC_V2, "October (#31)"),
    DatasetSpec(
        "bank77",
        "BANKING77 (#20)",
        (
            "prelim_bank77_lsa/20261014-105249",
            "prelim_bank77_qwen3-0.6b/20261014-105306",
            "prelim_bank77_qwen3-1.7b/20261014-110159",
            "prelim_bank77_qwen3-4b/20261014-112130",
        ),
        "upstream BANKING77",
    ),
    DatasetSpec(
        "bbc_tfidf",
        "BBC, TF-IDF (#18)",
        (
            "bbc_tfidf_lsa/20261112-194517",
            "bbc_tfidf_qwen3-0.6b/20261112-194544",
            "bbc_tfidf_qwen3-1.7b/20261112-194816",
            "bbc_tfidf_qwen3-4b/20261112-195223",
        ),
        "BBC prompt",
    ),
    DatasetSpec(
        "bbc_dense",
        "BBC, dense (#18)",
        (
            "bbc_dense_lsa/20261112-200101",
            "bbc_dense_qwen3-0.6b/20261112-200116",
            "bbc_dense_qwen3-1.7b/20261112-200302",
            "bbc_dense_qwen3-4b/20261112-200530",
        ),
        "BBC prompt",
    ),
    DatasetSpec(
        "arxiv",
        "arXiv 200k pilot (#48, seed 0)",
        ("arxiv_sub200k_pilot/20261116-175906", "arxiv_sub200k_lsa/20261116-181519"),
        "arXiv prompt",
        descriptive_only=True,
        row_notes={
            LSA: ("-", "excludes 3,319 s of one-off spaCy sentence segmentation (not summarizer time, CPU shared)"),
            "k-SLMmeans (Qwen3-1.7B)": (
                "E",
                "whole-method runtime incl. 8 s embedding and the k-means refits; ran next to other jobs (an upper "
                "bound on idle-card seconds, not clean); calls at most 3,360 scheduled, not logged",
            ),
        },
    ),
)


# ---------------------------------------------------------------------------------------------- prices


@dataclass(frozen=True)
class Prices:
    """GPU dollars per hour and API dollars per 1M input / output tokens, with where each came from."""

    gpu_usd_per_hour: float | None
    api: dict  # model -> (usd per 1M input tokens, usd per 1M output tokens)
    sources: tuple  # (item, usd, source, checked) per row

    def priced(self) -> bool:
        return self.gpu_usd_per_hour is not None or bool(self.api)


def _unit(raw) -> str:
    """A price unit compared loosely: case and spacing ignored, and an optional leading "per" ("per hour" is "hour")."""
    unit = re.sub(r"\s+", " ", str(raw)).strip().lower()
    return unit[4:] if unit.startswith("per ") else unit


def parse_prices(frame: pd.DataFrame) -> Prices:
    """Validate a prices frame (`item,unit,usd,source,checked`) and split it into GPU and API prices.

    `gpu` (unit `hour`) is the local GPU price; `api_input:<model>` and `api_output:<model>` (unit
    `1M tokens`) are one API model's list prices, and a model needs both. Raises `ValueError` for a
    missing column, an unknown item, a wrong unit, a non-numeric or negative price, a duplicate item, a
    model with only one of the two prices, or a row without a source and a check date.
    """
    missing = [c for c in PRICE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"prices file lacks columns {missing}, expected {list(PRICE_COLUMNS)}")
    items = frame["item"].astype(str).str.strip()
    if items.duplicated().any():
        raise ValueError(f"duplicate price items: {sorted(set(items[items.duplicated()]))}")
    gpu = None
    parts: dict[str, dict[str, float]] = {}
    sources = []
    for item, (_, row) in zip(items, frame.iterrows()):
        unit = _unit(row["unit"])
        try:
            usd = float(row["usd"])
        except (TypeError, ValueError):
            raise ValueError(f"price of '{item}' is not a number: {row['usd']!r}") from None
        if not np.isfinite(usd) or usd < 0:
            raise ValueError(f"price of '{item}' must be a non-negative number, got {usd}")
        for column in ("source", "checked"):
            if pd.isna(row[column]) or not str(row[column]).strip():
                raise ValueError(f"price of '{item}' has no {column}: every price needs its source and date")
        if item == GPU_ITEM:
            if unit != GPU_UNIT.lower():
                raise ValueError(f"'{item}' must be per {GPU_UNIT}, got unit '{row['unit']}'")
            gpu = usd
        else:
            prefix = next((p for p in API_PREFIXES if item.startswith(p)), None)
            model = item[len(prefix) :] if prefix else ""
            if not prefix or not model:
                raise ValueError(f"unknown price item '{item}': use '{GPU_ITEM}', 'api_input:<model>' or 'api_output:<model>'")
            if unit != API_UNIT.lower():
                raise ValueError(f"'{item}' must be per {API_UNIT}, got unit '{row['unit']}'")
            parts.setdefault(model, {})[API_PREFIXES[prefix]] = usd
        sources.append((item, usd, str(row["source"]).strip(), str(row["checked"]).strip()))
    for model, got in parts.items():
        if set(got) != {"input", "output"}:
            raise ValueError(f"API model '{model}' needs both api_input and api_output prices")
    api = {model: (got["input"], got["output"]) for model, got in parts.items()}
    return Prices(gpu, api, tuple(sources))


def load_prices(path: str | Path | None) -> Prices | None:
    """Read a prices CSV; `None` when no path is given (the dollar columns then stay empty)."""
    if path is None:
        return None
    return parse_prices(pd.read_csv(path, dtype=str, keep_default_na=False))


# ---------------------------------------------------------------------------------- cost per benefit


def cost_per_benefit(
    diff: float, lo: float, hi: float, extra_seconds: float, extra_usd: dict | None = None
) -> dict:
    """Extra cost per ACC point (100 x `diff`) over a reference, gated by the paired interval.

    `diff`, `lo`, `hi` are the paired mean difference (method minus reference) and its 95% interval over
    seeds, in ACC (0 to 1). `status` is `benefit` when `lo > 0`; `loss` when `hi < 0`; `no_benefit` when the
    interval contains 0; `n/a` when there is no interval (fewer than 2 seeds). Only `benefit` carries a
    cost per point: `seconds_per_point`, and `usd_per_point[name]` for each entry of `extra_usd` (name ->
    extra dollars). A benefit that costs no more than the reference (`extra_seconds <= 0`) has no ratio:
    `status` is `benefit_no_extra_cost`.
    """
    out = {"status": "n/a", "points": np.nan, "seconds_per_point": np.nan, "usd_per_point": {}}
    if any(pd.isna(v) for v in (diff, lo, hi)):
        return out
    out["points"] = 100.0 * diff
    if lo > 0:
        out["status"] = "benefit"
    elif hi < 0:
        out["status"] = "loss"
    else:
        out["status"] = "no_benefit"
    if out["status"] != "benefit":
        return out
    if extra_seconds <= 0:
        out["status"] = "benefit_no_extra_cost"
        return out
    out["seconds_per_point"] = extra_seconds / out["points"]
    for name, usd in (extra_usd or {}).items():
        if not pd.isna(usd):
            out["usd_per_point"][name] = usd / out["points"]
    return out


# ------------------------------------------------------------------------------------------ the rows


def describe(label: str) -> dict:
    """`family` (kmeans / lsa / slm / None for other baselines), `model` ('0.6B', ...) and `v2` from a method label."""
    if label == KMEANS:
        return {"family": "kmeans", "model": "", "v2": False}
    if label.startswith("k-NLPmeans"):
        return {"family": "lsa", "model": "", "v2": False}
    if label.startswith("k-SLMmeans"):
        found = re.search(r"Qwen3-([0-9.]+B)", label)
        return {"family": "slm", "model": found.group(1) if found else "", "v2": label.startswith("k-SLMmeans-v2")}
    return {"family": None, "model": "", "v2": False}


def _order_key(label: str) -> tuple:
    info = describe(label)
    rank = {"kmeans": 0, "lsa": 1, "slm": 2}[info["family"]]
    return (rank, info["v2"], float(info["model"].rstrip("B") or 0))


def _config_batch_sizes(run_dirs: list[Path]) -> dict:
    """Method label -> summarizer `batch_size` from `config.yaml` (runs without an `environment.json` have no other source)."""
    out = {}
    for run_dir in run_dirs:
        with open(run_dir / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        label = method_label(cfg)
        if label is not None:
            out[label] = (cfg.get("summarizer") or {}).get("batch_size")
    return out


TIMING_CODES = {
    "-": "CPU method, no card involved",
    "A": "card idle-checked at the start and clean during the run (within about our own CUDA context)",
    "B": "card not idle-checked (run predates #35); seconds inflated, step-1 inflation measured (CLINC150 October prompt)",
    "C": "card not idle-checked (run predates #35); seconds probably inflated, inflation not measured",
    "D": "card idle at the start but foreign memory seen during the run: not a clean timing",
    "E": "run overlapped other jobs: an upper bound on idle-card seconds",
}


def timing_source(info: dict, idle: bool | None, above_mib: float | None, dataset_key: str, batch: int | None) -> tuple[str, str]:
    """`(code, text)`: the timing source and its caveat for one row, from what the run logged about its card.

    Codes are in `TIMING_CODES`; the text names the batch size and whether the card was verified idle.
    """
    if info["family"] == "kmeans":
        return "-", "CPU method; total seconds as logged"
    if info["family"] == "lsa":
        return "-", "CPU summarizer; seconds as logged (the card is not used for the summaries)"
    size = info["model"]
    bs = f"batch {batch}" if batch else "batch size not logged"
    if idle is None:
        text = f"logged, {bs}; card not idle-checked (run predates #35), so the seconds are probably inflated"
        if dataset_key == "clinc150" and not info["v2"]:
            return "B", text + (
                f"; step 1 of seed 0 measured {STEP1_INFLATION[size]:.2f}x slower than on an idle card"
                " (whole-run inflation not measured)"
            )
        return "C", text + "; inflation not measured for this run (one BANKING77 check estimated about 1.7x)"
    if idle is False:
        return "E", f"logged, {bs}; card busy at the start of the run, seconds inflated"
    shown = "" if above_mib is None else f", {above_mib:+,.0f} MiB non-torch memory above the pre-run baseline"
    if above_mib is not None and above_mib > CLEAN_MIB:
        return "D", f"logged, {bs}; card idle at start but not clean during the run{shown}, not a clean timing"
    return "A", f"logged, {bs}; card idle at start{shown} (about our own CUDA context)"


def build_rows(
    spec: DatasetSpec,
    cost: pd.DataFrame,
    per_seed: pd.DataFrame,
    paired: pd.DataFrame,
    batch_sizes: dict | None = None,
    prices: Prices | None = None,
) -> pd.DataFrame:
    """One row per (method, prompt) of one dataset.

    `cost`, `per_seed` and `paired` are the outputs of `aggregate.cost_table`, `per_seed_table` and
    `paired_differences` (references LSA and k-means) for the dataset's runs. Methods that are not k-means,
    LSA or an SLM (the streaming baselines of the arXiv runs) are dropped. Dollars: `local_usd` is the row's
    seconds at the GPU price per hour (LSA is priced at the same rate although its summarizer runs on the
    CPU, so its dollars are an upper bound); `api_usd:<model>` is the hypothetical API price of the row's
    logged tokens, 0 for k-means and LSA (no API call), empty when the tokens were not logged. With no
    `prices`, the dollar columns are empty.
    """
    batch_sizes = batch_sizes or {}
    costs = cost.set_index("method")
    stats = per_seed.groupby("method")[list(METRICS)].agg(["mean", "std"])
    has_summarizer_seconds = any(
        describe(m)["family"] in ("lsa", "slm") and pd.notna(costs.loc[m, "summarizer_seconds_mean"]) for m in costs.index
    )
    labels = sorted((m for m in costs.index if describe(m)["family"]), key=_order_key)
    api_models = sorted(prices.api) if prices else []
    pair = paired.set_index(["reference", "method", "metric"])

    def sentinel(value):
        return np.nan if value is None or pd.isna(value) else float(value)

    rows = []
    for label in labels:
        info = describe(label)
        c = costs.loc[label]
        if info["family"] == "kmeans":
            seconds = 0.0 if has_summarizer_seconds else sentinel(c["runtime_s_mean"])
        else:
            logged = sentinel(c["summarizer_seconds_mean"])
            seconds = logged if has_summarizer_seconds else sentinel(c["runtime_s_mean"])
        batch = batch_sizes.get(label)
        if batch is None and pd.notna(c.get("batch_size")):
            batch = c["batch_size"]
        batch = None if batch is None or pd.isna(batch) else int(batch)
        row = {
            "dataset": spec.key,
            "method": label,
            "family": info["family"],
            "model": info["model"],
            "prompt": "" if info["family"] != "slm" else ("v2 (#46)" if info["v2"] else spec.default_prompt),
            "seeds": int(c["n_seeds"]),
            "batch_size": batch,
            "calls": sentinel(c["n_calls_mean"]),
            "input_tokens": sentinel(c["input_tokens_mean"]),
            "output_tokens": sentinel(c["output_tokens_mean"]),
            "seconds": seconds,
            "seconds_sd": sentinel(c["summarizer_seconds_std"] if has_summarizer_seconds else c["runtime_s_std"]),
            "seconds_basis": "summarizer" if has_summarizer_seconds else "run total",
            "total_seconds": sentinel(c["runtime_s_mean"]),
            "gpu_idle": c["gpu_idle"] if pd.notna(c["gpu_idle"]) else None,
            "non_torch_mib_above_baseline": sentinel(c["gpu_above_baseline_mib"]),
        }
        above = None if pd.isna(row["non_torch_mib_above_baseline"]) else row["non_torch_mib_above_baseline"]
        row["timing_code"], row["timing_source"] = timing_source(info, row["gpu_idle"], above, spec.key, batch)
        if label in spec.row_notes:
            code, note = spec.row_notes[label]
            if code != "-":
                row["timing_code"] = code
            row["timing_source"] += "; " + note
        idle = IDLE_S_PER_CALL.get(info["model"]) if spec.key == "clinc150" and info["family"] == "slm" and not info["v2"] else None
        row["idle_batch"], row["idle_s_per_call"] = idle if idle else (np.nan, np.nan)
        for metric in METRICS:
            row[f"{metric}_mean"] = float(stats.loc[label, (metric, "mean")])
            row[f"{metric}_sd"] = sentinel(stats.loc[label, (metric, "std")])
        for ref_name, ref in (("lsa", LSA), ("kmeans", KMEANS)):
            for metric in METRICS:
                key = (ref, label, metric)
                got = pair.loc[key] if key in pair.index else None
                for col, src in (("diff", "mean_diff"), ("lo", "ci_low"), ("hi", "ci_high")):
                    row[f"vs_{ref_name}_{metric}_{col}"] = np.nan if got is None else float(got[src])
        # dollars
        row["local_usd"] = np.nan
        for model in api_models:
            row[f"api_usd:{model}"] = np.nan
        if prices is not None:
            if prices.gpu_usd_per_hour is not None and pd.notna(seconds):
                row["local_usd"] = prices.gpu_usd_per_hour * seconds / 3600.0
            for model in api_models:
                price_in, price_out = prices.api[model]
                if info["family"] in ("kmeans", "lsa"):
                    row[f"api_usd:{model}"] = 0.0
                elif pd.notna(row["input_tokens"]) and pd.notna(row["output_tokens"]):
                    row[f"api_usd:{model}"] = (row["input_tokens"] * price_in + row["output_tokens"] * price_out) / 1e6
        row["descriptive_only"] = spec.descriptive_only
        rows.append(row)
    frame = pd.DataFrame(rows)
    return add_cost_per_benefit(frame, api_models)


def add_cost_per_benefit(frame: pd.DataFrame, api_models: list[str]) -> pd.DataFrame:
    """Fill the `<ref>_*` cost-per-benefit columns (ref `lsa`, `kmeans`) of a `build_rows` frame.

    Extra seconds and dollars are the row's minus the reference's (k-means has no summarizer: 0 seconds
    when the dataset's seconds are summarizer seconds). The reference rows themselves and the
    LSA-against-LSA / k-means-against-k-means cases are `n/a`.
    """
    frame = frame.copy()
    by_method = frame.set_index("method")
    if "descriptive_only" not in frame:
        frame["descriptive_only"] = False
    names = ["local", *api_models]
    for ref_name, ref_label in (("lsa", LSA), ("kmeans", KMEANS)):
        columns: dict[str, list] = {k: [] for k in ("status", "points", "extra_seconds", "seconds_per_point")}
        columns.update({f"extra_usd:{n}": [] for n in names})
        columns.update({f"usd_per_point:{n}": [] for n in names})
        for _, row in frame.iterrows():
            same = row["method"] == ref_label or ref_label not in by_method.index
            if same or (ref_name == "lsa" and row["family"] == "kmeans"):
                result = {"status": "n/a", "points": np.nan, "seconds_per_point": np.nan, "usd_per_point": {}}
                extra_seconds, extra_usd = np.nan, {n: np.nan for n in names}
            else:
                ref = by_method.loc[ref_label]
                # One-seed runs are descriptive: their seconds sit on different shared-machine conditions and bases.
                extra_seconds = np.nan if row["descriptive_only"] else row["seconds"] - ref["seconds"]
                extra_usd = {"local": row["local_usd"] - ref["local_usd"]}
                for model in api_models:
                    extra_usd[model] = row[f"api_usd:{model}"] - ref[f"api_usd:{model}"]
                result = cost_per_benefit(
                    row["vs_" + ref_name + "_acc_diff"],
                    row["vs_" + ref_name + "_acc_lo"],
                    row["vs_" + ref_name + "_acc_hi"],
                    extra_seconds if pd.notna(extra_seconds) else np.nan,
                    extra_usd,
                )
            columns["status"].append(result["status"])
            columns["points"].append(result["points"])
            columns["extra_seconds"].append(extra_seconds)
            columns["seconds_per_point"].append(result["seconds_per_point"])
            for n in names:
                columns[f"extra_usd:{n}"].append(extra_usd.get(n, np.nan))
                columns[f"usd_per_point:{n}"].append(result["usd_per_point"].get(n, np.nan))
        for key, values in columns.items():
            frame[f"{ref_name}_{key}"] = values
    return frame


def resolve_results_root(path: str | Path) -> Path:
    """The directory that holds the run directories: `path/results` when it exists, else `path` itself.

    So `--results` accepts either a checkout or its `results/` directory.
    """
    path = Path(path)
    return path / "results" if (path / "results").is_dir() else path


def collect(spec: DatasetSpec, root: str | Path, prices: Prices | None = None) -> pd.DataFrame:
    """Read `spec`'s run directories under `root` and build its rows (`build_rows`)."""
    dirs = [Path(root) / d for d in spec.run_dirs]
    missing = [str(d) for d in dirs if not d.is_dir()]
    if missing:
        raise FileNotFoundError(f"run directories not found under {root}: {missing}")
    per_seed = per_seed_table(dirs)
    paired = paired_differences(per_seed, list(REFERENCES))
    return build_rows(spec, cost_table(dirs), per_seed, paired, _config_batch_sizes(dirs), prices)


# ---------------------------------------------------------------------------------------- rendering


def _num(value, spec: str = ",.0f", empty: str = "n/a") -> str:
    return empty if value is None or pd.isna(value) else format(value, spec)


def _pm(mean, sd, spec: str = ".3f") -> str:
    return "n/a" if pd.isna(mean) else f"{mean:{spec}} ± " + ("n/a" if pd.isna(sd) else f"{sd:{spec}}")


def _interval(diff, lo, hi) -> str:
    if pd.isna(diff):
        return "n/a"
    if pd.isna(lo):
        return f"{diff:+.3f} (n=1, no interval)"
    return f"{diff:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def row_name(row) -> str:
    """Display name: `k-means`, `LSA`, `Qwen3-1.7B, October (#31)`."""
    if row["family"] == "kmeans":
        return "k-means"
    if row["family"] == "lsa":
        return "k-NLPmeans (LSA)"
    return f"k-SLMmeans Qwen3-{row['model']}, {row['prompt']}"


def short_name(row) -> str:
    """Label for figure points: `LSA`, `0.6B`, `v2 0.6B`."""
    if row["family"] == "kmeans":
        return "k-means"
    if row["family"] == "lsa":
        return "LSA"
    return ("v2 " if row["prompt"].startswith("v2") else "") + row["model"]


def _table(header: list[str], body: list[list[str]], left: tuple[str, ...] = ()) -> str:
    """Markdown table; the first column and the columns named in `left` are left-aligned, the rest right-aligned."""
    align = ["---" if i == 0 or h in left else "---:" for i, h in enumerate(header)]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _dollar_headers(api_models: list[str], per: str) -> list[str]:
    """Dollar column headers: local, then one per API model (a single empty placeholder before #10)."""
    return [f"local $ {per}", *([f"API $ {per} ({m})" for m in api_models] or [f"API $ {per}"])]


def _dollar_cells(values: list) -> list[str]:
    return [_num(v, ".4f", "") for v in values]


def render_cost(frame: pd.DataFrame, spec: DatasetSpec, api_models: list[str] | None = None) -> str:
    """Cost columns of one dataset: seeds, batch, calls, tokens, seconds, dollars (empty until priced) and the timing source."""
    api_models = api_models or []
    header = ["Method", "seeds", "batch", "calls", "input tokens", "output tokens", "seconds"]
    header += _dollar_headers(api_models, "per seed") + ["idle-card s per call (batch)", "timing", "timing source"]
    body = []
    for _, r in frame.iterrows():
        idle = "n/a" if pd.isna(r["idle_s_per_call"]) else f"{r['idle_s_per_call']:.2f} ({int(r['idle_batch'])})"
        seconds = _pm(r["seconds"], r["seconds_sd"], ",.1f") if r["family"] != "kmeans" or r["seconds_basis"] == "run total" else "0 (no summarizer)"
        body.append(
            [
                row_name(r),
                str(r["seeds"]),
                _num(r["batch_size"], ".0f", "n/a"),
                _num(r["calls"], ",.0f", "not logged" if r["family"] != "kmeans" else "n/a"),
                _num(r["input_tokens"], ",.0f", "not logged" if r["family"] == "slm" else "0"),
                _num(r["output_tokens"], ",.0f", "not logged" if r["family"] == "slm" else "0"),
                seconds + ("" if r["seconds_basis"] == "summarizer" else " (run total)"),
                *_dollar_cells([r["local_usd"], *([r[f"api_usd:{m}"] for m in api_models] or [np.nan])]),
                idle,
                r["timing_code"],
                r["timing_source"],
            ]
        )
    return _table(header, body, left=("timing", "timing source"))


def render_benefit(frame: pd.DataFrame) -> str:
    """ACC / NMI / ARI and the paired differences against LSA and against k-means (mean [95% interval over seeds])."""
    header = ["Method", "ACC", "NMI", "ARI"]
    header += [f"{m.upper()} vs LSA" for m in METRICS] + [f"{m.upper()} vs k-means" for m in METRICS]
    body = []
    for _, r in frame.iterrows():
        cells = [row_name(r)] + [_pm(r[f"{m}_mean"], r[f"{m}_sd"]) for m in METRICS]
        for ref in ("lsa", "kmeans"):
            skip = (ref == "lsa" and r["family"] in ("lsa", "kmeans")) or (ref == "kmeans" and r["family"] == "kmeans")
            cells += ["n/a" if skip else _interval(*(r[f"vs_{ref}_{m}_{c}"] for c in ("diff", "lo", "hi"))) for m in METRICS]
        body.append(cells)
    return _table(header, body)


_STATUS_TEXT = {
    "no_benefit": "no measured benefit",
    "loss": "measured loss",
    "n/a": "n/a",
    "benefit_no_extra_cost": "benefit at no extra cost",
}


def render_cost_per_benefit(frame: pd.DataFrame, api_models: list[str]) -> str:
    """Extra seconds (and dollars when priced) per ACC point over LSA and over k-means, interval-gated."""
    header = ["Method"]
    for ref in ("LSA", "k-means"):
        header += [f"ACC vs {ref}", f"extra s vs {ref}", f"s per ACC point vs {ref}"]
        header += _dollar_headers(api_models, f"per ACC point vs {ref}")
    body = []
    for _, r in frame.iterrows():
        cells = [row_name(r)]
        for ref in ("lsa", "kmeans"):
            status = r[f"{ref}_status"]
            points = _num(100 * r[f"vs_{ref}_acc_diff"], "+.2f") if status != "n/a" else "n/a"
            if status == "benefit":
                per = _num(r[f"{ref}_seconds_per_point"], ",.1f")
                dollars = _dollar_cells([r[f"{ref}_usd_per_point:{n}"] for n in ("local", *api_models)] or [])
            else:
                per = _STATUS_TEXT.get(status, status)
                dollars = [""] * (1 + len(api_models))
            dollars += [""] * (1 + max(len(api_models), 1) - len(dollars))
            cells += [points, _num(r[f"{ref}_extra_seconds"], "+,.1f") if status != "n/a" else "n/a", per, *dollars]
        body.append(cells)
    return _table(header, body)


def render_all(frames: dict[str, pd.DataFrame], specs: dict[str, DatasetSpec], prices: Prices | None) -> str:
    """The generated markdown (`cost_justification_table.md`): three tables per dataset, and the price sources."""
    api_models = sorted(prices.api) if prices else []
    out = ["Timing codes (column `timing`):\n\n" + "\n".join(f"- `{k}`: {v}" for k, v in TIMING_CODES.items()) + "\n"]
    for key, frame in frames.items():
        spec = specs[key]
        out.append(f"## {spec.title}\n")
        if spec.descriptive_only:
            out.append("One seed, descriptive only: no paired intervals, so no cost per benefit.\n")
        out.append("### Cost\n\n" + render_cost(frame, spec, api_models) + "\n")
        out.append("### Benefit (ACC, NMI, ARI; paired difference, 95% interval over seeds)\n\n" + render_benefit(frame) + "\n")
        out.append("### Cost per ACC point (only where the paired interval excludes 0)\n\n" + render_cost_per_benefit(frame, api_models) + "\n")
    if prices is None or not prices.priced():
        out.append("Dollar columns: empty, pending #10 (no `--prices` file). CONFIRM(#10)\n")
    else:
        rows = [[item, f"{usd:g}", source, checked] for item, usd, source, checked in prices.sources]
        out.append("### Prices used\n\n" + _table(["item", "usd", "source", "checked"], rows) + "\n")
    return "\n".join(out)


def flat_csv(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """All datasets in one frame (one row per dataset and method) for the CSV."""
    return pd.concat(list(frames.values()), ignore_index=True)

