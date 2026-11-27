"""`kslimeans.cost_table` on tiny hand-made inputs: rows, the interval gate, prices and the CSV parser."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kslimeans import cost_table as ct
from kslimeans.aggregate import paired_differences

FIXTURE = Path(__file__).parent / "fixtures" / "cost_prices_example.csv"
SPEC = ct.DatasetSpec("toy", "Toy", (), "toy prompt")
SLM = "k-SLMmeans (Qwen3-1.7B)"


def _cost(rows: list[dict]) -> pd.DataFrame:
    base = {"n_seeds": 3, "batch_size": None, "gpu_idle": None, "gpu_above_baseline_mib": np.nan}
    cols = ["n_calls", "input_tokens", "output_tokens", "summarizer_seconds", "runtime_s"]
    out = []
    for r in rows:
        rec = dict(base)
        for c in cols:
            rec[f"{c}_mean"] = r.get(c, np.nan)
            rec[f"{c}_std"] = r.get(f"{c}_std", np.nan)
        rec.update({k: v for k, v in r.items() if k not in cols})
        out.append(rec)
    return pd.DataFrame(out)


def _per_seed(slm_vs_lsa: float = 0.03) -> pd.DataFrame:
    # 3 seeds. LSA beats k-means by 0.02 every seed; the SLM differs from LSA by `slm_vs_lsa` every seed.
    lsa = [0.52, 0.53, 0.54]
    acc = {"k-means": [0.50, 0.51, 0.52], ct.LSA: lsa, SLM: [a + slm_vs_lsa for a in lsa]}
    rows = [
        {"method": m, "seed": s, "acc": v, "nmi": v, "ari": v, "runtime_s": 1.0}
        for m, vals in acc.items()
        for s, v in enumerate(vals)
    ]
    return pd.DataFrame(rows)


def _rows(prices=None, per_seed=None, slm_seconds: float = 602.0) -> pd.DataFrame:
    per_seed = _per_seed() if per_seed is None else per_seed
    cost = _cost(
        [
            {"method": "k-means", "runtime_s": 1.0},
            {"method": ct.LSA, "n_calls": 10, "input_tokens": 0, "output_tokens": 0, "summarizer_seconds": 2.0, "runtime_s": 3.0},
            {
                "method": SLM,
                "n_calls": 12,
                "input_tokens": 1_000_000,
                "output_tokens": 50_000,
                "summarizer_seconds": slm_seconds,
                "runtime_s": 610.0,
                "batch_size": 4,
                "gpu_idle": True,
                "gpu_above_baseline_mib": 140.0,
            },
        ]
    )
    paired = paired_differences(per_seed, list(ct.REFERENCES))
    return ct.build_rows(SPEC, cost, per_seed, paired, prices=prices)


# ----------------------------------------------------------------------------- the interval gate


def test_gate_benefit_gives_seconds_and_dollars_per_point():
    out = ct.cost_per_benefit(0.02, 0.01, 0.03, 200.0, {"local": 1.0})
    assert out["status"] == "benefit"
    assert out["points"] == pytest.approx(2.0)
    assert out["seconds_per_point"] == pytest.approx(100.0)
    assert out["usd_per_point"]["local"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "diff,lo,hi,status",
    [(0.01, -0.01, 0.03, "no_benefit"), (-0.02, -0.03, -0.01, "loss"), (0.01, 0.0, 0.02, "no_benefit"), (0.01, np.nan, np.nan, "n/a")],
)
def test_gate_withholds_cost_without_a_clear_benefit(diff, lo, hi, status):
    out = ct.cost_per_benefit(diff, lo, hi, 200.0, {"local": 1.0})
    assert out["status"] == status
    assert np.isnan(out["seconds_per_point"]) and out["usd_per_point"] == {}


def test_gate_benefit_that_costs_nothing_extra_has_no_ratio():
    out = ct.cost_per_benefit(0.02, 0.01, 0.03, -5.0)
    assert out["status"] == "benefit_no_extra_cost"
    assert np.isnan(out["seconds_per_point"])


# ------------------------------------------------------------------------------------------- rows


def test_rows_one_per_method_in_table_order_with_counters_and_seconds():
    rows = _rows()
    assert list(rows["method"]) == ["k-means", ct.LSA, SLM]
    slm = rows.set_index("method").loc[SLM]
    assert (slm["seeds"], slm["batch_size"], slm["calls"], slm["input_tokens"]) == (3, 4, 12, 1_000_000)
    assert slm["seconds"] == 602.0 and slm["seconds_basis"] == "summarizer"
    assert slm["prompt"] == "toy prompt"
    assert rows.set_index("method").loc["k-means", "seconds"] == 0.0  # no summarizer


def test_rows_paired_differences_match_the_helper():
    rows = _rows().set_index("method")
    assert rows.loc[SLM, "vs_lsa_acc_diff"] == pytest.approx(0.03)
    assert rows.loc[SLM, "vs_kmeans_acc_diff"] == pytest.approx(0.05)
    assert rows.loc[ct.LSA, "vs_kmeans_acc_diff"] == pytest.approx(0.02)


def test_rows_gate_applies_per_reference():
    # zero spread, so both intervals are the point estimate and exclude 0: both references get a ratio
    rows = _rows().set_index("method")
    assert rows.loc[SLM, "lsa_status"] == "benefit"
    assert rows.loc[SLM, "lsa_seconds_per_point"] == pytest.approx((602.0 - 2.0) / 3.0)
    assert rows.loc[SLM, "kmeans_seconds_per_point"] == pytest.approx(602.0 / 5.0)
    assert rows.loc[ct.LSA, "lsa_status"] == "n/a"  # no LSA-against-LSA
    assert rows.loc["k-means", "kmeans_status"] == "n/a"


def test_rows_noisy_difference_is_no_measured_benefit():
    per_seed = _per_seed()
    per_seed.loc[(per_seed.method == SLM) & (per_seed.seed == 0), ["acc", "nmi", "ari"]] -= 0.10
    rows = _rows(per_seed=per_seed).set_index("method")
    assert rows.loc[SLM, "lsa_status"] == "no_benefit"
    assert np.isnan(rows.loc[SLM, "lsa_seconds_per_point"])


def test_rows_loss_status_has_no_ratio():
    rows = _rows(per_seed=_per_seed(slm_vs_lsa=-0.03)).set_index("method")
    assert rows.loc[SLM, "lsa_status"] == "loss"
    assert rows.loc[SLM, "lsa_points"] == pytest.approx(-3.0)
    assert np.isnan(rows.loc[SLM, "lsa_seconds_per_point"])


def test_rows_benefit_that_is_not_costlier_has_no_ratio():
    rows = _rows(slm_seconds=1.0).set_index("method")  # cheaper than LSA's 2.0 s
    assert rows.loc[SLM, "lsa_status"] == "benefit_no_extra_cost"
    assert np.isnan(rows.loc[SLM, "lsa_seconds_per_point"])
    assert rows.loc[SLM, "kmeans_status"] == "benefit"  # still costlier than k-means (0 s)


def test_priced_no_benefit_row_keeps_dollars_per_point_empty():
    per_seed = _per_seed()
    per_seed.loc[(per_seed.method == SLM) & (per_seed.seed == 0), ["acc", "nmi", "ari"]] -= 0.10
    rows = _rows(prices=ct.load_prices(FIXTURE), per_seed=per_seed).set_index("method")
    assert rows.loc[SLM, "lsa_status"] == "no_benefit"
    assert rows.loc[SLM, "local_usd"] > 0  # priced
    assert np.isnan(rows.loc[SLM, "lsa_usd_per_point:local"])
    assert np.isnan(rows.loc[SLM, "lsa_usd_per_point:test-model"])


def test_descriptive_only_rows_have_no_extra_seconds():
    spec = ct.DatasetSpec("toy", "Toy", (), "p", descriptive_only=True)
    per_seed = _per_seed()
    per_seed = per_seed[per_seed.seed == 0]
    cost = _cost([{"method": "k-means", "runtime_s": 118.0}, {"method": ct.LSA, "runtime_s": 39.0}, {"method": SLM, "runtime_s": 5885.0}])
    rows = ct.build_rows(spec, cost, per_seed, paired_differences(per_seed, list(ct.REFERENCES)))
    assert rows[["lsa_extra_seconds", "kmeans_extra_seconds"]].isna().all().all()


def test_rows_run_total_seconds_when_no_summarizer_seconds_logged():
    cost = _cost([{"method": "k-means", "runtime_s": 0.3}, {"method": ct.LSA, "runtime_s": 1.6}, {"method": SLM, "runtime_s": 100.0}])
    per_seed = _per_seed()
    rows = ct.build_rows(SPEC, cost, per_seed, paired_differences(per_seed, list(ct.REFERENCES))).set_index("method")
    assert rows["seconds_basis"].eq("run total").all()
    assert rows.loc["k-means", "seconds"] == 0.3
    assert rows.loc[SLM, "lsa_extra_seconds"] == pytest.approx(98.4)
    assert np.isnan(rows.loc[SLM, "calls"])


def test_single_seed_rows_have_no_interval_and_no_ratio():
    per_seed = _per_seed()
    per_seed = per_seed[per_seed.seed == 0]
    rows = _rows(per_seed=per_seed).set_index("method")
    assert rows.loc[SLM, "lsa_status"] == "n/a"
    assert np.isnan(rows.loc[SLM, "vs_lsa_acc_lo"])


def test_timing_source_names_batch_and_idle_status():
    code, text = ct.timing_source(ct.describe(SLM), True, 140.0, "toy", 4)
    assert code == "A" and "batch 4" in text and "idle at start" in text
    code, text = ct.timing_source(ct.describe(SLM), True, 1625.0, "toy", 4)
    assert code == "D" and "not clean" in text
    code, text = ct.timing_source(ct.describe(SLM), None, None, "toy", 8)
    assert code == "C" and "not measured" in text
    code, text = ct.timing_source(ct.describe("k-SLMmeans (Qwen3-0.6B)"), None, None, "clinc150", 8)
    assert code == "B" and "1.76" in text and "batch 8" in text
    code, text = ct.timing_source(ct.describe("k-SLMmeans-v2 (Qwen3-0.6B)"), None, None, "clinc150", 8)
    assert code == "C"


# ----------------------------------------------------------------------------------------- prices


def test_prices_absent_leaves_dollar_columns_empty_and_says_pending():
    rows = _rows(prices=None)
    assert not [c for c in rows.columns if c.startswith("api_usd")]
    assert rows["local_usd"].isna().all()
    assert rows.filter(like="usd_per_point").isna().all().all()
    text = ct.render_all({"toy": rows}, {"toy": SPEC}, None)
    assert "pending #10" in text and "CONFIRM(#10)" in text


def test_prices_present_fill_local_and_api_dollars():
    prices = ct.load_prices(FIXTURE)
    rows = _rows(prices=prices).set_index("method")
    assert rows.loc[SLM, "local_usd"] == pytest.approx(0.50 * 602.0 / 3600)
    assert rows.loc[SLM, "api_usd:test-model"] == pytest.approx((1_000_000 * 2.0 + 50_000 * 8.0) / 1e6)
    assert rows.loc[ct.LSA, "api_usd:test-model"] == 0.0
    # cost per ACC point: extra dollars over LSA / points
    assert rows.loc[SLM, "lsa_usd_per_point:local"] == pytest.approx(0.50 * 600.0 / 3600 / 3.0)
    assert rows.loc[SLM, "lsa_usd_per_point:test-model"] == pytest.approx(2.4 / 3.0)
    text = ct.render_all({"toy": rows.reset_index()}, {"toy": SPEC}, prices)
    assert "Prices used" in text and "TEST DATA" in text and "pending #10" not in text


def test_prices_csv_parsing():
    prices = ct.load_prices(FIXTURE)
    assert prices.gpu_usd_per_hour == 0.50
    assert prices.api == {"test-model": (2.0, 8.0)}
    assert ct.load_prices(None) is None


def test_prices_units_accept_per_prefix_and_case():
    """`per hour` and `per 1M tokens` (the wording asked for on #10) mean `hour` and `1M tokens`."""
    body = (
        "item,unit,usd,source,checked\n"
        "gpu,per hour,0.25,x,2000-01-01\n"
        "api_input:m,per 1M tokens,1,x,2000-01-01\n"
        "api_output:m, Per  1m Tokens ,2,x,2000-01-01\n"
    )
    prices = ct.parse_prices(pd.read_csv(io.StringIO(body), dtype=str, keep_default_na=False))
    assert prices.gpu_usd_per_hour == 0.25
    assert prices.api == {"m": (1.0, 2.0)}


def test_committed_price_list_parses():
    """The #10 price list in docs/ feeds the cost table as committed."""
    prices = ct.load_prices(Path(__file__).resolve().parents[1] / "docs" / "cost_assumptions.csv")
    assert prices.gpu_usd_per_hour is not None and prices.gpu_usd_per_hour > 0
    assert len(prices.api) >= 2 and all(i >= 0 and o >= 0 for i, o in prices.api.values())


@pytest.mark.parametrize(
    "body,message",
    [
        ("item,unit,usd,source\ngpu,hour,1,x\n", "lacks columns"),
        ("item,unit,usd,source,checked\ngpu,hour,abc,x,2000-01-01\n", "not a number"),
        ("item,unit,usd,source,checked\ngpu,hour,-1,x,2000-01-01\n", "non-negative"),
        ("item,unit,usd,source,checked\ngpu,day,1,x,2000-01-01\n", "per hour"),
        ("item,unit,usd,source,checked\ngpu,hour,1,,2000-01-01\n", "no source"),
        ("item,unit,usd,source,checked\ngpu,hour,1,x,2000-01-01\ngpu,hour,2,x,2000-01-01\n", "duplicate"),
        ("item,unit,usd,source,checked\nfoo,hour,1,x,2000-01-01\n", "unknown price item"),
        ("item,unit,usd,source,checked\napi_input:m,1M tokens,1,x,2000-01-01\n", "needs both"),
    ],
)
def test_prices_csv_rejects_bad_rows(tmp_path, body, message):
    path = tmp_path / "prices.csv"
    path.write_text(body)
    with pytest.raises(ValueError, match=message):
        ct.load_prices(path)


# --------------------------------------------------------------------------------- run directories


def test_collect_reads_run_directories(tmp_path):
    """End to end on hand-written run directories: a baseline-bearing LSA run and one SLM run."""
    import json

    import yaml

    def write(name, method_name, summarizer, rows, env):
        d = tmp_path / "results" / name / "stamp"
        d.mkdir(parents=True)
        cfg = {"name": name, "dataset": "toy", "embedder": {"name": "x"}, "method_name": method_name, "summarizer": summarizer}
        (d / "config.yaml").write_text(yaml.safe_dump(cfg))
        (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        if env:
            (d / "environment.json").write_text(json.dumps(env))

    def seeds(method, acc, **extra):
        return [{"seed": s, "method": method, "acc": acc + 0.01 * s, "nmi": 0.5, "ari": 0.4, "runtime_s": 2.0, **extra} for s in range(3)]

    write("lsa", "k-NLPmeans", {"type": "lsa"}, seeds("kmeans", 0.5) + seeds("k-NLPmeans", 0.52, summarizer_seconds=1.0, n_calls=5), None)
    write(
        "slm",
        "k-SLMmeans",
        {"type": "hf", "model_id": "Qwen/Qwen3-1.7B", "batch_size": 4},
        seeds("kmeans", 0.5) + seeds("k-SLMmeans", 0.55, summarizer_seconds=100.0, n_calls=6, input_tokens=1000, output_tokens=50),
        {"summarizer": {"batch_size": 4}},
    )
    spec = ct.DatasetSpec("toy", "Toy", ("lsa/stamp", "slm/stamp"), "toy prompt")
    rows = ct.collect(spec, tmp_path / "results").set_index("method")
    assert list(rows.index) == ["k-means", ct.LSA, SLM]
    assert rows.loc[SLM, "batch_size"] == 4 and rows.loc[SLM, "seconds"] == 100.0
    assert rows.loc[SLM, "vs_lsa_acc_diff"] == pytest.approx(0.03)
    with pytest.raises(FileNotFoundError):
        ct.collect(ct.DatasetSpec("x", "X", ("missing/stamp",), "p"), tmp_path / "results")


def test_resolve_results_root_accepts_checkout_or_results_dir(tmp_path):
    (tmp_path / "results").mkdir()
    assert ct.resolve_results_root(tmp_path) == tmp_path / "results"
    assert ct.resolve_results_root(tmp_path / "results") == tmp_path / "results"
    assert ct.resolve_results_root(str(tmp_path)) == tmp_path / "results"


def test_intervals_match_committed_paired_csvs_when_results_are_available():
    """Optional: set KSLIMEANS_RESULTS to a checkout (or its results/) holding the runs."""
    import os

    env = os.environ.get("KSLIMEANS_RESULTS")
    if not env:
        pytest.skip("KSLIMEANS_RESULTS not set")
    docs = Path(__file__).resolve().parent.parent / "docs" / "results"
    for key, name in (("bbc_tfidf", "bbc_tfidf_paired.csv"), ("bbc_dense", "bbc_dense_paired.csv"), ("clinc150", "prompt_test_paired.csv")):
        spec = next(d for d in ct.DATASETS if d.key == key)
        rows = ct.collect(spec, ct.resolve_results_root(env)).set_index("method")
        committed = pd.read_csv(docs / name)
        for _, r in committed[committed["reference"].isin(ct.REFERENCES) & committed["method"].isin(rows.index)].iterrows():
            ref = "lsa" if r["reference"] == ct.LSA else "kmeans"
            for col, src in (("diff", "mean_diff"), ("lo", "ci_low"), ("hi", "ci_high")):
                assert rows.loc[r["method"], f"vs_{ref}_{r['metric']}_{col}"] == pytest.approx(r[src], abs=1e-9)


def test_script_is_executable_with_shebang():
    script = Path(__file__).resolve().parent.parent / "scripts" / "cost_table.py"
    assert script.read_text().startswith("#!/usr/bin/env python")
    assert script.stat().st_mode & 0o111
    spec = importlib.util.spec_from_file_location("cost_table_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
