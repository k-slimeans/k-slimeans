"""`scripts/estimate_stream_cost.py`: the call, token and GPU-hour arithmetic (no data, tokenizer or GPU)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "estimate_stream_cost.py"


@pytest.fixture(scope="module")
def est():
    spec = importlib.util.spec_from_file_location("estimate_stream_cost_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass looks the module up by name
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stats(est):
    return est.TokenStats(mean_abstract_tokens=200.0, prompt_overhead_tokens=50.0)


def test_batches_and_summarized_batches_round_up(est):
    assert est.n_batches(3_035_644, 10_000) == 304
    assert est.n_batches(20_000, 5_000) == 4
    assert est.summarized_batches(304, 1) == 304
    assert est.summarized_batches(304, 20) == 16  # batches 0, 20, ..., 300
    assert est.summarized_batches(4, 10) == 1  # the first batch is always summarized


def test_current_schedule_is_the_issue_upper_bound(est):
    option = est.ScheduleOption("current", 10_000, 5, 1)
    assert est.max_calls(3_035_644, 171, option) == 304 * 171 * 5 == 259_920


def test_calls_match_the_20k_smoke_run_shape(est):
    assert est.max_calls(20_000, 149, est.ScheduleOption("smoke", 5_000, 5, 1)) == 4 * 149 * 5


def test_summary_every_and_steps_scale_calls_linearly(est):
    base = est.ScheduleOption("b", 10_000, 4, 1)
    assert est.max_calls(100_000, 10, base) == 10 * 10 * 4
    assert est.max_calls(100_000, 10, est.ScheduleOption("b", 10_000, 4, 5)) == 2 * 10 * 4
    assert est.max_calls(100_000, 10, est.ScheduleOption("b", 10_000, 1, 1)) == 10 * 10


def test_tokens_per_call_uses_force_context_length_overhead_and_newlines(est, stats):
    assert est.tokens_per_call(stats, k=100, max_batch_size=10_000, force_context_length=10) == 50 + 10 * 201


def test_tokens_per_call_small_clusters_send_all_members(est, stats):
    # mean cluster of 5 members, force_context_length 10 -> only 5 abstracts
    assert est.tokens_per_call(stats, k=100, max_batch_size=500, force_context_length=10) == 50 + 5 * 201
    # force_context_length 0 means every member
    assert est.tokens_per_call(stats, k=100, max_batch_size=500, force_context_length=0) == 50 + 5 * 201


def test_tokens_per_call_is_capped_at_the_input_limit(est, stats):
    assert est.tokens_per_call(stats, k=1, max_batch_size=10_000, force_context_length=0) == est.MAX_INPUT_TOKENS


def test_seconds_per_input_token_is_the_clinc150_ratio_divided_by_the_slowdown(est):
    logged = est.CLINC150_LOGGED["Qwen3-1.7B"]
    assert est.seconds_per_input_token("Qwen3-1.7B", 1.0) == pytest.approx(
        logged["summarizer_s"] / logged["input_tokens"]
    )
    assert est.seconds_per_input_token("Qwen3-1.7B", 2.0) == pytest.approx(
        est.seconds_per_input_token("Qwen3-1.7B", 1.0) / 2
    )


def test_gpu_hours_is_calls_times_tokens_times_rate(est):
    rate = est.seconds_per_input_token("Qwen3-0.6B", 1.0)
    assert est.gpu_hours(1000, 2000, "Qwen3-0.6B", 1.0) == pytest.approx(1000 * 2000 * rate / 3600)
    assert est.gpu_hours(0, 2000, "Qwen3-0.6B") == 0
    # bigger models cost more per token
    hours = [est.gpu_hours(1000, 2000, m) for m in est.MODEL_ORDER]
    assert hours == sorted(hours)


def test_the_logged_clinc150_run_is_reproduced_at_slowdown_one(est):
    """Feeding the CLINC150 run's own calls and mean tokens/call back in returns its logged seconds."""
    for model, logged in est.CLINC150_LOGGED.items():
        tokens = logged["input_tokens"] / logged["calls"]
        hours = est.gpu_hours(logged["calls"], tokens, model, 1.0)
        assert hours * 3600 == pytest.approx(logged["summarizer_s"])


def test_kmeans_cpu_bound_does_not_depend_on_the_batching(est):
    assert est.kmeans_cpu_hours(1_000_000, 100, 120, 10.0) == pytest.approx(
        120 * 1_000_000 * 100 * 384 * 2 / 10e9 / 3600
    )


def test_estimate_table_rows(est, stats):
    options = [est.ScheduleOption("x", 10_000, 5, 1), est.ScheduleOption("y", 100_000, 1, 1)]
    rows = est.estimate_table(options, n_rows=1_000_000, k=100, stats=stats, force_context_length=10)
    assert [r["batches"] for r in rows] == [100, 10]
    assert [r["calls"] for r in rows] == [100 * 100 * 5, 10 * 100]
    assert rows[0]["tokens_per_call"] == rows[1]["tokens_per_call"] == 2060
    assert rows[0]["gpu_hours"]["Qwen3-0.6B"] == pytest.approx(
        est.gpu_hours(50_000, 2060, "Qwen3-0.6B")
    )
    assert set(rows[0]["gpu_hours"]) == set(est.MODEL_ORDER)


def test_format_table_bolds_cells_within_budget(est, stats):
    rows = est.estimate_table([est.ScheduleOption("cheap", 100_000, 1, 1)], 1_000_000, 100, stats, 10)
    assert "**" in est.format_table(rows, budget_hours=100.0)
    assert "**" not in est.format_table(rows, budget_hours=None)
    assert "cheap" in est.format_table(rows)


def test_parse_option(est):
    assert est.parse_option("150000:2") == est.ScheduleOption("150,000 batches, 2 steps", 150_000, 2, 1)
    assert est.parse_option("10000:1:4").summary_every == 4
    for bad in ("10000", "a:b", "10000:0", "1:2:3:4", "10000:-1"):
        with pytest.raises(ValueError, match="BATCH:STEPS"):
            est.parse_option(bad)


def test_default_options_cover_the_four_families(est):
    labels = [o.label for o in est.default_options()]
    assert any(label.startswith("(a)") for label in labels)
    assert any(label.startswith("(b)") for label in labels)
    assert any(label.startswith("(c") for label in labels)
    assert any(label.startswith("(d") for label in labels)
    assert any(o.summary_every > 1 for o in est.default_options())


def test_token_stats_round_trip_and_lsa_projection(est, tmp_path):
    path = tmp_path / "stats.json"
    lsa = {
        "spacy_docs_per_s": 50.0,
        "sentences_per_abstract": 6.0,
        "mean_sentence_chars": 100.0,
        "minilm_sentence_tokens_per_abstract": 240.0,
        "minilm_abstract_tokens": 200.0,
    }
    path.write_text(json.dumps({"mean_abstract_tokens": 210.0, "prompt_overhead_tokens": 53, "lsa": lsa}))
    assert est.TokenStats.from_json(path) == est.TokenStats(210.0, 53.0)

    proj = est.lsa_projection(3_600_000, lsa, embed_rows_per_s=100.0)
    assert proj["n_sentences"] == 21_600_000
    assert proj["spacy_hours"] == pytest.approx(3_600_000 / 50 / 3600)
    assert proj["embed_hours_by_tokens"] == pytest.approx(3_600_000 * 240 / (100 * 200) / 3600)
    assert proj["embed_hours_by_rows"] == pytest.approx(21_600_000 / 100 / 3600)
    assert proj["embeddings_gb"] == pytest.approx(21_600_000 * 384 * 4 / 1e9)
    assert est.lsa_projection(3_600_000, lsa, 100.0, spacy_procs=4)["spacy_hours"] == pytest.approx(
        proj["spacy_hours"] / 4
    )
