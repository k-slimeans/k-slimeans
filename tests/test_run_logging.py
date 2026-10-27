"""Run logging (#30, covering #5's cost fields): per-step cost deltas, per-step assignments,
the shared-initialization hash, `environment.json` and `summary.csv`.

Everything is CPU-only with fakes: no model, tokenizer or embedding downloads.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from kslimeans import experiment
from kslimeans.core import COST_KEYS, summary_kmeans
from kslimeans.schedules import FixedSchedule, Step, SummarySchedule
from kslimeans.summarizers.base import Cluster, Summarizer
from kslimeans.summarizers.hf import HFSummarizer, count_generated_tokens

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_experiment import (  # noqa: E402
    N_CLUSTERS,
    N_DOCS,
    _base_config,
    _FakeEmbedder,
    _FakeSummarizer,
    _make_synthetic_dataset,
)


class _CountingSummarizer(Summarizer):
    """Reports cumulative counters like `HFSummarizer`; cluster 0 summarizes to "" and cluster 1 to "  "."""

    def __init__(self):
        self._stats = dict.fromkeys(("input_tokens", "output_tokens", "n_truncated_clusters", "n_dropped_texts"), 0)

    def summarize(self, cluster: Cluster) -> str:
        self._stats["input_tokens"] += 10 * len(cluster.texts)
        self._stats["output_tokens"] += 3
        self._stats["n_dropped_texts"] += 1
        self._stats["n_truncated_clusters"] += 1
        return ["", "  "][cluster.index] if cluster.index < 2 else f"summary {cluster.index} of {len(cluster.texts)}"

    def stats(self):
        return dict(self._stats)


@pytest.fixture
def synth():
    ds, X = _make_synthetic_dataset()
    return ds, X, _FakeEmbedder({t: X[i] for i, t in enumerate(ds.texts)})


def _run(synth, schedule, summarizer=None, tol=-1.0, **kwargs):
    ds, X, embedder = synth
    return summary_kmeans(
        ds.texts,
        N_CLUSTERS,
        summarizer or _CountingSummarizer(),
        embedder,
        schedule,
        text_features=X,
        tol=tol,
        random_state=0,
        verbose=False,
        **kwargs,
    )


# --- cost deltas -------------------------------------------------------------------------------


def test_step_costs_sum_to_totals_and_count_empties(synth):
    result = _run(synth, FixedSchedule(3, 30, True, True), clock=itertools.count().__next__)

    for key in COST_KEYS:
        assert sum(row[key] for row in result.trace) == result.cost[key], key
    assert all(row[key] == 0 for row in (result.trace[0],) for key in COST_KEYS)
    first = result.trace[1]
    assert first["n_empty_summaries"] == 2  # "" and whitespace-only; empty-cluster handling is separate
    assert first["output_tokens"] == 3 * first["n_calls"]
    assert first["input_tokens"] > 0 and first["n_truncated_clusters"] == first["n_calls"]
    assert first["summarizer_seconds"] == 1  # the fake clock ticks once per call
    assert result.cost["output_tokens"] == 3 * result.n_calls


def test_tol_stopped_step_cost_is_in_totals_but_not_in_trace(synth):
    result = _run(synth, FixedSchedule(3, 30, True, True), tol=1e9)

    assert result.converged and len(result.trace) == 1
    assert result.n_calls == N_CLUSTERS
    assert result.cost["output_tokens"] == 3 * N_CLUSTERS
    assert result.cost["n_empty_summaries"] == 2
    assert sum(row["output_tokens"] for row in result.trace) == 0


def test_extractive_style_summarizer_reports_zero_model_counters(synth):
    result = _run(synth, FixedSchedule(2, 20, True, True), summarizer=_FakeSummarizer())
    assert result.cost["input_tokens"] == result.cost["output_tokens"] == 0
    assert result.cost["n_truncated_clusters"] == result.cost["n_dropped_texts"] == 0
    assert result.cost["summarizer_seconds"] >= 0
    assert _FakeSummarizer().describe() == {"class": "_FakeSummarizer"}


# --- assignments_evolution / summary_source_steps ------------------------------------------------


def test_assignments_align_with_trace_full_schedule(synth):
    result = _run(synth, FixedSchedule(3, 30, True, True))

    assert len(result.assignments_evolution) == len(result.trace) == 4
    assert result.summary_source_steps == [0, 1, 2]
    assert np.array_equal(result.assignments_evolution[-1], result.assignments)


def test_assignments_align_with_trace_when_last_summary_stops_on_tol(synth):
    result = _run(synth, FixedSchedule(3, 30, True, True), tol=1e9)
    assert len(result.assignments_evolution) == len(result.trace) == 1
    assert result.summary_source_steps == [0]  # no assignments_evolution[1]: that step fit nothing


class _MaskedMixed(SummarySchedule):
    _steps = [Step(False, 3), Step(True, 5, [True, False, True, False]), Step(False, 4), Step(True, 5)]

    def first_iters(self) -> int:
        return 10

    def next_step(self, state):
        return self._steps[state.step] if state.step < len(self._steps) else None


def test_assignments_align_with_trace_for_partial_schedule(synth):
    result = _run(synth, _MaskedMixed())

    assert len(result.assignments_evolution) == len(result.trace) == 5
    assert result.summary_source_steps == [1, 3]
    assert [r["summarize"] for r in result.trace] == [False, False, True, False, True]
    assert np.array_equal(result.assignments_evolution[-1], result.assignments)


# --- HFSummarizer counting (fake tokenizer / model) ---------------------------------------------


def test_count_generated_tokens_stops_at_first_eos_and_excludes_padding():
    eos, pad = 1, 0
    new = torch.tensor(
        [
            [5, 6, eos, pad, pad],  # 3: includes the EOS, not the padding
            [5, 6, 7, 8, 9],  # no EOS: all 5
            [eos, pad, pad, pad, pad],  # 1
            [5, eos, 6, eos, pad],  # first EOS only: 2
        ]
    )
    assert count_generated_tokens(new, eos) == 3 + 5 + 1 + 2


def test_count_generated_tokens_handles_eos_list_and_none():
    new = torch.tensor([[5, 2, 0, 0], [5, 6, 1, 0]])
    assert count_generated_tokens(new, [1, 2]) == 2 + 3
    assert count_generated_tokens(new, None) == 8
    assert count_generated_tokens(new, [1]) == 4 + 3


class _Enc(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    """Whitespace tokens; token id = word length + 10 (so never an eos/pad id)."""

    padding_side = "left"
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize, **kwargs):
        return " ".join(m["content"] for m in messages)

    def __call__(self, text, return_tensors=None, padding=False, add_special_tokens=False):
        if isinstance(text, str):
            return {"input_ids": [len(w) + 10 for w in text.split()]}
        ids = [[len(w) + 10 for w in t.split()] for t in text]
        width = max(map(len, ids))
        input_ids = torch.tensor([[0] * (width - len(i)) + i for i in ids])
        mask = torch.tensor([[0] * (width - len(i)) + [1] * len(i) for i in ids])
        return _Enc(input_ids=input_ids, attention_mask=mask)

    def batch_decode(self, tokens, skip_special_tokens=True):
        return [" tok " for _ in tokens]


class _FakeModel(torch.nn.Module):
    """Appends a fixed block of new tokens: row 0 stops after 2 tokens, row 1 never stops."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.generation_config = SimpleNamespace(eos_token_id=[1, 2])

    def generate(self, input_ids, attention_mask, **kwargs):
        new = torch.tensor([[7, 2, 0, 0], [7, 8, 9, 7]])[: len(input_ids)]
        return torch.cat([input_ids, new], dim=1)


def _fake_hf(max_input_tokens=4096) -> HFSummarizer:
    summarizer = HFSummarizer.__new__(HFSummarizer)
    summarizer.model_id = "fake/model"
    summarizer.prompt, summarizer.text_type, summarizer.system_prompt = "P", "", "S"
    summarizer.max_new_tokens, summarizer.batch_size = 4, 2
    summarizer.max_input_tokens = max_input_tokens
    summarizer.chat_template_kwargs = {"enable_thinking": False}
    summarizer.device = "cpu"
    summarizer.tokenizer, summarizer.model = _FakeTokenizer(), _FakeModel()
    summarizer._fold_system, summarizer._warned_trim = False, True
    summarizer._stats = dict.fromkeys(("input_tokens", "output_tokens", "n_truncated_clusters", "n_dropped_texts"), 0)
    return summarizer


def _clusters(*text_lists):
    return [Cluster(index=i, doc_ids=tuple(range(len(t))), texts=t) for i, t in enumerate(text_lists)]


def test_hf_counts_input_tokens_without_padding_and_output_tokens_through_eos():
    summarizer = _fake_hf()
    # Prompts "S P a b c" (5 tokens) and "S P a" (3): the shorter one is left-padded to 5.
    outputs = summarizer.summarize_batch(_clusters(["a b c"], ["a"]))

    assert outputs == ["tok", "tok"]
    stats = summarizer.stats()
    assert stats["input_tokens"] == 5 + 3
    assert stats["output_tokens"] == 2 + 4  # eos list [1, 2]: row 0 ends at its 2; row 1 has no eos
    assert stats["n_truncated_clusters"] == stats["n_dropped_texts"] == 0
    summarizer.summarize_batch(_clusters(["a b c"], ["a"]))
    assert summarizer.stats()["input_tokens"] == 16  # cumulative; callers take deltas


def test_hf_falls_back_to_tokenizer_eos_when_generation_config_has_none():
    summarizer = _fake_hf()
    summarizer.model.generation_config.eos_token_id = None
    summarizer.tokenizer.eos_token_id = 7
    summarizer.summarize_batch(_clusters(["a"], ["a"]))
    assert summarizer.stats()["output_tokens"] == 1 + 1


def test_hf_counts_truncated_clusters_and_dropped_texts():
    summarizer = _fake_hf(max_input_tokens=6)  # "S P" + 4 more words fit
    summarizer.summarize_batch(_clusters(["w1 w2", "w3 w4", "w5 w6", "w7 w8"], ["w1"], [" ".join(["x"] * 20)]))

    stats = summarizer.stats()
    assert stats["n_truncated_clusters"] == 2  # first dropped texts, third truncated characters
    assert stats["n_dropped_texts"] == 2


def test_hf_describe_is_json_serializable_and_reports_parameter_dtype():
    description = _fake_hf().describe()
    json.dumps(description)
    assert description["dtype"] == "torch.float32"
    assert description["padding_side"] == "left"
    assert description["batch_size"] == 2 and description["max_new_tokens"] == 4
    assert description["chat_template_kwargs"] == {"enable_thinking": False}


# --- run_offline outputs ----------------------------------------------------------------------


def _offline(tmp_path, monkeypatch, summarizer=None, **overrides):
    ds, X = _make_synthetic_dataset()
    embedder = _FakeEmbedder({t: X[i] for i, t in enumerate(ds.texts)})
    monkeypatch.setattr(experiment, "load_dataset", lambda name, **kwargs: ds)
    monkeypatch.setattr(experiment, "get_embedder", lambda name: embedder)
    monkeypatch.setattr(
        experiment, "build_summarizer", lambda cfg, prompt=None, text_type=None: summarizer or _CountingSummarizer()
    )
    return experiment.run_offline(_base_config(tmp_path, **overrides))


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_offline_logs_cost_assignments_and_shared_init(tmp_path, monkeypatch):
    run_dir = _offline(tmp_path, monkeypatch, seeds=[0, 1], tol=-1.0)
    rows = _jsonl(run_dir / "metrics.jsonl")
    base = {r["seed"]: r for r in rows if r["method"] == "kmeans"}
    method = {r["seed"]: r for r in rows if r["method"] == "k-Fakemeans"}

    for seed in (0, 1):
        assert base[seed]["step0_sha1"] == method[seed]["step0_sha1"]
        assert set(COST_KEYS) <= set(method[seed])
        assert {"peak_gpu_mem_gb", "peak_gpu_reserved_gb"} <= set(method[seed])
        assert set(COST_KEYS).isdisjoint(base[seed])
    assert base[0]["step0_sha1"] != base[1]["step0_sha1"]
    assert method[0]["output_tokens"] > 0 and method[0]["n_empty_summaries"] >= 2

    trace = [r for r in _jsonl(run_dir / "trace.jsonl") if r["seed"] == 0]
    assert sum(r["output_tokens"] for r in trace) == method[0]["output_tokens"]  # tol=-1: no hidden step

    for row in _jsonl(run_dir / "summaries.jsonl"):
        n_fits = len(row["assignments_evolution"])
        assert n_fits == len([r for r in _jsonl(run_dir / "trace.jsonl") if r["seed"] == row["seed"]])
        assert len(row["assignments_evolution"][0]) == N_DOCS
        assert row["assignments_evolution"][-1] == row["assignments"]
        assert len(row["summary_source_steps"]) == len(row["summaries_evolution"])


def test_summary_csv_has_cost_columns_and_tolerates_baseline_nans(tmp_path, monkeypatch):
    run_dir = _offline(tmp_path, monkeypatch, seeds=[0, 1])
    summary = pd.read_csv(run_dir / "summary.csv", header=[0, 1], index_col=0)

    columns = set(summary.columns.get_level_values(0))
    assert {"acc", "n_calls", "input_tokens", "output_tokens", "summarizer_seconds", "runtime_s"} <= columns
    assert np.isnan(summary.loc["kmeans", ("input_tokens", "mean")])
    assert summary.loc["k-Fakemeans", ("input_tokens", "mean")] > 0


def test_environment_json_written_with_expected_keys(tmp_path, monkeypatch):
    run_dir = _offline(tmp_path, monkeypatch, seeds=[0])
    env = json.loads((run_dir / "environment.json").read_text())

    expected = {
        "timestamp", "git_commit", "git_dirty", "python", "platform", "is_wsl", "torch", "torch_cuda", "cudnn",
        "cuda_available", "gpu_name", "gpu_total_memory_gb", "nvidia_driver", "packages", "summarizer",
        "embedder", "dataset",
    }  # fmt: skip
    assert expected <= set(env)
    assert set(env["packages"]) == {"transformers", "sentence-transformers", "scikit-learn", "numpy"}
    assert env["summarizer"]["class"] == "_CountingSummarizer"
    assert env["embedder"] == "fake"
    assert env["dataset"] == {"name": "synth", "n_texts": N_DOCS, "n_clusters": N_CLUSTERS}
    assert isinstance(env["is_wsl"], bool)
