"""`kslimeans.workload`, `scripts/fixed_workload.py` and `scripts/compare_workloads.py` with fake summarizers.

The logged run is a tiny synthetic one written by the real `RunDir`; nothing loads a model or reads the clock.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from kslimeans.data import Dataset
from kslimeans.runlog import load_run
from kslimeans.summarizers.base import Cluster, Summarizer
from kslimeans.workload import (
    build_workload,
    load_workload_dir,
    pick_record,
    summarize_workload,
    workload_surface,
    write_workload,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_runlog import _run_and_log, noisy  # noqa: E402,F401

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
FORCE_CONTEXT = 10  # `_run_and_log`'s default


class FakeSummarizer(Summarizer):
    """Texts from a function of the cluster; counts calls and keeps the batches it saw."""

    needs_sentences = False

    def __init__(self, text_of=lambda c, call: f"q{c.index}", batch_size=None):
        self.text_of = text_of
        self.batch_size = batch_size
        self.batches: list[list[int]] = []
        self.n_calls = 0

    def summarize(self, cluster: Cluster) -> str:
        self.n_calls += 1
        return self.text_of(cluster, self.n_calls)

    def summarize_batch(self, clusters):
        self.batches.append([c.index for c in clusters])
        return super().summarize_batch(clusters)

    def stats(self):
        return {"input_tokens": 10 * self.n_calls, "output_tokens": self.n_calls}


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _log_with_summarizer(noisy, tmp_path):
    """The synthetic run plus a `summarizer:` block in its config.yaml, as a real run has."""
    path, result, spy = _run_and_log(noisy, tmp_path)
    cfg = yaml.safe_load((path / "config.yaml").read_text())
    cfg["summarizer"] = {"type": "hf", "model_id": "Org/Tiny-1B", "batch_size": 4, "max_new_tokens": 64}
    (path / "config.yaml").write_text(yaml.safe_dump(cfg))
    return path, result, spy


def test_build_workload_reproduces_what_the_run_sent(noisy, tmp_path):
    texts, X, _ = noisy
    path, _, spy = _run_and_log(noisy, tmp_path)
    record = pick_record(load_run(path), None, 3)

    for step in range(record.n_summary_steps):
        got = build_workload(record, step, texts, X, FORCE_CONTEXT)
        sent = spy.steps[step]  # later steps skip memoized clusters, so the workload can be shorter than k
        assert [(c.index, c.doc_ids, c.texts) for c in got] == [(c.index, c.doc_ids, c.texts) for c in sent]


def test_pick_record_needs_a_method_when_ambiguous_and_summaries_when_logged(noisy, tmp_path):
    path, _, _ = _run_and_log(noisy, tmp_path)
    log = load_run(path)
    assert pick_record(log, "k-test", 3).method == "k-test"
    with pytest.raises(KeyError):
        pick_record(log, "k-test", 99)
    with pytest.raises(ValueError, match="logged no summaries"):
        pick_record(log, "kmeans", 3)
    log.records[("other", 3)] = dataclasses.replace(log.records[("k-test", 3)], method="other")
    with pytest.raises(ValueError, match="pass --method"):
        pick_record(log, None, 3)


def _clusters(n):
    return [Cluster(index=i, doc_ids=(i,), texts=[f"t{i}"]) for i in range(n)]


def test_summarize_workload_batches_repeats_and_deltas():
    ticks = itertools.count()
    fake = FakeSummarizer(batch_size=2)
    run = summarize_workload(fake, _clusters(5), repeat=3, clock=lambda: next(ticks) * 1.0)

    assert run.cluster_ids == [0, 1, 2, 3, 4]
    assert fake.batches == [[0, 1], [2, 3], [4]] * 3
    assert len(run.repeats) == 3
    for rep in run.repeats:
        assert rep.summaries == ["q0", "q1", "q2", "q3", "q4"]
        assert rep.batch_seconds == [1.0, 1.0, 1.0] and rep.seconds == 3.0
        assert rep.cost == {"input_tokens": 50, "output_tokens": 5}


def test_summarize_workload_batch_size_override_default_and_warmup():
    fake = FakeSummarizer(batch_size=2)
    summarize_workload(fake, _clusters(4), batch_size=3)
    assert fake.batches == [[0, 1, 2], [3]]

    fake = FakeSummarizer()  # no batch_size attribute value: everything in one call
    summarize_workload(fake, _clusters(4))
    assert fake.batches == [[0, 1, 2, 3]]

    fake = FakeSummarizer(batch_size=2)
    run = summarize_workload(fake, _clusters(4), warmup=True)
    assert fake.batches == [[0, 1], [0, 1], [2, 3]]  # the warm-up batch is extra and unrecorded
    assert run.repeats[0].cost == {"input_tokens": 40, "output_tokens": 4}


def test_workload_surface_counts():
    summaries = [
        "What is the main topic of the following cluster of questions?",
        "What is the main topic of the following cluster of questions?",
        "Summarize these.",
        "How do I book a flight?",
        "book a flight",
    ]
    s = workload_surface(summaries)
    assert (s["n"], s["mentions"], s["ends_with_question"]) == (5, 3, 3)
    assert s["mentions_share"] == pytest.approx(0.6)
    assert (s["duplicate_groups"], s["clusters_in_duplicate_groups"], s["largest_duplicate_group"]) == (1, 2, 2)
    assert s["mean_words"] == pytest.approx((11 + 11 + 2 + 6 + 3) / 5)
    assert s["ends_with_question_share"] == pytest.approx(0.6)
    assert workload_surface([])["mentions_share"] == 0.0


def test_write_workload_files_and_flags(tmp_path):
    fake = FakeSummarizer(text_of=lambda c, call: f"what is {c.index}?", batch_size=2)
    run = summarize_workload(fake, _clusters(3), repeat=2)
    logged = ["what is 0?", "other", "what is 2?"]
    surface = write_workload(tmp_path, run, logged)

    rows = [json.loads(line) for line in (tmp_path / "outputs.jsonl").read_text().splitlines()]
    assert [(r["repeat"], r["cluster"], r["summary"]) for r in rows] == [
        (r, c, f"what is {c}?") for r in range(2) for c in range(3)
    ]
    cost = [json.loads(line) for line in (tmp_path / "cost.jsonl").read_text().splitlines()]
    assert [(c["repeat"], c["n_calls"], c["n_batches"], c["input_tokens"]) for c in cost] == [
        (0, 3, 2, 30),
        (1, 3, 2, 30),
    ]
    assert len(cost[0]["batch_seconds"]) == 2
    assert json.loads((tmp_path / "surface.json").read_text()) == surface
    assert surface["repeats"] == 2 and surface["identical_across_repeats"] is True
    assert surface["logged_match"] == pytest.approx(2 / 3)
    assert surface["ends_with_question_share"] == 1.0


def test_write_workload_detects_changing_outputs(tmp_path):
    fake = FakeSummarizer(text_of=lambda c, call: f"answer {call}")
    surface = write_workload(tmp_path, summarize_workload(fake, _clusters(2), repeat=2))
    assert surface["identical_across_repeats"] is False
    assert "logged_match" not in surface


@pytest.fixture
def cli(noisy, tmp_path, monkeypatch):
    """The fixed_workload script with the dataset and summarizer faked; `main(*argv)` returns the out dir."""
    texts, X, _ = noisy
    path, _, spy = _log_with_summarizer(noisy, tmp_path / "logged")
    ds = Dataset("synth", texts, np.zeros(len(texts), dtype=int), 6, "DATASET PROMPT", "Question:")
    module = _load_script("fixed_workload")
    built: list[dict] = []

    def fake_build(cfg, prompt=None, text_type=None):
        built.append({"cfg": cfg, "prompt": prompt, "text_type": text_type})
        return FakeSummarizer(text_of=lambda c, call: f"{cfg.get('prompt') or prompt} {c.index}?", batch_size=cfg["batch_size"])

    monkeypatch.setattr(module, "load_run_dataset", lambda runlog, cache_dir=None: (ds, X))
    monkeypatch.setattr(module, "build_summarizer", fake_build)

    def main(*args):
        monkeypatch.setattr(sys, "argv", ["fixed_workload.py", str(path), "--seed", "3", "--results-dir", str(tmp_path / "out"), *args])
        module.main()
        return next((tmp_path / "out" / args[args.index("--name") + 1]).iterdir())

    return main, built, spy


def test_cli_reuses_the_logged_summarizer_and_writes_everything(cli):
    main, built, spy = cli
    out = main("--name", "old", "--seed", "3", "--repeat", "2", "--step", "1")

    assert built[0]["cfg"]["model_id"] == "Org/Tiny-1B" and built[0]["prompt"] == "DATASET PROMPT"
    assert {p.name for p in out.iterdir()} == {
        "config.yaml", "environment.json", "outputs.jsonl", "cost.jsonl", "surface.json"
    }
    config = yaml.safe_load((out / "config.yaml").read_text())
    assert config["source"]["method"] == "k-test" and config["source"]["step"] == 1
    assert config["source"]["n_sent"] == len(spy.steps[1]) and config["prompt"] == "DATASET PROMPT"
    outputs = [json.loads(line) for line in (out / "outputs.jsonl").read_text().splitlines()]
    assert [o["cluster"] for o in outputs if o["repeat"] == 1] == [c.index for c in spy.steps[1]]
    assert len((out / "cost.jsonl").read_text().splitlines()) == 2
    assert json.loads((out / "environment.json").read_text())["summarizer"]["class"] == "FakeSummarizer"


def test_cli_prompt_from_a_config_file_and_set_overrides(cli, tmp_path):
    main, built, _ = cli
    new = tmp_path / "v2.yaml"
    new.write_text(yaml.safe_dump({"summarizer": {"type": "hf", "model_id": "Org/Other", "prompt": "NEW", "batch_size": 8}}))
    out = main("--name", "v2", "--summarizer-config", str(new), "--set", "batch_size=3", "--set", "max_new_tokens=16")

    assert built[0]["cfg"] == {"type": "hf", "model_id": "Org/Other", "prompt": "NEW", "batch_size": 3, "max_new_tokens": 16}
    config = yaml.safe_load((out / "config.yaml").read_text())
    assert config["prompt"] == "NEW" and config["summarizer"]["batch_size"] == 3
    assert json.loads((out / "outputs.jsonl").read_text().splitlines()[0])["summary"] == "NEW 0?"


def test_resolve_summarizer_accepts_a_bare_block_and_returns_gpu_check(tmp_path):
    module = _load_script("fixed_workload")
    bare = tmp_path / "bare.yaml"
    bare.write_text(yaml.safe_dump({"type": "hf", "model_id": "M", "batch_size": 1}))
    assert module.resolve_summarizer({"summarizer": {"type": "lsa"}}, str(bare), []) == (
        {"type": "hf", "model_id": "M", "batch_size": 1},
        None,
    )
    cfg, gpu = module.resolve_summarizer({"summarizer": {"type": "hf"}, "gpu_check": {"require_idle": "warn"}}, None, [])
    assert cfg == {"type": "hf"} and gpu == {"require_idle": "warn"}


def test_compare_workloads_table(cli, tmp_path, monkeypatch, capsys):
    main, _, _ = cli
    old = main("--name", "old_run")
    new = main("--name", "v2_run", "--set", "prompt=NEW")
    capsys.readouterr()  # the script's own progress lines
    module = _load_script("compare_workloads")
    monkeypatch.setattr(sys, "argv", ["compare_workloads.py", str(old), str(new), "--show-prompt"])
    module.main()
    lines = capsys.readouterr().out.strip().splitlines()

    assert lines[0].startswith("| Run | Model | Prompt |")
    assert lines[2].startswith("| old_run | Tiny-1B | dataset | 4 |") and lines[3].startswith("| v2_run | Tiny-1B | override | 4 |")
    assert lines[-2:] == ["- old_run: DATASET PROMPT", "- v2_run: NEW"]
    assert load_workload_dir(new)["surface"]["n"] == workload_surface(
        [json.loads(line)["summary"] for line in (new / "outputs.jsonl").read_text().splitlines()]
    )["n"]
