"""`scripts/bbc_report.py` on hand-written runs: OOV and zero-vector totals, step-0 and final partition tables."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bbc_report.py"
NAMES = ["a", "b", "c"]
GOLD = np.array([0] * 12 + [1] * 12 + [2] * 12)


def _load_script():
    spec = importlib.util.spec_from_file_location("bbc_report_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(path, name, rows, metrics):
    path.mkdir(parents=True)
    (path / "config.yaml").write_text(yaml.safe_dump({"name": name, "dataset": "d", "embedder": "e"}))
    (path / "metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metrics))
    (path / "summaries.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _labels(merged: bool) -> list[int]:
    """Gold-aligned labels, or topics a and b merged into cluster 0 with cluster 2 left empty."""
    if not merged:
        return [int(g) for g in GOLD]
    return [0] * 24 + [1] * 12


@pytest.fixture
def run(tmp_path):
    rows = [
        {
            "seed": 0,
            "method": "k-SLMmeans",
            "summaries": ["x", "y", "z"],
            "assignments": _labels(False),
            "summaries_evolution": [["p", "q", None], ["r", "s", "t"]],
            "assignments_evolution": [_labels(True), _labels(False)],
        }
    ]
    metrics = [
        {"seed": 0, "method": "kmeans"},
        {"seed": 0, "method": "k-SLMmeans", "embed_tokens": 40, "embed_oov_tokens": 10, "embed_zero_vectors": 1},
        {"seed": 1, "method": "k-SLMmeans", "embed_tokens": 20, "embed_oov_tokens": 5, "embed_zero_vectors": 0},
    ]
    _write_run(tmp_path / "run", "r", rows, metrics)
    return tmp_path / "run"


def test_oov_rows_sum_over_seeds_and_skip_baselines(run):
    rows = _load_script().oov_rows([run])

    assert len(rows) == 1  # the kmeans baseline has no embed counters
    row = rows[0]
    assert (row["run"], row["method"]) == ("r", "k-SLMmeans")
    assert (row["tokens"], row["oov_tokens"], row["zero_vectors"]) == (60, 15, 1)
    assert row["oov_share"] == pytest.approx(0.25)
    assert row["summaries"] == 5  # the None is not a summary


def test_oov_rows_skip_runs_without_counters(tmp_path):
    _write_run(tmp_path / "dense", "d", [], [{"seed": 0, "method": "k-SLMmeans"}])

    assert _load_script().oov_rows([tmp_path / "dense"]) == []


def test_crosstab_and_matched_accuracy():
    module = _load_script()
    counts = module.crosstab(np.array(_labels(True)), GOLD, 3)

    assert counts.tolist() == [[12, 12, 0], [0, 0, 12], [0, 0, 0]]
    assert module.matched_accuracy(counts) == pytest.approx(24 / 36)
    assert module.matched_accuracy(module.crosstab(np.array(_labels(False)), GOLD, 3)) == 1.0


def test_composition_names_topics_with_enough_members():
    module = _load_script()

    assert module.composition(np.array([93, 3, 95]), NAMES) == "c 95 + a 93"
    assert module.composition(np.array([9, 9, 9]), NAMES) == "mixed (no topic has 10)"


def test_partition_rows_step0_once_per_seed_and_final(run):
    module = _load_script()
    rows = module.partition_rows([run, run], GOLD, NAMES)

    step0 = [r for r in rows if r["partition"] == "step0"]
    final = [r for r in rows if r["partition"] == "final"]
    assert len(step0) == 3  # one seed, three clusters, the second run adds no second step-0
    assert [r["acc"] for r in step0] == [pytest.approx(24 / 36)] * 3
    assert [r["composition"] for r in step0] == ["a 12 + b 12", "c 12", "mixed (no topic has 10)"]
    assert len(final) == 6  # two runs, one seed each
    assert [r["acc"] for r in final] == [1.0] * 6


def test_script_writes_tables(run, tmp_path, monkeypatch):
    ids = tmp_path / "ids.csv"
    ids.write_text("# sha256: x\nfile_id,topic\n" + "".join(f"{n}/{i},{n}\n" for n in NAMES for i in range(12)))
    out = {k: tmp_path / k for k in ("oov.md", "oov.csv", "part.md", "part.csv")}
    argv = ["bbc_report.py", str(run), "--ids", str(ids)]
    for flag, key in (("--oov-md", "oov.md"), ("--oov-csv", "oov.csv"), ("--partitions-md", "part.md"), ("--partitions-csv", "part.csv")):
        argv += [flag, str(out[key])]
    monkeypatch.setattr(sys, "argv", argv)

    _load_script().main()

    assert "| r | k-SLMmeans | 5 | 60 | 15 | 25.0% | 1 |" in out["oov.md"].read_text()
    assert "| step0 | all methods | 0 | 0.667 | a 12 + b 12 ; c 12 ; mixed (no topic has 10) |" in out["part.md"].read_text()
    assert out["oov.csv"].read_text().startswith("run,method,summaries,tokens")
    assert out["part.csv"].read_text().startswith("partition,run,seed,acc,cluster,size,a,b,c,composition")


def test_composition_boundary_is_inclusive():
    module = _load_script()

    assert module.composition(np.array([10, 9, 0]), NAMES) == "a 10"
    assert module.composition(np.array([10, 10, 9]), NAMES) == "a 10 + b 10"
    assert module.composition(np.array([9, 9, 9]), NAMES) == f"mixed (no topic has {module.MIN_SHOWN})"


def test_k_must_match_the_topics(run):
    module = _load_script()

    with pytest.raises(ValueError, match="topics but 2 names"):
        module.partition_rows([run], GOLD, NAMES[:2])
    with pytest.raises(ValueError, match="one cluster per topic"):
        module.crosstab(np.array([0, 3]), np.array([0, 1]), 3)
