"""`kslimeans.surface` and `scripts/surface_counts.py` on hand-written summaries."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from kslimeans.surface import duplicate_groups, mentions_cluster, surface_counts

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "surface_counts.py"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("What is the main topic of the following cluster of questions?", True),
        ("Summarize the questions about travel.", True),
        ("CLUSTERS of things", True),
        ("The summary says nothing.", True),
        ("How do I book a flight?", False),
        ("", False),
        (None, False),
    ],
)
def test_mentions_cluster(text, expected):
    assert mentions_cluster(text) is expected


def test_duplicate_groups_exact_nonempty_only():
    summaries = ["a", "b", "a", "", "", None, None, "A", "a", "b"]
    assert duplicate_groups(summaries) == [[0, 2, 8], [1, 9]]
    assert duplicate_groups(["x", "y"]) == []
    assert duplicate_groups([]) == []


# Two seeds, two summary steps each, three clusters.
SEED_A = [['"Quoted?" yes', "two words?", "<think>x</think> hi"], ["same", "same", "about the cluster?"]]
SEED_B = [["a b c", "a b c", None], ["x", "y", "z"]]


def test_surface_counts_hand_computed():
    c = surface_counts([SEED_A, SEED_B])

    assert c.n == 11  # one None skipped
    assert c.mean_words == pytest.approx(20 / 11)
    assert c.ends_with_question == 2  # "two words?", "about the cluster?"
    assert c.mentions == 1
    assert c.starts_with_quote == 1
    assert c.think == 1
    assert (c.final_n, c.final_mentions) == (6, 1)
    assert (c.final_shared_min, c.final_shared_max) == (0, 2)
    assert c.final_largest_group == 2


def test_surface_counts_empty_input():
    c = surface_counts([])
    assert (c.n, c.mean_words, c.final_shared_min, c.final_shared_max, c.final_largest_group) == (0, 0.0, 0, 0, 0)


def test_documented_rule_gaps_are_pinned():
    # Empty summaries count toward n and mean_words (as 0 words) but never form a duplicate group.
    c = surface_counts([[["", "", "a b"]]])
    assert (c.n, c.final_n, c.final_shared_max) == (3, 3, 0)
    assert c.mean_words == pytest.approx(2 / 3)
    # Literal rules: `?"` is not a question ending, curly quotes are not quotes.
    c = surface_counts([[['Why?"', "\u201cHi\u201d"]]])
    assert (c.ends_with_question, c.starts_with_quote) == (0, 0)


def _load_script():
    spec = importlib.util.spec_from_file_location("surface_counts_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(path, summaries_rows):
    path.mkdir(parents=True)
    cfg = {"name": "r", "dataset": "d", "embedder": "e", "summarizer": {"type": "hf", "model_id": "Org/Tiny-1B"}}
    (path / "config.yaml").write_text(yaml.safe_dump(cfg))
    metrics = [{"seed": r["seed"], "method": r["method"]} for r in summaries_rows] + [{"seed": 0, "method": "kmeans"}]
    (path / "summary.csv").write_text("method\n")
    (path / "metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metrics))
    (path / "summaries.jsonl").write_text("".join(json.dumps(r) + "\n" for r in summaries_rows))


def test_script_prints_one_row_per_summarizing_method(tmp_path, capsys):
    rows = [
        {"seed": 0, "method": "k-SLMmeans", "summaries": SEED_A[-1], "summaries_evolution": SEED_A},
        {"seed": 1, "method": "k-SLMmeans", "summaries": SEED_B[-1], "summaries_evolution": SEED_B},
    ]
    _write_run(tmp_path / "run", rows)

    module = _load_script()
    argv = sys.argv
    sys.argv = ["surface_counts.py", str(tmp_path / "run")]
    try:
        module.main()
    finally:
        sys.argv = argv
    lines = capsys.readouterr().out.strip().splitlines()

    assert len(lines) == 3  # header, rule, one row: the kmeans baseline has no summaries
    assert lines[2].startswith("| r/run | Tiny-1B (k-SLMmeans) | 11 |")
    assert "| 0–2 | 2 | 1 | 1 |" in lines[2]


def test_script_skips_a_run_without_summary_csv(tmp_path, capsys):
    rows = [{"seed": 0, "method": "k-SLMmeans", "summaries": SEED_B[-1], "summaries_evolution": SEED_B}]
    _write_run(tmp_path / "done", rows)
    _write_run(tmp_path / "aborted", rows)
    (tmp_path / "aborted" / "summary.csv").unlink()

    argv = sys.argv
    sys.argv = ["surface_counts.py", str(tmp_path / "aborted"), str(tmp_path / "done")]
    try:
        _load_script().main()
    finally:
        sys.argv = argv
    captured = capsys.readouterr()

    assert len(captured.out.strip().splitlines()) == 3
    assert "r/done" in captured.out and "aborted" not in captured.out
    assert "skipping" in captured.err and "aborted" in captured.err
