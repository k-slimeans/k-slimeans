"""`kslimeans.prompt_test`: the pilot rows, the question-form check and the example selection, on synthetic data."""

from __future__ import annotations

import json

import pytest
import yaml

from kslimeans.prompt_test import (
    format_example,
    gpu_idle_at_start,
    latest_stamp,
    old_prompt_flags,
    pilot_row,
    question_form,
    read_outputs,
    select_examples,
)


def _write_pilot(path, summaries, batch_size=4, logged_match=1.0, idle=True):
    path.mkdir(parents=True)
    (path / "config.yaml").write_text(yaml.safe_dump({"name": path.parent.name, "summarizer": {"batch_size": batch_size}}))
    rows = [{"repeat": 0, "cluster": i, "summary": s} for i, s in enumerate(summaries)]
    rows += [{"repeat": 1, "cluster": 0, "summary": "ignored"}]
    (path / "outputs.jsonl").write_text("\n".join(json.dumps(r) for r in reversed(rows)) + "\n")
    (path / "cost.jsonl").write_text(json.dumps({"input_tokens": 100, "output_tokens": 7}) + "\n")
    surface = {
        "n": len(summaries), "mentions": 1, "duplicate_groups": 1, "clusters_in_duplicate_groups": 2,
        "largest_duplicate_group": 2, "mean_words": 3.456, "ends_with_question": 1, "logged_match": logged_match,
    }
    (path / "surface.json").write_text(json.dumps(surface))
    if idle is not None:
        (path / "gpu.json").write_text(json.dumps({"idle_check": {"available": True, "idle": idle}}))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("how old are you", True),
        ("What is your age?", True),
        ("**How can I reset it?**", True),
        ('"Can you book an Uber?"', True),
        ("play the next song on the playlist", False),
        ("please remove an item from my list", False),
        ("The cluster of questions revolves around money.", False),
        ("", False),
    ],
)
def test_question_form(text, expected):
    assert question_form(text) is expected


def test_read_outputs_uses_repeat_zero_in_cluster_order(tmp_path):
    _write_pilot(tmp_path / "pilot_v2_4b" / "s1", ["a?", "b", "c"])
    assert read_outputs(tmp_path / "pilot_v2_4b" / "s1") == ["a?", "b", "c"]


def test_pilot_row(tmp_path):
    path = tmp_path / "pilot_v2_4b" / "s1"
    _write_pilot(path, ["why?", "how old are you", "set a timer", "The cluster is here."], batch_size=1, logged_match=0.02)
    row = pilot_row("4b", "v2", path)
    assert row["batch_size"] == 1 and row["n"] == 4 and row["mean_words"] == 3.46
    assert row["no_question_mark"] == 3 and row["no_question_mark_question_form"] == 1
    assert (row["input_tokens"], row["output_tokens"]) == (100, 7)
    assert row["equal_to_logged"] == 0.02 and row["gpu_idle_at_start"] is True
    assert row["run_dir"] == "pilot_v2_4b/s1"


def test_gpu_idle_at_start_missing_or_unavailable(tmp_path):
    assert gpu_idle_at_start(tmp_path) is None
    (tmp_path / "gpu.json").write_text(json.dumps({"idle_check": {"available": False}}))
    assert gpu_idle_at_start(tmp_path) is None
    (tmp_path / "gpu.json").write_text(json.dumps({"idle_check": {"available": True, "idle": False}}))
    assert gpu_idle_at_start(tmp_path) is False


def test_latest_stamp(tmp_path):
    for stamp in ("20261112-100000", "20261112-150000"):
        (tmp_path / "pilot_old_4b" / stamp).mkdir(parents=True)
    assert latest_stamp(tmp_path, "pilot_old_4b").name == "20261112-150000"
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        latest_stamp(tmp_path, "empty")


OLD = {
    "4b": ["What is the cluster about?", "dup", "dup", "fine", "ok one", "ok two", "ok three", "ok four"],
    "0.6b": ["a", "b", "c", "Summarize this", "d", "e", "f", "g"],
}


def test_old_prompt_flags():
    flags = old_prompt_flags(OLD)
    assert flags == {0: ["4b mentions the cluster"], 1: ["4b duplicate"], 2: ["4b duplicate"], 3: ["0.6b mentions the cluster"]}


def test_select_examples_is_seeded_sorted_and_flag_first():
    chosen = select_examples(OLD, n=6, n_flagged=2, seed=1)
    assert chosen == select_examples(OLD, n=6, n_flagged=2, seed=1)
    assert chosen == sorted(chosen) and len(chosen) == 6 and len(set(chosen)) == 6
    assert sum(i in {0, 1, 2, 3} for i in chosen) == 2


def test_select_examples_with_few_flagged_and_few_clusters():
    chosen = select_examples(OLD, n=20, n_flagged=10, seed=0)
    assert chosen == list(range(8))
    assert select_examples({"4b": ["x", "y", "z"]}, n=2, n_flagged=1) in ([0, 1], [0, 2], [1, 2])



def test_format_example_escapes_pipes_and_cuts_long_text():
    text = format_example(7, ["q1", "q2"], {"4b": "a | b"}, {"4b": "x" * 300}, ["4b duplicate"])
    assert text.startswith("**Cluster 7** (old prompt: 4b duplicate)")
    assert "- q1" in text and "a \\| b" in text and "[...]" in text and "x" * 201 not in text
