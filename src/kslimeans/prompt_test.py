"""The pilot of the prompt test (#46): rows, a question-form check and example selection from fixed-workload outputs.

The pilot re-summarized seed 0's first summary step with the old and the clearer CLINC150 prompt at each model size
(`scripts/fixed_workload.py`, directories `pilot_<old|v2>_<size>/<stamp>`). `scripts/prompt_test_pilot.py` turns
those directories into `docs/results/prompt_test_pilot.csv` and the side-by-side examples. This module holds the
parts that need no model or dataset, so tests can drive them with plain data.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .surface import duplicate_groups, mentions_cluster
from .workload import load_workload_dir

SIZES = ("4b", "1.7b", "0.6b")
PROMPTS = ("old", "v2")

# Words that open a question in English: wh-words and the auxiliary or modal verbs of a yes/no question.
_QUESTION_OPENERS = frozenset(
    "what whats how when where why who whom whose which whether "
    "can could do does did is isn't are aren't am was were will would shall should may might have has had".split()
)
_WORD = re.compile(r"[a-z][a-z']*")

CSV_FIELDS = (
    "size", "prompt", "batch_size", "n", "mentions", "duplicate_groups", "clusters_in_duplicate_groups",
    "largest_duplicate_group", "mean_words", "ends_with_question", "no_question_mark", "no_question_mark_question_form",
    "input_tokens", "output_tokens", "equal_to_logged", "gpu_idle_at_start", "run_dir",
)


def question_form(text: str) -> bool:
    """Whether `text` opens like a question (a wh-word or an auxiliary verb), whatever it ends with."""
    match = _WORD.search(text.lower())
    return match is not None and match.group() in _QUESTION_OPENERS


def latest_stamp(root: str | Path, name: str) -> Path:
    """The newest `<root>/<name>/<stamp>` directory (stamps sort by time)."""
    stamps = sorted(p for p in (Path(root) / name).iterdir() if p.is_dir())
    if not stamps:
        raise FileNotFoundError(f"no run directory under {Path(root) / name}")
    return stamps[-1]


def read_outputs(path: str | Path) -> list[str]:
    """Repeat 0's summaries of a workload directory, in cluster order."""
    with open(Path(path) / "outputs.jsonl") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [row["summary"] for row in sorted((r for r in rows if r["repeat"] == 0), key=lambda r: r["cluster"])]


def gpu_idle_at_start(path: str | Path) -> bool | None:
    """The idle verdict `gpu.json` recorded before the model loaded (None if there is no usable check)."""
    gpu_json = Path(path) / "gpu.json"
    if not gpu_json.exists():
        return None
    with open(gpu_json) as f:
        check = json.load(f).get("idle_check") or {}
    return check.get("idle") if check.get("available") else None


def pilot_row(size: str, prompt: str, path: str | Path) -> dict:
    """One CSV row for a pilot directory: surface counts, tokens, the "?" check and the idle verdict."""
    path = Path(path)
    w = load_workload_dir(path)
    s, cost = w["surface"], w["cost"][0]
    summaries = read_outputs(path)
    without_mark = [t for t in summaries if not t.rstrip().endswith("?")]
    return {
        "size": size,
        "prompt": prompt,
        "batch_size": w["config"]["summarizer"]["batch_size"],
        "n": s["n"],
        "mentions": s["mentions"],
        "duplicate_groups": s["duplicate_groups"],
        "clusters_in_duplicate_groups": s["clusters_in_duplicate_groups"],
        "largest_duplicate_group": s["largest_duplicate_group"],
        "mean_words": round(s["mean_words"], 2),
        "ends_with_question": s["ends_with_question"],
        "no_question_mark": len(without_mark),
        "no_question_mark_question_form": sum(question_form(t) for t in without_mark),
        "input_tokens": cost.get("input_tokens", 0),
        "output_tokens": cost.get("output_tokens", 0),
        "equal_to_logged": None if s.get("logged_match") is None else round(s["logged_match"], 4),
        "gpu_idle_at_start": gpu_idle_at_start(path),
        "run_dir": f"{path.parent.name}/{path.name}",
    }


def old_prompt_flags(old: dict[str, list[str]]) -> dict[int, list[str]]:
    """Per cluster index, the old-prompt problems: "<size> mentions the cluster" and "<size> duplicate" labels."""
    flags: dict[int, list[str]] = {}
    for size, summaries in old.items():
        for i, text in enumerate(summaries):
            if mentions_cluster(text):
                flags.setdefault(i, []).append(f"{size} mentions the cluster")
        for group in duplicate_groups(summaries):
            for i in group:
                flags.setdefault(i, []).append(f"{size} duplicate")
    return flags


def select_examples(old: dict[str, list[str]], n: int = 8, n_flagged: int = 3, seed: int = 46) -> list[int]:
    """Cluster indices to show side by side, chosen by a seeded rule that never looks at the new summaries.

    `old` maps a size to the old prompt's summaries (cluster order). A cluster is flagged when any size's old
    summary mentions the cluster or is an exact duplicate of another's (`old_prompt_flags`). The rule: `n_flagged`
    flagged clusters drawn at random (`random.Random(seed)`, or all of them if fewer), then random unflagged
    clusters up to `n`; the result is sorted by index.
    """
    n_clusters = len(next(iter(old.values())))
    flagged = sorted(old_prompt_flags(old))
    rng = random.Random(seed)
    picked = rng.sample(flagged, min(n_flagged, len(flagged)))
    rest = [i for i in range(n_clusters) if i not in set(flagged)]
    picked += rng.sample(rest, min(n - len(picked), len(rest)))
    return sorted(picked)


def format_example(
    index: int, members: list[str], old: dict[str, str], new: dict[str, str], flags: list[str] = ()
) -> str:
    """Markdown for one cluster: its sampled member questions, then the old and new summary per size.

    `flags` (from `old_prompt_flags`) are named in the heading, so a reader sees why a cluster was drawn.
    """
    note = f" (old prompt: {', '.join(flags)})" if flags else ""
    lines = [f"**Cluster {index}**{note}", "", "Members (sampled):"]
    lines += [f"- {text}" for text in members]
    lines += ["", "| Size | Old prompt | Clearer prompt |", "| --- | --- | --- |"]
    for size in old:
        lines.append(f"| {size} | {_cell(old[size])} | {_cell(new[size])} |")
    return "\n".join(lines)


def _cell(text: str) -> str:
    """A table cell: one line, pipes escaped, long text cut (the cut is marked)."""
    flat = " ".join(text.split()).replace("|", "\\|")
    return flat if len(flat) <= 200 else flat[:200] + " [...]"
