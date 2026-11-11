"""configs/clinc150_prompt_v2.yaml (#46) differs from the #31 config in the prompt and its names, nothing else."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from kslimeans.aggregate import method_label
from kslimeans.config import load_config
from kslimeans.experiment import _check_keys
from kslimeans.schedules import build_schedule

ROOT = Path(__file__).resolve().parent.parent
V1 = ROOT / "configs" / "clinc150_summarizers.yaml"
V2 = ROOT / "configs" / "clinc150_prompt_v2.yaml"

PROMPT = (
    "The following questions were asked by users. Write one new question that a user could have asked, in the same "
    "style, that covers what these questions have in common. Write only the question. Do not mention the "
    "questions, a group, a cluster or a topic."
)


def _flatten(node, prefix=""):
    if isinstance(node, dict) and node:
        for key, value in node.items():
            yield from _flatten(value, f"{prefix}{key}.")
    else:
        yield prefix.rstrip("."), node


def _diff(a: dict, b: dict) -> dict[str, tuple]:
    fa, fb = dict(_flatten(a)), dict(_flatten(b))
    return {k: (fa.get(k), fb.get(k)) for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k)}


def test_v2_config_loads_and_validates():
    cfg = load_config(V2)
    _check_keys(cfg)
    build_schedule(cfg["schedule"])
    assert cfg["summarizer"]["prompt"] == PROMPT


def test_v2_differs_from_31_only_in_the_prompt_and_names():
    diff = _diff(load_config(V1), load_config(V2))
    assert set(diff) == {"name", "method_name", "baselines.kmeans", "summarizer.prompt"}
    assert diff["summarizer.prompt"] == (None, PROMPT)
    assert diff["baselines.kmeans"] == ({}, None)  # k-means is not rerun: it does not depend on the prompt


def test_v2_names_never_collide_with_31():
    v1, v2 = load_config(V1), load_config(V2)
    assert v1["name"] != v2["name"]
    for size in ("0.6B", "1.7B", "4B"):
        for cfg in (v1, v2):
            cfg["summarizer"]["model_id"] = f"Qwen/Qwen3-{size}"
        assert method_label(v1) != method_label(v2)


def test_runner_is_valid_executable_bash_with_the_safety_settings():
    script = ROOT / "scripts" / "run_clinc150_prompt_v2.sh"
    assert os.access(script, os.X_OK)
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in text
    assert "configs/clinc150_prompt_v2.yaml" in text and "configs/clinc150_summarizers.yaml" not in text
    assert text.index("run_slm 4b") < text.index("run_slm 0.6b") < text.index("run_slm 1.7b")
