"""configs/clinc150_prompt_v2.yaml (#46) differs from the #31 config in the prompt, its names and the (result-neutral) gpu_check policy."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from kslimeans.aggregate import method_label
from kslimeans.config import load_config
from kslimeans.data import _CLINC_PROMPT
from kslimeans.experiment import _check_keys
from kslimeans.gpu_monitor import parse_gpu_check
from kslimeans.schedules import build_schedule
from kslimeans.summarizers import build_summarizer
from kslimeans.summarizers.hf import DEFAULT_SYSTEM_PROMPT, build_messages, render_conversation

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
    parse_gpu_check(cfg["gpu_check"])
    build_schedule(cfg["schedule"])
    assert cfg["summarizer"]["prompt"] == PROMPT


def test_v2_differs_from_31_only_in_the_prompt_names_and_gpu_policy():
    diff = _diff(load_config(V1), load_config(V2))
    assert set(diff) == {"name", "method_name", "baselines.kmeans", "summarizer.prompt", "gpu_check.require_idle"}
    assert diff["summarizer.prompt"] == (None, PROMPT)
    assert diff["gpu_check.require_idle"] == (None, "warn")  # a policy, not a result: #31 predates the monitor
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


def test_runner_default_steps_are_4b_then_0_6b_then_1_7b():
    line = next(l for l in (ROOT / "scripts" / "run_clinc150_prompt_v2.sh").read_text().splitlines() if "-eq 0" in l)
    assert line.endswith("STEPS=(4b 0.6b 1.7b)")


def test_summarizer_prompt_overrides_the_dataset_prompt_in_the_real_build(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr("kslimeans.summarizers.hf.HFSummarizer", Recorder)
    summarizer_cfg = load_config(V2)["summarizer"]
    build_summarizer(summarizer_cfg, prompt=_CLINC_PROMPT, text_type="Question:")
    assert seen["prompt"] == PROMPT and seen["text_type"] == "Question:"
    assert "system_prompt" not in seen  # the hf.py default system prompt stays

    build_summarizer({k: v for k, v in summarizer_cfg.items() if k != "prompt"}, prompt=_CLINC_PROMPT, text_type="Question:")
    assert seen["prompt"] == _CLINC_PROMPT


class StubTokenizer:
    """A chat template that prints every message on its own tagged line; `continue_final_message` leaves the last open."""

    def apply_chat_template(self, messages, tokenize, chat_template_kwargs=None, **kw):
        lines = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
        return "\n".join(lines)


def test_v1_and_v2_renders_differ_only_in_the_instruction_line():
    members = ["book a table for two", "reserve a table tonight"]

    def render(prompt):
        messages = build_messages(DEFAULT_SYSTEM_PROMPT, prompt, "Question:", members)
        return render_conversation(StubTokenizer(), messages, "Question:", {}).split("\n")

    old, new = render(_CLINC_PROMPT), render(PROMPT)
    assert len(old) == len(new)
    differing = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
    assert len(differing) == 1 and old[differing[0]] == _CLINC_PROMPT and new[differing[0]] == PROMPT
    assert old[-2:] == ["<|assistant|>", "Question:"] and old[1] == DEFAULT_SYSTEM_PROMPT
