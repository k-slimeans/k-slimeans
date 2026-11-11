"""`kslimeans.costcheck`: rebuild a logged run's cost counters and compare them with what was logged.

The runs are real `summary_kmeans` runs on the synthetic corpus, written with the real `RunDir`. Their summarizer
is a fake that renders prompts through `hf.fit_and_render` (the function `HFSummarizer` uses) with a
whitespace tokenizer and reports the token counters the way `HFSummarizer` does, so the logged numbers are known.
Seconds come from an injected fake clock; nothing here depends on wall-clock time. The one `@pytest.mark.slow`
test uses the real Qwen3 tokenizer from the local Hugging Face cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from kslimeans.cache import cached_embeddings
from kslimeans.core import COST_KEYS, summary_kmeans
from kslimeans.costcheck import (
    PromptSettings,
    check_record,
    count_tokens,
    format_report,
    rebuild_prompt,
    run_checks,
    verify_run,
)
from kslimeans.data import Dataset
from kslimeans.results import RunDir
from kslimeans.runlog import load_run
from kslimeans.schedules import FixedSchedule
from kslimeans.summarizers.base import Summarizer
from kslimeans.summarizers.hf import fit_and_render

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from test_core_equivalence import ClosestToMeanSummarizer, LookupEmbedder, synthetic  # noqa: E402,F401
from synth import N_BLOBS, make_lookup_embedder  # noqa: E402

SEED = 3
CONFIG = {"name": "synth_run", "dataset": "synth", "embedder": "fake", "seeds": [SEED]}
NOISY_FCL = 0


class WordTokenizer:
    """One token per whitespace-separated word; a chat template that prints `<role> content` per message."""

    def apply_chat_template(self, messages, tokenize=False, continue_final_message=False, **kwargs):
        assert not tokenize
        return "\n".join(f"<{m['role']}> {m['content']}" for m in messages)

    def __call__(self, text, add_special_tokens=True):
        assert not add_special_tokens
        return {"input_ids": text.split()}


SETTINGS = PromptSettings(
    prompt="Summarize:",
    text_type="Sentence:",
    system_prompt="You summarize.",
    max_input_tokens=10_000,
    max_new_tokens=64,
    chat_template_kwargs={"enable_thinking": False},
)


class FakeHFSummarizer(Summarizer):
    """Picks the member closest to the mean (as `ClosestToMeanSummarizer`), counting tokens like `HFSummarizer`.

    Input tokens are the rendered, budget-fitted prompt's words; a summary of one word costs 1 token + 1 EOS.
    """

    needs_sentences = False

    def __init__(self, embed, settings):
        self.inner = ClosestToMeanSummarizer(embed)
        self.tokenizer = WordTokenizer()
        self.settings = settings
        self._stats = {"input_tokens": 0, "output_tokens": 0, "n_truncated_clusters": 0, "n_dropped_texts": 0}

    def summarize(self, cluster):
        s = self.settings
        rendered, n_dropped, truncated = fit_and_render(
            self.tokenizer,
            cluster.texts,
            system_prompt=s.system_prompt,
            prompt=s.prompt,
            text_type=s.text_type,
            chat_template_kwargs=s.chat_template_kwargs,
            fold_system=False,
            max_input_tokens=s.max_input_tokens,
        )
        summary = self.inner.summarize(cluster)
        self._stats["input_tokens"] += len(rendered.split())
        self._stats["output_tokens"] += len(summary.split()) + 1
        self._stats["n_truncated_clusters"] += int(bool(n_dropped or truncated))
        self._stats["n_dropped_texts"] += n_dropped
        return summary

    def stats(self):
        return dict(self._stats)


class TickClock:
    """Advances 0.5 s per reading, so each summarizer call that is timed costs exactly 0.5 s."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.25
        return self.t


@pytest.fixture(scope="module")
def noisy():
    """Overlapping blobs: with short Lloyd blocks the partition changes at every step, so little is memoized."""
    rng = np.random.default_rng(1)
    centers = rng.normal(scale=2.0, size=(N_BLOBS, 8))
    X = (np.repeat(centers, 40, axis=0) + rng.normal(scale=2.0, size=(240, 8))).astype(np.float32)
    texts = [f"doc{i}" for i in range(len(X))]
    return texts, X, make_lookup_embedder(texts, X)


@pytest.fixture
def run(noisy, synthetic, tmp_path):
    """`make(...)` -> (record, texts, X, result) of a logged run; `separated=True` uses the clean corpus."""

    def make(settings=SETTINGS, tol=-1.0, extractive=False, fcl=NOISY_FCL, max_iter=5, separated=False):
        texts, X, embed = synthetic if separated else noisy
        summarizer = ClosestToMeanSummarizer(embed) if extractive else FakeHFSummarizer(embed, settings)
        result = summary_kmeans(
            texts,
            N_BLOBS,
            summarizer,
            LookupEmbedder(embed),
            FixedSchedule(3, max_iter, True, True),
            text_features=X,
            force_context_length=fcl,
            tol=tol,
            random_state=SEED,
            verbose=False,
            clock=TickClock(),
        )
        rd = RunDir(tmp_path, "synth_run", CONFIG)
        runtime = result.cost["summarizer_seconds"] + 2.0  # 2 s of clustering and embedding
        rd.log_metrics(
            {"seed": SEED, "method": "k-test", "n_calls": result.n_calls, **result.cost, "runtime_s": runtime}
        )
        rd.log_trace({"seed": SEED, "method": "k-test"}, result.trace)
        rd.log_summaries(
            {
                "seed": SEED,
                "method": "k-test",
                "summaries": result.summaries,
                "assignments": result.assignments,
                "summaries_evolution": result.summaries_evolution,
                "assignments_evolution": result.assignments_evolution,
                "summary_source_steps": result.summary_source_steps,
            }
        )
        return load_run(rd.path).record("k-test", SEED), texts, X, result

    return make


def _check(record, texts, X, settings=SETTINGS, fcl=NOISY_FCL, tokenizer=WordTokenizer()):
    return check_record(record, texts, X, fcl, tokenizer, settings)


def _by_name(checks):
    return {c.name: c for c in checks}


def _failed(checks):
    return [c.name for c in checks if not c.passed]


def test_an_honest_run_passes_every_check(run):
    record, texts, X, result = run()
    rc = _check(record, texts, X)
    checks = run_checks([rc])

    assert _failed(checks) == []
    # Guard against a vacuous pass: there are calls, input tokens, and several steps compared.
    assert result.n_calls > N_BLOBS and result.cost["input_tokens"] > 0
    assert len(rc.steps) == record.n_summary_steps >= 3
    assert sum(s.counts["input_tokens"] for s in rc.steps) == result.cost["input_tokens"]
    assert sum(s.output_model for s in rc.steps) == result.cost["output_tokens"]
    assert _by_name(checks)["per step: logged `n_calls` = clusters sent"].n == record.n_summary_steps


def test_memo_hits_are_not_calls(run):
    # Well-separated blobs: after step 0 the partition is stable, so later steps are all memo hits.
    record, texts, X, result = run(separated=True, max_iter=40)
    rc = _check(record, texts, X)
    calls = [s.n_sent for s in rc.steps]
    assert calls[0] == N_BLOBS and calls[-1] < N_BLOBS
    assert [s.logged["n_calls"] for s in rc.steps] == calls
    assert _failed(run_checks([rc])) == []


def test_truncated_prompts_are_rebuilt_exactly(run):
    # ~50 members per cluster do not fit 40 words, so every sent prompt is trimmed to a seeded subset.
    tight = PromptSettings(**{**SETTINGS.__dict__, "max_input_tokens": 40})
    record, texts, X, result = run(settings=tight)
    assert result.cost["n_truncated_clusters"] > 0 and result.cost["n_dropped_texts"] > 0

    rc = _check(record, texts, X, settings=tight)
    checks = run_checks([rc])
    assert _failed(checks) == []
    assert sum(s.counts["n_dropped_texts"] for s in rc.steps) == result.cost["n_dropped_texts"]
    assert sum(s.counts["input_tokens"] for s in rc.steps) == result.cost["input_tokens"]
    # The wrong budget rebuilds different prompts and is caught.
    wrong = _check(record, texts, X, settings=PromptSettings(**{**SETTINGS.__dict__, "max_input_tokens": 60}))
    assert {"per step: logged `input_tokens` = re-tokenized prompts"} <= set(_failed(run_checks([wrong])))


def _bump(target, key, delta=1):
    """A tamper function adding `delta` to `key` of the record's trace row 1 (`target="trace"`) or metrics."""

    def tamper(record):
        row = record.trace[1] if target == "trace" else record.metrics
        row[key] += delta

    return tamper


@pytest.mark.parametrize(
    "tamper, expected_failure",
    [
        (_bump("trace", "n_calls"), "per step: logged `n_calls` = clusters sent"),
        (_bump("trace", "input_tokens", -1), "per step: logged `input_tokens` = re-tokenized prompts"),
        (_bump("trace", "n_dropped_texts", 3), "per step: logged `n_dropped_texts` = rebuilt trims"),
        (_bump("trace", "n_empty_summaries"), "per step: logged `n_empty_summaries` = blank summaries sent"),
        (_bump("metrics", "n_calls"), "per run: `n_calls` in metrics = rebuild over all steps"),
        (_bump("metrics", "summarizer_seconds"), "per run: `summarizer_seconds` in metrics = sum of trace rows"),
        (_bump("metrics", "runtime_s", -100), "per run: `summarizer_seconds` <= `runtime_s`"),
    ],
)
def test_each_miscount_is_caught(run, tamper, expected_failure):
    record, texts, X, _ = run()
    tamper(record)
    failed = _failed(run_checks([_check(record, texts, X)]))
    assert expected_failure in failed


def test_a_miscounted_trace_row_reports_the_size_of_the_discrepancy(run):
    record, texts, X, _ = run()
    record.trace[2]["n_calls"] += 4
    check = _by_name(run_checks([_check(record, texts, X)]))["per step: logged `n_calls` = clusters sent"]
    assert (check.n_failed, check.max_abs) == (1, 4)


def test_output_tokens_are_an_expectation_with_a_relative_tolerance(run):
    record, texts, X, _ = run()
    rc = _check(record, texts, X)
    record.metrics["output_tokens"] += 1  # 1 token off of ~20: inside 10%, outside 1%

    def output_failed(tol):
        return [n for n in _failed(run_checks([rc], output_tol=tol)) if "output_tokens` in metrics ~" in n]

    assert output_failed(0.01) and not output_failed(0.1)
    info = [c for c in run_checks([rc]) if c.informational]
    assert len(info) == 1 and info[0].passed


def test_a_final_step_that_stops_on_tol_is_in_the_totals_only(run):
    record, texts, X, result = run(tol=1e9)
    assert result.converged and record.n_summary_steps == 1 and result.n_calls == N_BLOBS
    assert all(row["n_calls"] == 0 for row in record.trace)

    rc = _check(record, texts, X)
    assert rc.steps[0].logged is None
    checks = _by_name(run_checks([rc]))
    assert _failed(list(checks.values())) == []
    assert checks["per step: logged `n_calls` = clusters sent"].n == 0
    assert checks["per run: `n_calls` in metrics = sum of trace rows"].note.startswith("1 record")

    record.metrics["input_tokens"] += 1  # the tol step's tokens must be what the rebuild says
    assert "per run: `input_tokens` in metrics = sum of trace rows" in _failed(run_checks([rc]))


def test_extractive_run_checks_calls_without_a_tokenizer(run):
    record, texts, X, result = run(extractive=True)
    rc = check_record(record, texts, X, NOISY_FCL)
    checks = run_checks([rc])
    assert not rc.with_tokens and _failed(checks) == []
    assert not any("input_tokens" in c.name and c.name.startswith("per step") for c in checks)
    assert sum(s.n_sent for s in rc.steps) == result.n_calls


def test_prompt_settings_follow_the_run_config_then_the_dataset_defaults():
    cfg = {"type": "hf", "model_id": "m", "max_input_tokens": 100, "max_new_tokens": 8}
    s = PromptSettings.from_run(cfg, "ds prompt", "Utterance:")
    assert (s.prompt, s.text_type) == ("ds prompt", "Utterance:")
    assert s.system_prompt == "You are a helpful assistant summarizing text clusters."
    assert (s.max_input_tokens, s.max_new_tokens, s.chat_template_kwargs) == (100, 8, {"enable_thinking": False})

    over = PromptSettings.from_run({**cfg, "prompt": "mine", "text_type": "T:", "chat_template_kwargs": {}}, "ds", "ds")
    assert (over.prompt, over.text_type, over.chat_template_kwargs) == ("mine", "T:", {})
    assert PromptSettings.from_run({"type": "hf"}, "p", "t").max_input_tokens == 4096


def test_rebuild_prompt_counts_the_rendered_prompt():
    prompt = rebuild_prompt(WordTokenizer(), SETTINGS, ["a b", "c"], fold_system=False)
    # "<system> You summarize." (3) + "<user> Summarize:\n\na b\nc" (5) + "<assistant> Sentence:" (2)
    assert (prompt.input_tokens, prompt.n_dropped, prompt.truncated) == (3 + 5 + 2, 0, False)
    assert count_tokens(WordTokenizer(), "x y z") == 3


def test_fit_and_render_trims_to_the_budget():
    texts = [f"w{i}" for i in range(30)]
    kwargs = dict(
        system_prompt="s", prompt="p", text_type="T:", chat_template_kwargs={}, fold_system=False, max_input_tokens=20
    )
    rendered, n_dropped, truncated = fit_and_render(WordTokenizer(), texts, **kwargs)
    assert n_dropped > 0 and not truncated
    assert len(rendered.split()) <= 20
    assert fit_and_render(WordTokenizer(), texts[:3], **{**kwargs, "max_input_tokens": 100})[1:] == (0, False)


def test_verify_run_and_report_on_a_logged_run(run, noisy, tmp_path, monkeypatch):
    run()
    texts, X, embed = noisy
    ds = Dataset("synth", texts, np.zeros(len(texts), dtype=int), N_BLOBS, "Summarize:", "Sentence:")
    monkeypatch.setattr("kslimeans.experiment.load_dataset", lambda name, **kw: ds)
    monkeypatch.setattr("kslimeans.cache.get_embedder", lambda name: pytest.fail("encode attempted"))
    cache_dir = tmp_path / "cache"
    cached_embeddings("synth", texts, LookupEmbedder(embed), str(cache_dir))

    run_path = next((tmp_path / "synth_run").iterdir())
    log = load_run(run_path)
    log.config["force_context_length"] = NOISY_FCL
    log.config["summarizer"] = {
        "type": "hf",
        "model_id": "fake/model",
        "system_prompt": SETTINGS.system_prompt,
        "max_input_tokens": SETTINGS.max_input_tokens,
        "max_new_tokens": SETTINGS.max_new_tokens,
        "batch_size": 4,
    }
    loaded = []
    report = verify_run(log, cache_dir, lambda model_id: loaded.append(model_id) or WordTokenizer())
    assert loaded == ["fake/model"]
    assert _failed(report.methods[0].checks) == []

    markdown = format_report([report])
    assert "FAIL" not in markdown and "PASS" in markdown
    assert "batch size 4" in markdown and "logged input" in markdown and "runtime - summarizer" in markdown

    log.record("k-test", SEED).trace[1]["n_calls"] += 1
    broken = format_report([verify_run(log, cache_dir, lambda model_id: WordTokenizer())])
    assert "FAIL" in broken


@pytest.mark.slow
def test_real_qwen3_tokenizer_trims_and_renders_the_non_thinking_prompt():
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not in the local Hugging Face cache")
    settings = PromptSettings(**{**SETTINGS.__dict__, "max_input_tokens": 200})
    texts = [f"what is the weather like in city number {i} today" for i in range(100)]
    prompt = rebuild_prompt(tokenizer, settings, texts, fold_system=False)
    assert prompt.n_dropped > 0 and 0 < prompt.input_tokens <= 200
    rendered, *_ = fit_and_render(
        tokenizer,
        texts[:2],
        system_prompt=settings.system_prompt,
        prompt=settings.prompt,
        text_type=settings.text_type,
        chat_template_kwargs=settings.chat_template_kwargs,
        fold_system=False,
        max_input_tokens=200,
    )
    assert rendered.rstrip().endswith("Sentence:") and "<think>" in rendered
