"""Checks for the `HFSummarizer` SLM backend.

Fast tests exercise the pure helpers (`build_messages`, `render_conversation`,
`supports_system_role`, `fit_texts_to_budget`) with stubbed-out tokenizers, so they
run without downloading or loading a real model. `@pytest.mark.slow` tests load the
tiny cached `HuggingFaceTB/SmolLM2-135M-Instruct` and exercise the real thing,
including a full `kLLMmeans` run. Run the slow ones with `pytest -m slow tests/test_hf.py`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from kslimeans.summarizers.base import Cluster
from kslimeans.summarizers.hf import (
    DEFAULT_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    HFSummarizer,
    build_messages,
    fit_texts_to_budget,
    render_conversation,
    supports_system_role,
)

SMOLLM2 = "HuggingFaceTB/SmolLM2-135M-Instruct"


# ----------------------------------------------------------------------------
# Fast: message construction, no tokenizer/model involved.
# ----------------------------------------------------------------------------


def test_build_messages_default_conversation_has_system_and_assistant_prefill():
    messages = build_messages("SYS", "PROMPT", "Sentence:", ["t1", "t2"])
    assert messages == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "PROMPT\n\nt1\nt2"},
        {"role": "assistant", "content": "Sentence:"},
    ]


def test_build_messages_empty_text_type_omits_assistant_turn():
    messages = build_messages("SYS", "PROMPT", "", ["t1"])
    assert messages == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "PROMPT\n\nt1"},
    ]


def test_build_messages_folds_system_into_user_when_requested():
    messages = build_messages("SYS", "PROMPT", "Sentence:", ["t1"], fold_system=True)
    assert messages == [
        {"role": "user", "content": "SYS\n\nPROMPT\n\nt1"},
        {"role": "assistant", "content": "Sentence:"},
    ]


# ----------------------------------------------------------------------------
# Fast: chat-template rendering and system-role detection, with stub tokenizers.
# ----------------------------------------------------------------------------


class _StubTokenizer:
    """Records the kwargs `render_conversation`/`supports_system_role` pass through."""

    def __init__(self, raise_on_system: bool = False):
        self.raise_on_system = raise_on_system
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, tokenize, **kwargs):
        self.calls.append(kwargs)
        if self.raise_on_system and any(m["role"] == "system" for m in messages):
            raise ValueError("this template has no system role")
        return "RENDERED"


def test_render_conversation_continues_the_prefill_when_text_type_is_set():
    tok = _StubTokenizer()
    out = render_conversation(tok, [{"role": "user", "content": "x"}], "Sentence:", {"a": 1})
    assert out == "RENDERED"
    assert tok.calls[-1] == {"continue_final_message": True, "a": 1}


def test_render_conversation_opens_a_fresh_turn_when_text_type_is_empty():
    tok = _StubTokenizer()
    render_conversation(tok, [{"role": "user", "content": "x"}], "", {"a": 1})
    assert tok.calls[-1] == {"add_generation_prompt": True, "a": 1}


def test_supports_system_role_true_for_a_permissive_template():
    assert supports_system_role(_StubTokenizer(raise_on_system=False), {}) is True


def test_supports_system_role_false_for_a_template_that_rejects_system():
    assert supports_system_role(_StubTokenizer(raise_on_system=True), {}) is False


# ----------------------------------------------------------------------------
# Fast: token-budget trimming, with fake render/count functions.
# ----------------------------------------------------------------------------


def _join(texts: list[str]) -> str:
    return " ".join(texts)


def test_fit_texts_to_budget_is_a_noop_when_already_within_budget():
    texts = ["a", "b", "c"]
    fitted, dropped, truncated = fit_texts_to_budget(texts, _join, len, max_tokens=100)
    assert fitted == texts
    assert dropped == 0
    assert truncated is False


def test_fit_texts_to_budget_drops_trailing_texts_first():
    texts = ["aaaa", "bbbb", "cccc", "dddd"]
    # " ".join(...) of the first two is 9 chars; three or more overflows a budget of 9.
    fitted, dropped, truncated = fit_texts_to_budget(texts, _join, len, max_tokens=9)
    assert fitted == ["aaaa", "bbbb"]
    assert dropped == 2
    assert truncated is False


def test_fit_texts_to_budget_keeps_at_least_one_text_and_truncates_its_characters():
    texts = ["a" * 50]
    fitted, dropped, truncated = fit_texts_to_budget(texts, _join, len, max_tokens=10)
    assert len(fitted) == 1
    assert dropped == 0
    assert truncated is True
    assert len(fitted[0]) <= 10
    assert set(fitted[0]) == {"a"}


def test_fit_texts_to_budget_drops_then_truncates_the_lone_survivor():
    texts = ["x" * 50, "y" * 5]
    fitted, dropped, truncated = fit_texts_to_budget(texts, _join, len, max_tokens=10)
    assert dropped == 1
    assert truncated is True
    assert fitted == ["x" * 6]  # 50 -> 25 -> 12 -> 6, the first length <= 10


def test_fit_texts_to_budget_binary_search_is_logarithmic_in_cluster_size():
    """Big clusters (force_context_length=0) used to drop one trailing text at a time,
    re-rendering + re-tokenizing the whole prompt on every drop -- O(n) `count_tokens`
    calls. The binary search should need only O(log n)."""
    n = 1000
    texts = ["x" * 5] * n
    # " ".join of the first k: 5*k chars + (k-1) separators = 6k - 1; exactly fits at k=500.
    max_tokens = 6 * 500 - 1

    calls = 0

    def counting_count_tokens(rendered: str) -> int:
        nonlocal calls
        calls += 1
        return len(rendered)

    fitted, dropped, truncated = fit_texts_to_budget(texts, _join, counting_count_tokens, max_tokens)

    assert len(fitted) == 500
    assert dropped == 500
    assert truncated is False
    assert calls <= 20  # log2(1000) =~ 10, plus a couple of fixed checks -- nowhere near 1000


# ----------------------------------------------------------------------------
# Slow: real cached SmolLM2-135M-Instruct.
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smollm2():
    return HFSummarizer(model_id=SMOLLM2, max_new_tokens=24)


@pytest.mark.slow
def test_rendered_prompt_ends_with_the_text_type_prefill_not_a_fresh_turn():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(SMOLLM2, padding_side="left")
    messages = build_messages(
        DEFAULT_SYSTEM_PROMPT, DEFAULT_PROMPT, "Sentence:", ["cats are great", "dogs are great too"]
    )
    rendered = render_conversation(tokenizer, messages, "Sentence:", {"enable_thinking": False})
    assert rendered.endswith("Sentence:")
    assert rendered.count("<|im_start|>assistant") == 1


@pytest.mark.slow
def test_system_folding_with_a_custom_template_that_rejects_system_messages():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(SMOLLM2, padding_side="left")
    assert supports_system_role(tokenizer, {"enable_thinking": False}) is True

    # A minimal template in the style of Gemma's, which raises on a system message.
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ raise_exception('system role not supported') }}"
        "{% endif %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    assert supports_system_role(tokenizer, {"enable_thinking": False}) is False

    messages = build_messages("SYS", "PROMPT", "Sentence:", ["t1"], fold_system=True)
    rendered = render_conversation(tokenizer, messages, "Sentence:", {"enable_thinking": False})
    assert rendered == "user: SYS\n\nPROMPT\n\nt1\nassistant: Sentence:"


@pytest.mark.slow
def test_batch_output_order_preserved_and_deterministic_across_two_calls(smollm2):
    clusters = [
        Cluster(index=0, doc_ids=(0,), texts=["My cat loves to nap in sunny windows all afternoon."]),
        Cluster(index=1, doc_ids=(1,), texts=["The bank charged an unexpected overdraft fee this month."]),
        Cluster(
            index=2,
            doc_ids=(2,),
            texts=["Heavy rain and thunderstorms are expected across the region tonight."],
        ),
    ]

    first = smollm2.summarize_batch(clusters)
    second = smollm2.summarize_batch(clusters)
    assert first == second  # greedy decoding is deterministic
    assert len(first) == 3
    assert len(set(first)) == 3  # three distinct clusters produce three distinct summaries

    reversed_result = smollm2.summarize_batch(list(reversed(clusters)))
    assert reversed_result == list(reversed(first))  # output order tracks input order


@pytest.mark.slow
def test_max_input_tokens_trimming_warns_once_and_still_produces_output():
    summarizer = HFSummarizer(model_id=SMOLLM2, max_new_tokens=16, max_input_tokens=24)
    texts = [
        "Customer service was excellent and the representative resolved my card issue quickly.",
        "I was charged a foreign transaction fee I did not expect on my last purchase overseas.",
        "The mobile app crashed twice while I was trying to transfer money between accounts.",
        "My debit card was declined at checkout even though I had sufficient funds available.",
    ]
    cluster = Cluster(index=0, doc_ids=tuple(range(len(texts))), texts=texts)

    with pytest.warns(UserWarning, match="max_input_tokens"):
        summary = summarizer.summarize(cluster)
    assert isinstance(summary, str)
    assert summary != ""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summarizer.summarize(cluster)
        assert not any("max_input_tokens" in str(w.message) for w in caught)


@pytest.mark.slow
def test_build_summarizer_hf_integration():
    from kslimeans.summarizers import build_summarizer

    summarizer = build_summarizer(
        {"type": "hf", "model_id": SMOLLM2, "max_new_tokens": 16},
        prompt="Write a single sentence that represents the following cluster concisely:",
        text_type="Sentence:",
    )
    assert isinstance(summarizer, HFSummarizer)
    assert summarizer.model_id == SMOLLM2
    assert summarizer.text_type == "Sentence:"

    cluster = Cluster(
        index=0, doc_ids=(0, 1), texts=["I love hiking in the mountains.", "Camping trips are the best."]
    )
    summary = summarizer.summarize(cluster)
    assert isinstance(summary, str)
    assert summary != ""


@pytest.mark.slow
def test_kllmmeans_end_to_end_bank77_smollm2():
    from kslimeans import kLLMmeans, load_dataset

    dataset = load_dataset("bank77")
    texts = dataset.texts[:400]

    result = kLLMmeans(
        texts,
        10,
        model=SMOLLM2,
        prompt=dataset.prompt,
        text_type=dataset.text_type,
        max_llm_iter=2,
        max_iter=30,
        emb_type="distilbert",
        random_state=0,
    )

    assert len(result.summaries) == 10
    counts = np.bincount(result.assignments, minlength=10)
    for cluster_id, summary in enumerate(result.summaries):
        if counts[cluster_id] > 0:
            assert summary != ""
        else:
            assert summary == ""
