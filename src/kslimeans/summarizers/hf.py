"""Local small-language-model (SLM) summarizer backend, via Hugging Face `transformers`.

`HFSummarizer` replaces upstream's hosted-LLM calls (`summaryCentroids.llm_api`) with a
local `AutoModelForCausalLM`: same {system, user, assistant-prefill} conversation, rendered
with the model's own chat template and generated on-device instead of over an API.

Swap models via config (`model_id`) -- nothing else here is model-specific. VRAM rule of
thumb: fp16/bf16 weights cost ~2 bytes/param (a 0.5B model is ~1GB), plus activations and
KV-cache that grow with batch size and sequence length -- leave headroom on top of that.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .base import Cluster, Summarizer

DEFAULT_PROMPT = "Write a single sentence that represents the following cluster concisely:"
DEFAULT_TEXT_TYPE = "Sentence:"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant summarizing text clusters."

# Seed of the permutation `fit_texts_to_budget` uses to pick which members survive.
TRUNCATION_SEED = 0

_TORCH_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def build_messages(
    system_prompt: str, prompt: str, text_type: str, texts: list[str], fold_system: bool = False
) -> list[dict[str, str]]:
    """Build the conversation upstream sent to hosted LLMs (see `llm_api`/`summarize_cluster`).

    `text_type` (e.g. "Sentence:") becomes an assistant turn the model continues via prefill
    rather than opens itself; empty `text_type` drops that turn. `fold_system=True` merges the
    system prompt into the user turn instead, for templates that reject a system role --
    see `supports_system_role`.
    """
    user_content = prompt + "\n\n" + "\n".join(texts)
    if fold_system:
        messages = [{"role": "user", "content": system_prompt + "\n\n" + user_content}]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    if text_type:
        messages.append({"role": "assistant", "content": text_type})
    return messages


def render_conversation(
    tokenizer, messages: list[dict[str, str]], text_type: str, chat_template_kwargs: dict
) -> str:
    """Render `messages` with `tokenizer`'s chat template.

    Non-empty `text_type`: continue the assistant prefill (`continue_final_message=True`).
    Empty `text_type`: open a fresh assistant turn (`add_generation_prompt=True`) instead.
    """
    if text_type:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=True, **chat_template_kwargs
        )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
    )


def supports_system_role(tokenizer, chat_template_kwargs: dict) -> bool:
    """Probe whether `tokenizer`'s chat template accepts a system message (most do; Gemma's
    notably doesn't). Called once at `HFSummarizer.__init__` time, not per call.
    """
    probe = [{"role": "system", "content": "probe"}, {"role": "user", "content": "probe"}]
    try:
        tokenizer.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
        return True
    except Exception:
        return False


def fit_texts_to_budget(
    texts: list[str],
    render: Callable[[list[str]], str],
    count_tokens: Callable[[str], int],
    max_tokens: int,
    seed: int = TRUNCATION_SEED,
) -> tuple[list[str], int, bool]:
    """Trim `texts` until `render(texts)` fits `max_tokens`.

    If everything already fits, `texts` is returned unchanged, in the same order. Otherwise a
    fixed seeded permutation of the members is drawn and the largest fitting prefix of it is
    kept, rendered in the members' original relative order. If not even one text fits, one
    survives and its characters are halved until it does.

    A random subset rather than a leading prefix because member order can correlate with
    labels: CLINC150 is stored grouped by intent, so with `force_context_length: 0`
    (members in ascending dataset index) a leading prefix would drop whole trailing intents
    from a big mixed cluster instead of a representative sample (#28). The same rule applies
    whatever order members arrive in, including the k-means++-trimmed path.

    The kept sets `order[:k]` are nested in `k`, so `count_tokens(render(...))` is
    non-decreasing in `k` and the largest fitting `k` is found by binary search -- O(log n)
    `render`/`count_tokens` calls, not one per dropped text.
    Returns `(fitted_texts, n_dropped, truncated)`.
    """
    texts = list(texts)
    n = len(texts)

    if n == 0 or count_tokens(render(texts)) <= max_tokens:
        return texts, 0, False

    order = np.random.default_rng(seed).permutation(n)

    def subset(k: int) -> list[str]:
        return [texts[i] for i in sorted(order[:k])]

    # Largest k in [1, n-1] whose render fits; if none do (not even k=1), fall back
    # to 1 and let the truncation loop below shrink that lone survivor instead.
    kept = 1
    lo, hi = 1, n - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if count_tokens(render(subset(mid))) <= max_tokens:
            kept = mid
            lo = mid + 1
        else:
            hi = mid - 1

    fitted = subset(kept)

    truncated = False
    while len(fitted) == 1 and len(fitted[0]) > 1 and count_tokens(render(fitted)) > max_tokens:
        fitted[0] = fitted[0][: len(fitted[0]) // 2]
        truncated = True

    return fitted, n - kept, truncated


class HFSummarizer(Summarizer):
    """Summarizes clusters with a local Hugging Face instruct/chat model.

    See the module docstring for the VRAM rule of thumb. Constructor kwargs are kept to
    primitive types (str/int/float/dict) since they come straight from YAML configs via
    `build_summarizer`.
    """

    needs_sentences = False

    def __init__(
        self,
        model_id: str,
        prompt: str | None = None,
        text_type: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 128,
        batch_size: int = 16,
        max_input_tokens: int = 4096,
        dtype: str = "auto",
        device: str = "auto",
        chat_template_kwargs: dict | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id = model_id
        self.prompt = DEFAULT_PROMPT if prompt is None else prompt
        self.text_type = DEFAULT_TEXT_TYPE if text_type is None else text_type
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        self.chat_template_kwargs = (
            {"enable_thinking": False} if chat_template_kwargs is None else chat_template_kwargs
        )
        self.device = device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, padding_side="left", trust_remote_code=trust_remote_code
        )
        if self.tokenizer.chat_template is None:
            raise ValueError(
                f"'{model_id}' has no chat template; HFSummarizer needs an instruct/chat model "
                "(e.g. a '-Instruct' or '-chat' checkpoint), not a base language model."
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if dtype == "auto":
            torch_dtype = "auto"
        elif dtype in _TORCH_DTYPES:
            torch_dtype = _TORCH_DTYPES[dtype]
        else:
            raise ValueError(f"Unknown dtype '{dtype}', expected 'auto' or one of {sorted(_TORCH_DTYPES)}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch_dtype, trust_remote_code=trust_remote_code
        ).to(self.device)
        self.model.eval()

        # Many checkpoints ship sampling defaults (temperature/top_p/top_k) in their
        # generation_config.json; clear them once so our greedy `generate()` calls below
        # don't trip "generation flags are not valid" warnings every time they're overridden.
        clean_generation_config = GenerationConfig.from_dict(self.model.generation_config.to_dict())
        clean_generation_config.do_sample = False
        clean_generation_config.temperature = None
        clean_generation_config.top_p = None
        clean_generation_config.top_k = None
        self.model.generation_config = clean_generation_config

        self._fold_system = not supports_system_role(self.tokenizer, self.chat_template_kwargs)
        self._warned_trim = False

    def _render(self, texts: list[str]) -> str:
        """Render one cluster's texts, trimming to `max_input_tokens` if needed."""

        def render(candidate_texts: list[str]) -> str:
            messages = build_messages(
                self.system_prompt, self.prompt, self.text_type, candidate_texts, self._fold_system
            )
            return render_conversation(self.tokenizer, messages, self.text_type, self.chat_template_kwargs)

        def count_tokens(rendered: str) -> int:
            return len(self.tokenizer(rendered, add_special_tokens=False)["input_ids"])

        fitted, n_dropped, truncated = fit_texts_to_budget(texts, render, count_tokens, self.max_input_tokens)
        if (n_dropped or truncated) and not self._warned_trim:
            self._warned_trim = True
            detail = f"dropped {n_dropped} of {len(texts)} text(s)"
            if truncated:
                detail += " and truncated the remaining one's characters"
            warnings.warn(
                f"HFSummarizer({self.model_id!r}): a cluster's rendered prompt exceeded "
                f"max_input_tokens={self.max_input_tokens}; {detail}. "
                "(only warned once for this summarizer instance)",
                stacklevel=2,
            )
        return render(fitted)

    def summarize_batch(self, clusters: list[Cluster]) -> list[str]:
        outputs: list[str] = [""] * len(clusters)
        for start in range(0, len(clusters), self.batch_size):
            chunk = clusters[start : start + self.batch_size]
            rendered = [self._render(cluster.texts) for cluster in chunk]
            enc = self.tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = enc.to(self.device)

            with torch.inference_mode():
                generated = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            new_tokens = generated[:, enc["input_ids"].shape[1] :]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for i, text in enumerate(decoded):
                outputs[start + i] = text.strip()
        return outputs

    def summarize(self, cluster: Cluster) -> str:
        return self.summarize_batch([cluster])[0]
