"""Summarizer registry.

`build_summarizer` maps a small config dict to a `Summarizer` instance.
It is intentionally a plain if/elif chain, not a plugin registry -- if
you're adding a new summarizer backend, add another branch here (and
`from .yours import YourSummarizer` inside it, so backends with heavy
imports don't slow down `import kslimeans` for everyone else).
"""

from __future__ import annotations

from .base import Cluster, Summarizer
from .extractive import ExtractiveSummarizer

_EXTRACTIVE_TYPES = ("lsa", "centroid", "textrank", "lexrank")
_VALID_TYPES = _EXTRACTIVE_TYPES + ("hf",)


def build_summarizer(
    cfg: dict, prompt: str | None = None, text_type: str | None = None
) -> Summarizer:
    """Build a `Summarizer` from a config dict.

    `cfg["type"]` selects the backend; the rest of `cfg` is backend-specific
    kwargs. `prompt`/`text_type` are dataset-level defaults (e.g. from a
    `Dataset`) used when `cfg` doesn't override them -- only meaningful for
    backends that take a prompt, like `hf`.
    """
    kind = cfg.get("type")
    # Adding a new backend? Add another branch here.
    if kind in _EXTRACTIVE_TYPES:
        return ExtractiveSummarizer(method=kind, top_k=cfg.get("top_k", 5))
    elif kind == "hf":
        from .hf import HFSummarizer

        rest = {k: v for k, v in cfg.items() if k not in ("type", "prompt", "text_type")}
        return HFSummarizer(
            **rest,
            prompt=cfg.get("prompt") or prompt,
            text_type=cfg.get("text_type") or text_type,
        )
    else:
        raise ValueError(f"Unknown summarizer type '{kind}', expected one of {_VALID_TYPES}")


__all__ = ["Cluster", "Summarizer", "ExtractiveSummarizer", "build_summarizer"]
