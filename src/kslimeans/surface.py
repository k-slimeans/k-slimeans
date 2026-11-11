"""Surface checks on logged summaries: length, shape and wording counts that need no model.

These reproduce the table in `docs/results/clinc150_summarizers.md` section 2.4 from a run's
`summaries_evolution`. "All steps" pools every summary step, "final" is the last step (identical to the
logged `summaries`). A `None` summary (a cluster a masked schedule had not summarized) is skipped everywhere.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

_MENTION = re.compile(r"cluster|summar", re.IGNORECASE)


def mentions_cluster(text: str | None) -> bool:
    """Whether `text` talks about the cluster instead of standing in for it ("cluster" or "summar", any case)."""
    return text is not None and _MENTION.search(text) is not None


def duplicate_groups(summaries: Sequence[str | None]) -> list[list[int]]:
    """Cluster indices that share exactly the same non-empty summary, one list per text with 2+ clusters.

    Groups are ordered by their first index and indices ascend within a group.
    """
    by_text: dict[str, list[int]] = defaultdict(list)
    for i, text in enumerate(summaries):
        if text:
            by_text[text].append(i)
    return [group for group in by_text.values() if len(group) > 1]


@dataclass(frozen=True)
class SurfaceCounts:
    n: int
    """Summaries over all steps and seeds."""
    mean_words: float
    """Mean of `len(text.split())` over all of them."""
    ends_with_question: int
    mentions: int
    starts_with_quote: int
    """Summaries whose text, left-stripped, starts with a double quote."""
    think: int
    """Summaries containing `<think>`."""
    final_n: int
    final_mentions: int
    final_shared_min: int
    """Over seeds, the fewest final-step clusters sharing their summary with another cluster."""
    final_shared_max: int
    final_largest_group: int
    """Largest set of final-step clusters with an identical summary in any one seed (0 if none repeat)."""


def surface_counts(runs: Iterable[Sequence[Sequence[str | None]]]) -> SurfaceCounts:
    """Counts over `runs`, one `summaries_evolution` (steps x clusters) per seed."""
    runs = list(runs)
    texts = [t for evolution in runs for step in evolution for t in step if t is not None]
    final = [[t for t in evolution[-1]] for evolution in runs if len(evolution)]
    final_texts = [t for step in final for t in step if t is not None]
    shared = [sum(len(g) for g in duplicate_groups(step)) for step in final]
    return SurfaceCounts(
        n=len(texts),
        mean_words=sum(len(t.split()) for t in texts) / len(texts) if texts else 0.0,
        ends_with_question=sum(t.rstrip().endswith("?") for t in texts),
        mentions=sum(mentions_cluster(t) for t in texts),
        starts_with_quote=sum(t.lstrip().startswith('"') for t in texts),
        think=sum("<think>" in t for t in texts),
        final_n=len(final_texts),
        final_mentions=sum(mentions_cluster(t) for t in final_texts),
        final_shared_min=min(shared, default=0),
        final_shared_max=max(shared, default=0),
        final_largest_group=max((len(g) for step in final for g in duplicate_groups(step)), default=0),
    )
