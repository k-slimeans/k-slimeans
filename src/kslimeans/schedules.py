"""Schedules controlling *when* summary-centroid updates happen.

`summary_kmeans` (see `core.py`) alternates between Lloyd iterations of
k-means and "summarize the current clusters, re-embed the summaries, and
use them as the next centroids" steps. A `SummarySchedule` decides, after
each k-means fit, whether to summarize again, run more plain Lloyd
iterations, or stop.

`FixedSchedule` reproduces the schedule hard-coded into the original
`kSummariesmeans`: summarize every step, with the total k-means iteration
budget split evenly across `n_summaries + 1` blocks (fewer if the first
and/or last block is skipped).

This is the extension point for "scheduled" variants that change *when*
summarization happens rather than *how* a cluster is summarized (that is
`Summarizer`'s job, in `summarizers/`). A schedule is deliberately
stateless: `next_step` may only look at the `State` snapshot handed to it,
not at any instance attribute set by a previous call. For example, a
schedule could summarize only every third step, or only once
`state.frac_changed` drops below some threshold, by returning
`Step(summarize=False, kmeans_iters=...)` on the steps where it wants a
plain k-means continuation instead of a summarize step.

One caveat for anyone building such a schedule: restarting `KMeans` with
`init=km.cluster_centers_` is *not* bit-identical to continuing Lloyd's
algorithm past its current iteration count, because `sklearn`'s `KMeans`
mutates/copies the data it is fit on internally (e.g. it may recenter or
cast it) and re-runs its own convergence bookkeeping from iteration 0.
Two schedules that "should" produce the same total number of Lloyd steps
can therefore diverge in the last few bits of the centroids.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class State:
    """Snapshot of the loop state after each KMeans fit.

    Schedules decide only from this snapshot -- nothing else is passed to
    `next_step`, so a schedule cannot smuggle in hidden state.
    """

    step: int  # number of schedule steps taken so far (0 before the first next_step call)
    n_summaries: int  # summary-centroid updates actually applied so far
    lloyd_iters_used: int  # running sum of KMeans.n_iter_ over every fit so far
    labels: np.ndarray
    prev_labels: np.ndarray | None
    frac_changed: float | None  # fraction of points whose label changed vs prev_labels
    centers: np.ndarray
    inertia: float
    prev_inertia: float | None
    last_n_iter: int  # KMeans.n_iter_ of the most recent fit
    last_max_iter: int  # max_iter that was requested for the most recent fit
    summary_shift: float | None  # ||ref - summary_embeddings||-sum from the most recent summarize step


@dataclass(frozen=True)
class Step:
    """One instruction from a schedule: what to do next, and for how long."""

    summarize: bool
    kmeans_iters: int


class SummarySchedule(ABC):
    """Decides when to summarize clusters vs. keep running plain Lloyd iterations."""

    @abstractmethod
    def first_iters(self) -> int:
        """Lloyd iterations to run for the very first KMeans fit (before any summarizing)."""

    @abstractmethod
    def next_step(self, state: State) -> Step | None:
        """Return the next `Step`, or `None` to stop the loop."""


class FixedSchedule(SummarySchedule):
    """The paper's schedule: summarize every step, `n_summaries` times total.

    The `max_iter` Lloyd-iteration budget is split into `d` equal blocks,
    where `d` is the number of "summarize" steps plus one for the initial
    fit, minus one for each of `initial_iter`/`final_iter` that is turned
    off (those replace a full block with a single Lloyd iteration).
    """

    def __init__(
        self,
        n_summaries: int = 5,
        max_iter: int = 100,
        initial_iter: bool = True,
        final_iter: bool = True,
    ) -> None:
        d = n_summaries + 1 - (not initial_iter) - (not final_iter)
        if d <= 0:
            raise ValueError(
                f"n_summaries={n_summaries} with initial_iter={initial_iter}, "
                f"final_iter={final_iter} leaves no iteration blocks (d={d})"
            )
        block = int(max_iter / d)  # float division then truncate, matching upstream
        if block < 1:
            raise ValueError(
                f"max_iter={max_iter} split across d={d} blocks gives block={block} < 1"
            )
        self.n_summaries = n_summaries
        self.max_iter = max_iter
        self.initial_iter = initial_iter
        self.final_iter = final_iter
        self._block = block

    def first_iters(self) -> int:
        return self._block if self.initial_iter else 1

    def next_step(self, state: State) -> Step | None:
        if state.step >= self.n_summaries:
            return None
        is_last = state.step == self.n_summaries - 1
        iters = 1 if (is_last and not self.final_iter) else self._block
        return Step(summarize=True, kmeans_iters=iters)


def build_schedule(cfg: dict) -> SummarySchedule:
    """Build a `SummarySchedule` from a config dict (the `schedule:` block of a run config).

    `cfg["type"]` selects the schedule; the rest of `cfg` is passed to its
    constructor. Adding a new schedule? Add another branch here.
    """
    kind = cfg.get("type", "fixed")
    rest = {k: v for k, v in cfg.items() if k != "type"}
    if kind == "fixed":
        return FixedSchedule(**rest)
    else:
        raise ValueError(f"Unknown schedule type '{kind}', expected one of ('fixed',)")
