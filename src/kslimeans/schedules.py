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

A summary step may also carry a `summarize_mask` to re-summarize only some clusters (the rest keep
their k-means centers), and `State.n_calls` reports the summarizer calls spent so far, so a schedule can
be budget-aware.

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
    n_calls: int = 0  # summarizer calls so far: clusters actually sent to the summarizer (memo hits are free)
    # Per-cluster summary anchors, for schedules that re-summarize only clusters that moved. All three
    # describe the PRE-summary state of the step that last summarized cluster j (not the post-fit state,
    # which would read zero drift at the very next candidate step). Set by `summary_kmeans` (None only
    # on a hand-built State).
    last_summarized: np.ndarray | None = None  # (k,) int: `step` of the fit that followed j's last summary; -1 = never
    anchor_centers: np.ndarray | None = None  # (k, d): j's k-means center at that moment (step-0 fit if never)
    anchor_members: tuple[np.ndarray, ...] | None = None  # k int arrays: j's doc indices then (step-0 fit if never)
    cluster_rms_radius: np.ndarray | None = None  # (k,): sqrt(mean sq. distance of j's members to its center); 0 if empty


@dataclass(frozen=True)
class Step:
    """One instruction from a schedule: what to do next, and for how long."""

    summarize: bool
    kmeans_iters: int
    summarize_mask: tuple[bool, ...] | None = None
    """Which clusters (by id, length `n_clusters`) to re-summarize on a summary step. `None` means all
    of them. Masked-off clusters keep their current k-means center as the next start centroid. Any bool
    array-like is accepted and normalized to a tuple of Python bools (an ndarray field would make
    `Step == Step` raise). Only valid with `summarize=True` (checked here); `summary_kmeans` checks the
    length. An all-False mask still counts as a summary step in `State.n_summaries` but costs 0 calls, so
    budget on `State.n_calls`, not `n_summaries`."""

    def __post_init__(self) -> None:
        if self.summarize_mask is not None:
            if not self.summarize:
                raise ValueError("Step.summarize_mask requires summarize=True")
            object.__setattr__(self, "summarize_mask", tuple(bool(m) for m in self.summarize_mask))


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


def centroid_drift(
    centers: np.ndarray, anchor_centers: np.ndarray, rms_radius: np.ndarray
) -> np.ndarray:
    """Normalized centroid drift per cluster: `||centers[j] - anchor_centers[j]|| / rms_radius[j]`.

    A cluster with `rms_radius == 0` (single point, identical members, or empty) has drift 0 if its
    center did not move and `inf` if it did.
    """
    moved = np.linalg.norm(centers.astype(np.float64) - anchor_centers.astype(np.float64), axis=1)
    radius = np.asarray(rms_radius, dtype=np.float64)
    safe = np.where(radius > 0, radius, 1.0)
    return np.where(radius > 0, moved / safe, np.where(moved > 0, np.inf, 0.0))


def membership_churn(
    labels: np.ndarray, anchor_members: tuple[np.ndarray, ...], n_clusters: int
) -> np.ndarray:
    """Jaccard distance per cluster between its anchor members and its current members.

    `1 - |A_j & M_j| / |A_j | M_j|`, 0 when both sets are empty. Costs O(sum_j |A_j|), no n x k work.
    """
    sizes = np.bincount(labels, minlength=n_clusters)
    churn = np.zeros(n_clusters, dtype=np.float64)
    for j, anchor in enumerate(anchor_members):
        inter = int(np.count_nonzero(labels[anchor] == j))
        union = len(anchor) + int(sizes[j]) - inter
        if union > 0:
            churn[j] = 1.0 - inter / union
    return churn


class AdaptiveSchedule(FixedSchedule):
    """Drift/churn-triggered summaries: the candidate steps are `FixedSchedule`'s, but at each one a
    cluster is re-summarized only if it has moved enough since its last summary.

    Cluster j triggers if any of these holds (empty clusters never trigger):

    - it was never summarized;
    - `centroid_drift` > `tau_drift` (center displacement since its last summary, in units of the
      cluster's current RMS radius);
    - `membership_churn` > `tau_churn` (Jaccard distance to its member set at its last summary);
    - `s_max` is set and `step - last_summarized >= s_max`, where `step` is this candidate step's index
      (`state.step + 1`). So a cluster skips at most `s_max - 1` consecutive candidate steps, and
      `s_max=1` re-summarizes every candidate step.

    If nothing triggers the step is a plain k-means continuation; if every cluster triggers it is a full
    step (`summarize_mask=None`), otherwise a partial step over the triggered clusters. Block structure
    (`first_iters`, per-step `kmeans_iters`, number of candidate steps) is `FixedSchedule`'s.
    """

    def __init__(
        self,
        n_summaries: int = 5,
        max_iter: int = 100,
        initial_iter: bool = True,
        final_iter: bool = True,
        tau_drift: float = 0.3,
        tau_churn: float = 0.4,
        s_max: int | None = None,
    ) -> None:
        super().__init__(n_summaries, max_iter, initial_iter, final_iter)
        if tau_drift < 0 or tau_churn < 0:
            raise ValueError(f"tau_drift and tau_churn must be >= 0, got {tau_drift}, {tau_churn}")
        if s_max is not None and s_max < 1:
            raise ValueError(f"s_max must be None or >= 1, got {s_max}")
        self.tau_drift = tau_drift
        self.tau_churn = tau_churn
        self.s_max = s_max

    def next_step(self, state: State) -> Step | None:
        base = super().next_step(state)
        if base is None:
            return None
        if state.anchor_centers is None or state.anchor_members is None or state.last_summarized is None:
            raise ValueError("AdaptiveSchedule needs the summary anchors on State (set by summary_kmeans)")
        if state.cluster_rms_radius is None:
            raise ValueError("AdaptiveSchedule needs State.cluster_rms_radius (set by summary_kmeans)")

        k = len(state.centers)
        nonempty = np.bincount(state.labels, minlength=k) > 0
        trigger = (
            (state.last_summarized < 0)
            | (centroid_drift(state.centers, state.anchor_centers, state.cluster_rms_radius) > self.tau_drift)
            | (membership_churn(state.labels, state.anchor_members, k) > self.tau_churn)
        )
        if self.s_max is not None:
            trigger |= (state.step + 1 - state.last_summarized) >= self.s_max
        trigger &= nonempty

        if not trigger.any():
            return Step(summarize=False, kmeans_iters=base.kmeans_iters)
        if trigger.all():
            return base
        return Step(summarize=True, kmeans_iters=base.kmeans_iters, summarize_mask=trigger)


class RandomSubsetSchedule(FixedSchedule):
    """Control for `AdaptiveSchedule` matched on summarizer calls, with the *choice* of clusters random.

    Same candidate steps as `FixedSchedule`; at candidate step c (1-based, `state.step + 1`) it
    re-summarizes `counts[c - 1]` clusters drawn uniformly without replacement, using
    `np.random.default_rng([seed, state.step])` so the draw is stateless and reproducible. Pass the
    per-step `n_calls` of an adaptive run (`Result.trace`) so both runs spend the same calls.

    Why `changed_only` (default True): summarizing a cluster whose member set equals the one recorded at
    its last summary is a memo-cache hit and costs no call. The adaptive schedule only selects clusters
    that changed (or were never summarized), so its selections are essentially all real calls; drawing
    among *all* non-empty clusters would waste part of the budget on free hits and spend fewer calls.
    With `changed_only` the draw is over the "dirty" clusters instead: non-empty and either never
    summarized (`last_summarized < 0`) or with current members different from `anchor_members`. This needs
    the summary anchors on `State`. `changed_only=False` draws among all non-empty clusters.

    A count of 0 is a plain continuation; a count >= the number of candidate clusters selects all of
    them. The step is a full step (`summarize_mask=None`) only if that is every cluster; no candidate
    left means a plain continuation.
    """

    def __init__(
        self,
        n_summaries: int = 5,
        max_iter: int = 100,
        initial_iter: bool = True,
        final_iter: bool = True,
        counts: list[int] | None = None,
        seed: int = 0,
        changed_only: bool = True,
    ) -> None:
        super().__init__(n_summaries, max_iter, initial_iter, final_iter)
        if counts is None:
            raise ValueError("RandomSubsetSchedule needs counts (clusters to summarize at each candidate step)")
        if len(counts) != n_summaries:
            raise ValueError(f"counts has length {len(counts)}, expected n_summaries={n_summaries}")
        if any(isinstance(c, bool) or not isinstance(c, (int, np.integer)) or c < 0 for c in counts):
            raise ValueError(f"counts must be non-negative integers, got {list(counts)}")
        self.counts = tuple(int(c) for c in counts)
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
        self.seed = int(seed)
        self.changed_only = bool(changed_only)

    def _candidates(self, state: State) -> np.ndarray:
        k = len(state.centers)
        candidate = np.bincount(state.labels, minlength=k) > 0
        if self.changed_only:
            if state.anchor_members is None or state.last_summarized is None:
                raise ValueError("RandomSubsetSchedule(changed_only=True) needs the summary anchors on State")
            dirty = (state.last_summarized < 0) | (membership_churn(state.labels, state.anchor_members, k) > 0)
            candidate &= dirty
        return np.flatnonzero(candidate)

    def next_step(self, state: State) -> Step | None:
        base = super().next_step(state)
        if base is None:
            return None
        count = self.counts[state.step]
        if count == 0:
            return Step(summarize=False, kmeans_iters=base.kmeans_iters)
        k = len(state.centers)
        candidates = self._candidates(state)
        if len(candidates) == 0:
            return Step(summarize=False, kmeans_iters=base.kmeans_iters)
        if count >= len(candidates):
            chosen = candidates  # every candidate, like an adaptive step where all of them trigger
        else:
            rng = np.random.default_rng([self.seed, state.step])
            chosen = rng.choice(candidates, size=count, replace=False)
        if len(chosen) == k:
            return base
        mask = np.zeros(k, dtype=bool)
        mask[chosen] = True
        return Step(summarize=True, kmeans_iters=base.kmeans_iters, summarize_mask=mask)


def build_schedule(cfg: dict) -> SummarySchedule:
    """Build a `SummarySchedule` from a config dict (the `schedule:` block of a run config).

    `cfg["type"]` selects the schedule; the rest of `cfg` is passed to its
    constructor. Adding a new schedule? Add another branch here.
    """
    kind = cfg.get("type", "fixed")
    rest = {k: v for k, v in cfg.items() if k != "type"}
    if kind == "fixed":
        return FixedSchedule(**rest)
    elif kind == "adaptive":
        return AdaptiveSchedule(**rest)
    elif kind == "random_subset":
        return RandomSubsetSchedule(**rest)
    else:
        raise ValueError(
            f"Unknown schedule type '{kind}', expected one of ('fixed', 'adaptive', 'random_subset')"
        )
