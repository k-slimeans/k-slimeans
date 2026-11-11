"""Offline replay of a logged run: every Lloyd iteration inside the blocks the logs do not record.

`summary_kmeans` logs one row per KMeans fit, so the objective, centers and assignments *inside* a block of Lloyd
iterations (and the centers before a replacement) are not in the logs. The summaries are logged, though, and each
block is a deterministic function of its start centers, so it can be recomputed without any language model:

- step 0 is the same `KMeans(init="k-means++", max_iter=first_iters, random_state=seed)` fit core makes;
- summary step `j` starts from `embed_summaries(embedder, summaries_evolution[j])` and runs a block of
  `max_iter` Lloyd iterations (`last_max_iter` of its trace row).

Method. `LloydReplayer.run_block` advances sklearn's own Lloyd step (`_kmeans_single_lloyd` with `max_iter=1`) one
iteration at a time, in the centered space `KMeans.fit` works in (data and start centers minus the data mean, the
mean added back to every reported center, as `fit` does at the end). Because the data are not re-centered between
iterations, each iteration equals the corresponding iteration of one `max_iter=T` fit. Chaining public
`KMeans(max_iter=1)` fits would not: each fit adds the mean back and the next subtracts it again, which changes
float32 bits. `KMeans(init=start, max_iter=t)` for `t = 1..T` is exact too but costs `T(T+1)/2` iterations. The
two stops `fit` makes are reproduced from the recorded states, so they act exactly as in a single fit:

- strict convergence: the E-step labels of iteration `t` equal those of iteration `t - 1` (never at `t = 1`);
  `fit` then returns the labels of `t - 1` and the centers of `t`;
- tol: the squared center shift `sum_c ||c_t - c_{t-1}||^2` is at most `tol * mean(var(X, axis=0))`.

Empty-cluster relocation happens inside sklearn's Lloyd step, so it behaves as in the fit. Uses the private
`sklearn.cluster._kmeans._kmeans_single_lloyd` and `_tolerance` and `_k_means_lloyd.lloyd_iter_chunked_dense`; `tests/test_replay.py` compares
the result with real `KMeans` fits, so a sklearn change that alters them fails the tests.

Conventions. In a `Block`, iteration `t` holds the centers after `t` updates and the labels of the nearest-center
assignment to them (what `fit` returns when it stops there), so `iterations[0]` is the start state: the summary
centers with every point assigned to the nearest one. `Iteration.objective` is `assigned_objective` with the
nearest-center labels: core's `_objective` (the trace's `summary_inertia` at `t = 0`) up to float ties, and sklearn's
float32 `inertia_` to about 1e-7 relative at the last `t`.

Reuse for counterfactual replacements (#47): `LloydReplayer.run_block(start_centers, max_iter)` replays a block from
any start centers, `replay_step` wraps it with the pre-replacement state and the no-replacement counterfactual, and
`replay_run(..., swap=...)` lets a caller change the start centers of a step (e.g. keep the k-means center of some
clusters, or put in another model's summary embedding) before the block runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster._k_means_lloyd import lloyd_iter_chunked_dense
from sklearn.cluster._kmeans import _kmeans_single_lloyd, _tolerance
from sklearn.utils import check_array
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

from .core import _objective
from .embeddings import Embedder
from .runlog import RunRecord, embed_summaries

KMEANS_TOL = 1e-4
"""sklearn's default `KMeans.tol`, which core's Lloyd blocks use (not the summary-shift `tol` of the config)."""


@dataclass
class Iteration:
    """The state after `t` Lloyd iterations of a block (`t = 0`: the start centers)."""

    t: int
    labels: np.ndarray
    """Nearest-center assignment to `centers`."""
    centers: np.ndarray
    objective: float
    n_reassigned: int
    """Points whose label differs from iteration `t - 1` (0 at `t = 0`)."""
    center_shift: float
    """`sqrt(sum_c ||c_t - c_{t-1}||^2)` (0 at `t = 0`)."""


@dataclass
class Block:
    """One block of Lloyd iterations replayed from given start centers."""

    iterations: list[Iteration]
    """`t = 0 .. n_iter`."""
    stop: str
    """Why it ended: `"max_iter"`, `"tol"` or `"strict"` (strict label convergence); `"fixed"` for `stop=False`."""
    final_labels: np.ndarray
    """The labels `KMeans.fit` returns (those of `t - 1` after a strict-convergence stop, else of `t`)."""

    @property
    def n_iter(self) -> int:
        """Iterations run, as sklearn's `n_iter_`."""
        return len(self.iterations) - 1

    @property
    def start(self) -> Iteration:
        return self.iterations[0]

    @property
    def end(self) -> Iteration:
        return self.iterations[-1]

    @property
    def final_centers(self) -> np.ndarray:
        return self.end.centers

    @property
    def objective_by_iteration(self) -> np.ndarray:
        return np.array([it.objective for it in self.iterations])


@dataclass
class State:
    """A clustering after a fit: its partition and centers."""

    labels: np.ndarray
    centers: np.ndarray
    objective: float


def state_of(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> State:
    """`State` with the objective of `centers` on `X` (core's `_objective`)."""
    return State(labels=np.asarray(labels), centers=np.asarray(centers), objective=_objective(X, centers))


class LloydReplayer:
    """sklearn's Lloyd iteration on `X`, advanced one step at a time (see the module docstring)."""

    def __init__(self, X: np.ndarray, tol: float = KMEANS_TOL) -> None:
        self.X = np.asarray(X)
        Xc = check_array(self.X, dtype=[np.float64, np.float32], order="C", copy=True)
        self.dtype = Xc.dtype
        self._tol = _tolerance(Xc, tol)
        self._x_mean = Xc.mean(axis=0)
        Xc -= self._x_mean
        self._Xc = Xc
        self._weights = np.ones(len(Xc), dtype=Xc.dtype)
        self._n_threads = _openmp_effective_n_threads()

    def _labels(self, centers_c: np.ndarray) -> np.ndarray:
        """Nearest-center labels of centered `centers_c`: sklearn's E-step alone (what `fit` reruns at the end)."""
        labels = np.full(len(self._Xc), -1, dtype=np.int32)
        lloyd_iter_chunked_dense(
            self._Xc,
            self._weights,
            centers_c,
            centers_c.copy(),
            np.zeros(len(centers_c), dtype=self.dtype),
            labels,
            np.zeros(len(centers_c), dtype=self.dtype),
            self._n_threads,
            update_centers=False,
        )
        return labels

    def _iterate(self, centers_c: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """One Lloyd iteration from centered `centers_c`: new centered centers, their labels, squared shift.

        With `max_iter=1` sklearn's single run ends with the E-step rerun, so its labels are those of the new centers.
        """
        labels, _, new_c, _ = _kmeans_single_lloyd(
            self._Xc, self._weights, centers_c.copy(), max_iter=1, tol=self._tol, n_threads=self._n_threads
        )
        shift = np.linalg.norm(new_c - centers_c, axis=1)
        return new_c, labels, float((shift**2).sum())

    def run_block(self, start_centers: np.ndarray, max_iter: int, *, stop: bool = True) -> Block:
        """Replay `KMeans(init=start_centers, max_iter=max_iter)` one iteration at a time.

        With `stop=False` the tol and strict-convergence stops are ignored and exactly `max_iter` iterations
        run (the counterfactual's fixed-length continuation).
        """
        start = check_array(start_centers, dtype=self.dtype, copy=True, order="C")
        centers_c = start - self._x_mean
        labels = self._labels(centers_c)
        its = [Iteration(0, labels, start, assigned_objective(self.X, start, labels), 0, 0.0)]
        reason = "max_iter" if stop else "fixed"
        final_labels = labels
        for t in range(1, max_iter + 1):
            prev_c = centers_c
            centers_c, labels, shift_sq = self._iterate(prev_c)
            centers = centers_c + self._x_mean
            prev = its[-1]
            its.append(
                Iteration(
                    t,
                    labels,
                    centers,
                    assigned_objective(self.X, centers, labels),
                    int(np.count_nonzero(labels != prev.labels)),
                    float(np.sqrt(shift_sq)),
                )
            )
            final_labels = labels
            if not stop:
                continue
            if t >= 2 and np.array_equal(prev.labels, its[-3].labels):
                reason, final_labels = "strict", prev.labels
                break
            if shift_sq <= self._tol:
                reason = "tol"
                break
        return Block(iterations=its, stop=reason, final_labels=final_labels)


def fit_initial(X: np.ndarray, n_clusters: int, max_iter: int, seed: int | None, init="k-means++") -> tuple[State, int]:
    """Step 0 as core fits it: `KMeans(init, max_iter, random_state=seed)`. Returns the state and `n_iter_`."""
    km = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, random_state=seed)
    labels = km.fit_predict(X)
    return state_of(X, labels, km.cluster_centers_), int(km.n_iter_)


def assigned_objective(X: np.ndarray, centers: np.ndarray, labels: np.ndarray, chunk_rows: int = 65536) -> float:
    """`sum_i ||x_i - centers[labels_i]||^2` in float64: the objective of `centers` for the given assignment.

    With the nearest-center labels it equals core's `_objective` (which re-derives them with an argmin pass) up to
    float ties. With old labels it is the objective of new centers before points are re-assigned.
    """
    centers64 = centers.astype(np.float64)
    total = 0.0
    for lo in range(0, len(X), chunk_rows):
        diff = X[lo : lo + chunk_rows].astype(np.float64) - centers64[labels[lo : lo + chunk_rows]]
        total += float(np.sum(diff**2))
    return total


@dataclass
class StepReplay:
    """One summary step: the pre-replacement state, the replacement and the block that follows."""

    step: int
    """Summary step `j` (0-based, entry `j` of `summaries_evolution`)."""
    pre: State
    start_centers: np.ndarray
    """The centers the block started from (the summary embeddings, unless `swap` changed rows)."""
    old_assignment_objective: float
    """Objective of `start_centers` with the pre-replacement labels."""
    block: Block
    counterfactual: Block | None
    """Plain Lloyd from `pre.centers` for `block.n_iter` iterations (no stops), or `None`."""

    @property
    def post(self) -> State:
        """The state the next step's replacement starts from."""
        return State(self.block.final_labels, self.block.final_centers, self.block.end.objective)


def replay_step(
    replayer: LloydReplayer,
    pre: State,
    start_centers: np.ndarray,
    max_iter: int,
    *,
    step: int = 0,
    counterfactual: bool = True,
) -> StepReplay:
    """Replay one replacement from `pre` with arbitrary `start_centers` (the hook for counterfactual swaps).

    `start_centers` is a `(k, d)` array, usually the summary embeddings with some rows swapped. The block
    is run with `max_iter` iterations (sklearn's stops apply); the counterfactual continues `pre.centers`
    for as many iterations as the block ran.
    """
    block = replayer.run_block(start_centers, max_iter)
    cf = replayer.run_block(pre.centers, block.n_iter, stop=False) if counterfactual else None
    return StepReplay(
        step=step,
        pre=pre,
        start_centers=block.start.centers,
        old_assignment_objective=assigned_objective(replayer.X, block.start.centers, pre.labels),
        block=block,
        counterfactual=cf,
    )


def summary_embeddings(record: RunRecord, embedder: Embedder) -> list[np.ndarray]:
    """The start centers of every summary step of `record`: `embed_summaries` of each `summaries_evolution[j]`."""
    return [embed_summaries(embedder, summaries) for summaries in record.summaries_evolution]


@dataclass
class RunReplay:
    """A replayed (method, seed): the step-0 fit and one `StepReplay` per summary step that had a block."""

    initial: State
    initial_n_iter: int
    steps: list[StepReplay] = field(default_factory=list)


def replay_run(
    record: RunRecord,
    X: np.ndarray,
    embeddings: Sequence[np.ndarray],
    *,
    replayer: LloydReplayer | None = None,
    counterfactual: bool = True,
    swap: Callable[[int, State, np.ndarray], np.ndarray] | None = None,
    initial: tuple[State, int] | None = None,
) -> RunReplay:
    """Replay every Lloyd block of `record` (a fixed-schedule run that logged its assignments and trace).

    `embeddings[j]` are step `j`'s start centers (`summary_embeddings`). `swap(j, pre, embeddings[j])`, if given,
    returns the start centers to use instead (`pre` is the state the replacement starts from; the step-0
    partition is shared by all methods of a seed, so rows correspond across methods at step 1). Each block
    continues from the replayed previous block, so a swap at step `j` carries into later steps.
    `initial` is a precomputed `fit_initial` result (state, `n_iter_`) to reuse: step 0 is a real k-means++ fit,
    the slowest part of a replay, and all methods of a seed share it. It is still verified against the log.
    A final summary step that stopped on `tol` (no fit after it) has no block and ends the replay.
    """
    if record.summary_source_steps is None or not record.trace:
        raise ValueError(f"({record.method!r}, seed {record.seed}) lacks summary_source_steps or a trace")
    replayer = replayer or LloydReplayer(X)
    k = len(record.summaries_evolution[0])
    by_step = {row["step"]: row for row in record.trace}
    initial, n_iter0 = initial or fit_initial(X, k, by_step[0]["last_max_iter"], record.seed)
    out = RunReplay(initial=initial, initial_n_iter=n_iter0)
    pre = initial
    for j, source in enumerate(record.summary_source_steps):
        if source + 1 not in by_step:
            break
        start = embeddings[j] if swap is None else swap(j, pre, embeddings[j])
        step = replay_step(
            replayer, pre, start, by_step[source + 1]["last_max_iter"], step=j, counterfactual=counterfactual
        )
        out.steps.append(step)
        pre = step.post
    return out


@dataclass
class FitCheck:
    """Replay against the logs for one KMeans fit (`fit` 0 is the initial fit)."""

    fit: int
    n_label_mismatch: int
    """Points whose replayed label differs from `assignments_evolution[fit]`."""
    n_iter: int
    logged_n_iter: int
    rel_err_summary_inertia: float | None
    """`|replayed - logged| / logged` of the objective at the start centers (`None` for fit 0)."""
    rel_err_inertia: float
    """The same for the objective after the fit (sklearn's `inertia_` float32 vs core's `_objective`)."""

    @property
    def exact(self) -> bool:
        return self.n_label_mismatch == 0 and self.n_iter == self.logged_n_iter


def _rel_err(replayed: float, logged: float) -> float:
    return abs(replayed - logged) / abs(logged) if logged else abs(replayed - logged)


def verify_replay(record: RunRecord, replay: RunReplay) -> list[FitCheck]:
    """Compare a replay with the logged labels, `last_n_iter`, `summary_inertia` and `inertia` of every fit."""
    by_step = {row["step"]: row for row in record.trace}
    ae = record.assignments_evolution
    checks = [
        FitCheck(
            0,
            int(np.count_nonzero(replay.initial.labels != ae[0])),
            replay.initial_n_iter,
            by_step[0]["last_n_iter"],
            None,
            _rel_err(replay.initial.objective, by_step[0]["inertia"]),
        )
    ]
    for step in replay.steps:
        fit = record.summary_source_steps[step.step] + 1
        row = by_step[fit]
        checks.append(
            FitCheck(
                fit,
                int(np.count_nonzero(step.block.final_labels != ae[fit])),
                step.block.n_iter,
                row["last_n_iter"],
                _rel_err(step.block.start.objective, row["summary_inertia"]),
                _rel_err(step.block.end.objective, row["inertia"]),
            )
        )
    return checks


class SummaryEmbeddings:
    """Embeddings of logged summaries, keyed by (run name, method, seed, summary step), plus how they were made.

    Written by `scripts/embed_logged_summaries.py`, so the replay and the diagnostic never load an encoder: the
    start centers of a replay are exactly the file's rows. `meta` records the device and library versions.
    """

    def __init__(self, arrays: dict[tuple[str, str, int, int], np.ndarray], meta: dict) -> None:
        self.arrays = arrays
        self.meta = meta

    def steps(self, run: str, method: str, seed: int) -> list[np.ndarray]:
        """The start centers of every summary step of (run, method, seed), in step order."""
        out = []
        while (run, method, seed, len(out)) in self.arrays:
            out.append(self.arrays[(run, method, seed, len(out))])
        if not out:
            raise KeyError(f"no embeddings for run {run!r}, method {method!r}, seed {seed}")
        return out

    def save(self, path) -> None:
        payload = {f"{r}|{m}|{s}|{j}": a for (r, m, s, j), a in self.arrays.items()}
        np.savez_compressed(path, __meta__=np.array(json.dumps(self.meta)), **payload)

    @classmethod
    def load(cls, path) -> "SummaryEmbeddings":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["__meta__"]))
            arrays = {}
            for key in data.files:
                if key != "__meta__":
                    run, method, seed, step = key.rsplit("|", 3)
                    arrays[(run, method, int(seed), int(step))] = data[key]
        return cls(arrays, meta)
