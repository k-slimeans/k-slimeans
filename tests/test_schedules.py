"""FixedSchedule must reproduce upstream kSummariesmeans's integer arithmetic exactly."""

from __future__ import annotations

import numpy as np
import pytest

from kslimeans.schedules import FixedSchedule, State


def _expected_block(n_summaries: int, max_iter: int, initial_iter: bool, final_iter: bool) -> int | None:
    """Independent re-derivation of upstream's tmp_kmeans_iterations, for cross-checking."""
    d = n_summaries + 1 - (not initial_iter) - (not final_iter)
    if d <= 0:
        return None
    block = int(max_iter / d)
    if block < 1:
        return None
    return block


def _state(step: int) -> State:
    return State(
        step=step,
        n_summaries=step,
        lloyd_iters_used=0,
        labels=np.zeros(1, dtype=int),
        prev_labels=None,
        frac_changed=None,
        centers=np.zeros((1, 1)),
        inertia=0.0,
        prev_inertia=None,
        last_n_iter=0,
        last_max_iter=0,
        summary_shift=None,
    )


GRID = [
    (n, max_iter, initial_iter, final_iter)
    for n in (1, 2, 3, 5, 8)
    for max_iter in (5, 10, 30, 100, 120)
    for initial_iter in (True, False)
    for final_iter in (True, False)
]


@pytest.mark.parametrize("n_summaries,max_iter,initial_iter,final_iter", GRID)
def test_schedule_matches_upstream_formula(n_summaries, max_iter, initial_iter, final_iter):
    expected_block = _expected_block(n_summaries, max_iter, initial_iter, final_iter)

    if expected_block is None:
        with pytest.raises(ValueError):
            FixedSchedule(n_summaries, max_iter, initial_iter, final_iter)
        return

    schedule = FixedSchedule(n_summaries, max_iter, initial_iter, final_iter)
    assert schedule.first_iters() == (expected_block if initial_iter else 1)

    for step in range(n_summaries):
        step_result = schedule.next_step(_state(step))
        assert step_result is not None
        assert step_result.summarize is True
        is_last = step == n_summaries - 1
        expected_iters = 1 if (is_last and not final_iter) else expected_block
        assert step_result.kmeans_iters == expected_iters

    assert schedule.next_step(_state(n_summaries)) is None
    assert schedule.next_step(_state(n_summaries + 3)) is None


def test_zero_summaries_and_no_initial_iter_is_invalid():
    # d = 0 + 1 - 1 - 0 = 0 <= 0
    with pytest.raises(ValueError):
        FixedSchedule(n_summaries=0, max_iter=10, initial_iter=False, final_iter=True)


def test_block_smaller_than_one_is_invalid():
    # d = 5 + 1 = 6, max_iter=3 -> block = 0
    with pytest.raises(ValueError):
        FixedSchedule(n_summaries=5, max_iter=3, initial_iter=True, final_iter=True)
