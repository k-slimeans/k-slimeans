"""The hand check of the automatic summary scores: a rule-based sample, a blinded sheet, and the agreement.

The rubric is `docs/summary_audit_rubric.md`. `scripts/make_audit_sheet.py` writes the sheet and a key;
`scripts/audit_agreement.py` compares a filled sheet with the automatic flags stored in the key.

Sampling rule (deterministic given the seed). Items are gold intents. For each intent and each method, the matched
cluster is the one holding the most of the intent's members (the max-count rule of `docs/results/prelim_bank77.md`
section 6, lowest cluster id on ties) and its recovery is the Jaccard overlap with the intent. An intent's score is
the minimum over methods, so that every method recovers a "well" intent and none recovers a "poor" one (section 6
takes the same minimum over methods). Intents with a score of at least `WELL_FROM` are well recovered, below
`POOR_BELOW` poorly, the rest mixed. `PLAN` intents are drawn per class with `random.Random(seed)` from the
candidates sorted by intent id, skipping an intent whose matched cluster for some method was already drawn
(a summary is rated once). A drawn intent yields one row per method: that method's summary, its cluster size
and up to `N_MEMBERS` of the cluster's members drawn with a generator seeded by (seed, intent, method index). The
four rows of an intent get shuffled letters A to D so the method is hidden.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

WELL_FROM = 0.5
POOR_BELOW = 0.25
PLAN = (("well", 4), ("mixed", 3), ("poor", 3))
N_MEMBERS = 10
LETTERS = "ABCDEFGH"

SHEET_COLUMNS = (
    "row_id",
    "item",
    "summary",
    "cluster_size",
    "members_shown",
    "members",
    "members_fit",
    "generic",
    "unsupported",
    "fluent",
    "notes",
)


@dataclass(frozen=True)
class MethodPartition:
    """One method's last summary step as the audit sees it."""

    method: str
    run: str
    seed: int
    step: int
    summaries: Sequence[str | None]
    labels: np.ndarray
    approx: bool = False
    """The summaries are scored against the final partition, not the one they were written from."""


def matched_cluster(labels: np.ndarray, gold: np.ndarray, intent: int) -> tuple[int, float]:
    """The cluster with the most members of `intent` (lowest id on ties) and its Jaccard overlap with the intent."""
    in_intent = gold == intent
    counts = np.bincount(labels[in_intent])
    c = int(counts.argmax())
    n = int(counts[c])
    return c, n / (int((labels == c).sum()) + int(in_intent.sum()) - n)


def recovery_class(score: float) -> str:
    return "well" if score >= WELL_FROM else "poor" if score < POOR_BELOW else "mixed"


def intent_scores(partitions: Sequence[MethodPartition], gold: np.ndarray, n_intents: int) -> dict[int, float]:
    """Per gold intent, the minimum over methods of the Jaccard overlap with its matched cluster."""
    return {
        g: min(matched_cluster(np.asarray(p.labels), gold, g)[1] for p in partitions)
        for g in range(n_intents)
        if np.any(gold == g)
    }


def sample_items(
    scores: dict[int, float], seed: int, plan=PLAN, accept: Callable[[int], bool] = lambda g: True
) -> list[tuple[int, str]]:
    """The audited intents with their class, in plan order; a class with too few candidates gives all it has.

    Per class the candidates (sorted by intent id) are put in a random order by `random.Random(seed)` and the first
    `n` that `accept` takes are kept. `accept` is called once per intent in that order, so it may keep state.
    """
    rng = random.Random(seed)
    items: list[tuple[int, str]] = []
    for cls, n in plan:
        candidates = sorted(g for g, s in scores.items() if recovery_class(s) == cls)
        taken = 0
        for g in rng.sample(candidates, len(candidates)):
            if taken < n and accept(g):
                items.append((g, cls))
                taken += 1
    return items


def build_audit(
    dataset: str,
    texts: Sequence[str],
    gold: np.ndarray,
    label_names: Sequence[str] | None,
    partitions: Sequence[MethodPartition],
    seed: int = 0,
    n_members: int = N_MEMBERS,
    plan=PLAN,
) -> tuple[list[dict], list[dict]]:
    """The blinded sheet rows and the key rows (same `row_id`s) for one dataset."""
    gold = np.asarray(gold)
    scores = intent_scores(partitions, gold, int(gold.max()) + 1)
    sheet: list[dict] = []
    key: list[dict] = []
    used: set[tuple[int, int]] = set()

    def unused_clusters(intent: int) -> bool:
        """Reject an intent whose matched cluster of some method is already audited (one row per summary)."""
        matched = {(m, matched_cluster(np.asarray(p.labels), gold, intent)[0]) for m, p in enumerate(partitions)}
        if matched & used:
            return False
        used.update(matched)
        return True

    for item_no, (intent, cls) in enumerate(sample_items(scores, seed, plan, unused_clusters), start=1):
        order = np.random.default_rng([seed, intent]).permutation(len(partitions))
        for letter, m in zip(LETTERS, order):
            p = partitions[int(m)]
            labels = np.asarray(p.labels)
            cluster, jac = matched_cluster(labels, gold, intent)
            idx = np.flatnonzero(labels == cluster)
            pick = np.random.default_rng([seed, intent, int(m)]).permutation(len(idx))[:n_members]
            shown = [texts[i] for i in sorted(idx[pick])]
            row_id = f"{dataset}-{item_no:02d}-{letter}"
            sheet.append(
                {
                    "row_id": row_id,
                    "item": f"{dataset}-{item_no:02d}",
                    "summary": p.summaries[cluster],
                    "cluster_size": len(idx),
                    "members_shown": len(shown),
                    "members": "\n".join(f"{i}. {t}" for i, t in enumerate(shown, start=1)),
                    "members_fit": "",
                    "generic": "",
                    "unsupported": "",
                    "fluent": "",
                    "notes": "",
                }
            )
            key.append(
                {
                    "row_id": row_id,
                    "dataset": dataset,
                    "item": f"{dataset}-{item_no:02d}",
                    "gold_intent": label_names[intent] if label_names else intent,
                    "recovery_class": cls,
                    "item_score": scores[intent],
                    "method": p.method,
                    "run": p.run,
                    "seed": p.seed,
                    "step": p.step,
                    "cluster": cluster,
                    "cluster_jaccard": jac,
                    "approx_partition": p.approx,
                }
            )
    return sheet, key


# ------------------------------------------------------------------ agreement

HAND_LABELS = {
    # hand label -> the automatic flag column it is compared with
    "low_coverage": "flag_low_coverage",
    "generic": "flag_generic",
    "unsupported": "flag_unsupported",
    "not_fluent": "flag_not_fluent",
}


def hand_label(row: dict, name: str) -> bool | None:
    """The hand label `name` of a filled sheet row (a mapping), `None` if its cell is empty."""

    def cell(col):
        v = row.get(col)
        return None if v is None or str(v).strip() == "" or str(v).lower() == "nan" else float(v)

    if name == "low_coverage":
        fit, shown = cell("members_fit"), cell("members_shown")
        return None if fit is None or not shown else fit / shown < 0.5
    if name == "generic":
        v = cell("generic")
        return None if v is None else v == 1
    if name == "unsupported":
        v = cell("unsupported")
        return None if v is None else v == 1
    if name == "not_fluent":
        v = cell("fluent")
        return None if v is None else v == 0
    raise KeyError(name)


def cohens_kappa(hand: Sequence[bool], auto: Sequence[bool]) -> float:
    """Cohen's kappa of two binary labellings; `NaN` when chance agreement is already 1 (or there are no pairs)."""
    n = len(hand)
    if n == 0:
        return float("nan")
    observed = sum(h == a for h, a in zip(hand, auto)) / n
    p_hand, p_auto = sum(hand) / n, sum(auto) / n
    expected = p_hand * p_auto + (1 - p_hand) * (1 - p_auto)
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


def agreement(hand: Sequence[bool], auto: Sequence[bool]) -> dict:
    """The 2 x 2 counts (hand label x automatic flag), raw agreement and kappa."""
    pairs = list(zip(hand, auto))
    tp = sum(h and a for h, a in pairs)
    fp = sum((not h) and a for h, a in pairs)
    fn = sum(h and (not a) for h, a in pairs)
    tn = sum((not h) and (not a) for h, a in pairs)
    n = len(pairs)
    return {
        "n": n,
        "hand_pos": tp + fn,
        "auto_pos": tp + fp,
        "both": tp,
        "hand_only": fn,
        "auto_only": fp,
        "neither": tn,
        "raw_agreement": (tp + tn) / n if n else float("nan"),
        "kappa": cohens_kappa([h for h, _ in pairs], [a for _, a in pairs]),
    }
