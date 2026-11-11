"""`kslimeans.summary_audit` and the audit scripts on toy data."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kslimeans.summary_audit import (
    MethodPartition,
    agreement,
    build_audit,
    cohens_kappa,
    hand_label,
    intent_scores,
    matched_cluster,
    recovery_class,
    sample_items,
)

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 20 intents of 10 documents. Method "perfect" recovers every intent. Method "blurry" recovers intents 0-5, merges
# intents 6-11 in triples (Jaccard 1/3) and scatters the documents of intents 12-19 over 8 clusters.
N_INTENTS, PER = 20, 10
GOLD = np.repeat(np.arange(N_INTENTS), PER)
TEXTS = [f"text {i} of intent {g}" for i, g in enumerate(GOLD)]
PERFECT = GOLD.copy()
BLURRY = np.where(GOLD < 6, GOLD, np.where(GOLD < 12, 6 + 3 * ((GOLD - 6) // 3), 12 + np.arange(len(GOLD)) % 8))


def partitions() -> list[MethodPartition]:
    return [
        MethodPartition("perfect-m", "run/a", 0, 4, [f"sum {c}" for c in range(N_INTENTS)], PERFECT),
        MethodPartition("blurry-m", "run/b", 0, 4, [f"blur {c}" for c in range(20)], BLURRY),
    ]


def test_matched_cluster_is_max_count_with_lowest_id_on_ties():
    gold = np.array([0, 0, 0, 1])
    labels = np.array([2, 1, 1, 1])
    assert matched_cluster(labels, gold, 0) == (1, pytest.approx(2 / 4))  # 2 of 3, cluster of 3
    assert matched_cluster(np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1]), 0)[0] == 0  # tie -> lowest id


@pytest.mark.parametrize("score, cls", [(1.0, "well"), (0.5, "well"), (0.4999, "mixed"), (0.25, "mixed"),
                                        (0.2499, "poor"), (0.0, "poor")])  # fmt: skip
def test_recovery_class_thresholds(score, cls):
    assert recovery_class(score) == cls


def test_intent_scores_take_the_minimum_over_methods():
    scores = intent_scores(partitions(), GOLD, N_INTENTS)
    assert scores[0] == 1.0
    assert scores[6] == pytest.approx(10 / 30)  # blurry merges intents 6-8: 10 of 30
    assert scores[12] < 0.25  # scattered by the blurry method
    assert set(scores) == set(range(N_INTENTS))


def test_sample_items_is_deterministic_and_follows_the_plan():
    scores = intent_scores(partitions(), GOLD, N_INTENTS)
    plan = (("well", 2), ("mixed", 2), ("poor", 2))
    a, b = sample_items(scores, 0, plan), sample_items(scores, 0, plan)
    assert a == b
    assert [cls for _, cls in a] == ["well"] * 2 + ["mixed"] * 2 + ["poor"] * 2
    assert all(recovery_class(scores[g]) == cls for g, cls in a)
    assert len({g for g, _ in a}) == 6
    assert sample_items(scores, 1, plan) != a  # a different seed draws differently
    # Asking for more than a class holds returns what it has.
    assert len(sample_items({0: 1.0, 1: 0.9}, 0, (("well", 5),))) == 2


def test_build_audit_is_deterministic_blinded_and_keyed():
    plan = (("well", 1), ("mixed", 1), ("poor", 1))
    sheet, key = build_audit("toy", TEXTS, GOLD, [f"intent_{g}" for g in range(N_INTENTS)], partitions(), 0, 4, plan)
    assert (sheet, key) == build_audit("toy", TEXTS, GOLD, [f"intent_{g}" for g in range(N_INTENTS)], partitions(), 0, 4, plan)
    assert len(sheet) == len(key) == 6  # 3 items x 2 methods
    assert [r["row_id"] for r in sheet] == [r["row_id"] for r in key]
    for row in sheet:
        assert "perfect-m" not in str(row.values()) and "blurry-m" not in str(row.values())
        assert row["members_shown"] == 4 and len(row["members"].split("\n")) == 4
        assert row["members_fit"] == ""
    for row, k in zip(sheet, key):
        assert row["summary"].startswith("sum" if k["method"] == "perfect-m" else "blur")
        assert k["recovery_class"] in {"well", "mixed", "poor"}
    # Within an item both methods appear once, and the shown members belong to the key's cluster.
    labels = {"perfect-m": PERFECT, "blurry-m": BLURRY}
    for item in {r["item"] for r in sheet}:
        assert sorted(k["method"] for k in key if k["item"] == item) == ["blurry-m", "perfect-m"]
    for row, k in zip(sheet, key):
        shown = [int(m.split(" ")[1]) for m in (line.split(". ", 1)[1] for line in row["members"].split("\n"))]
        assert all(labels[k["method"]][i] == k["cluster"] for i in shown)


def test_cohens_kappa_known_values():
    assert cohens_kappa([True, True, False, False], [True, True, False, False]) == 1.0
    assert cohens_kappa([True, False, True, False], [True, True, False, False]) == 0.0
    assert cohens_kappa([True, True, False, False], [False, False, True, True]) == -1.0
    assert np.isnan(cohens_kappa([True, True], [True, True]))  # chance agreement is already 1
    assert np.isnan(cohens_kappa([], []))


def test_agreement_counts():
    out = agreement([True, True, False, False, True], [True, False, True, False, True])
    assert (out["n"], out["both"], out["hand_only"], out["auto_only"], out["neither"]) == (5, 2, 1, 1, 1)
    assert out["raw_agreement"] == pytest.approx(3 / 5)


def test_hand_label_rules_and_empty_cells():
    assert hand_label({"members_fit": 4, "members_shown": 10}, "low_coverage") is True
    assert hand_label({"members_fit": 5, "members_shown": 10}, "low_coverage") is False
    assert hand_label({"generic": 1}, "generic") is True and hand_label({"generic": 0}, "generic") is False
    assert hand_label({"fluent": 0}, "not_fluent") is True and hand_label({"fluent": 1}, "not_fluent") is False
    assert hand_label({"unsupported": "1"}, "unsupported") is True
    assert hand_label({"generic": ""}, "generic") is None
    assert hand_label({"generic": float("nan")}, "generic") is None
    assert hand_label({"members_fit": 3, "members_shown": 0}, "low_coverage") is None


def test_audit_agreement_script_on_a_toy_filled_sheet():
    script = load_script("audit_agreement")
    sheet = pd.DataFrame(
        {
            "row_id": ["d-01-A", "d-01-B", "d-01-C", "d-01-D"],
            "members_shown": [10, 10, 10, 10],
            "members_fit": [9, 2, 8, 1],  # low coverage by hand: B, D
            "generic": [0, 1, 0, 1],
            "unsupported": [0, 0, 1, ""],  # D unrated
            "fluent": [1, 1, 0, 1],  # not fluent by hand: C
            "notes": ["", "", "", ""],
        }
    )
    key = pd.DataFrame(
        {
            "row_id": ["d-01-A", "d-01-B", "d-01-C", "d-01-D"],
            "flag_low_coverage": [False, True, True, True],
            "flag_generic": [False, True, False, True],
            "flag_unsupported": [False, False, True, True],
            "flag_not_fluent": [False, False, True, False],
        }
    )
    table = script.compare(sheet, key).set_index("property")
    assert table.loc["low_coverage", ["n", "both", "hand_only", "auto_only", "neither"]].tolist() == [4, 2, 0, 1, 1]
    assert table.loc["generic", "raw_agreement"] == 1.0 and table.loc["generic", "kappa"] == 1.0
    assert table.loc["unsupported", "n"] == 3  # the unrated row is dropped
    assert table.loc["not_fluent", "raw_agreement"] == 1.0
    assert "| generic | 4 |" in script.to_markdown(table.reset_index())


def test_make_audit_sheet_markdown_view():
    script = load_script("make_audit_sheet")
    sheet, _ = build_audit("toy", TEXTS, GOLD, None, partitions(), 0, 3, (("well", 1),))
    md = script.sheet_markdown("toy", sheet, approx=True)
    assert "## toy-01" in md and "### toy-01-A" in md and "FINAL assignments" in md
    assert "perfect-m" not in md and "blurry-m" not in md
