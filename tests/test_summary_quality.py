"""`kslimeans.summary_quality` on tiny hand-made clusters with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from kslimeans.runlog import RunRecord
from kslimeans.summary_quality import (
    aggregate_scores,
    analyze_corpus,
    cluster_overlap,
    content_lemmas,
    fluency_checks,
    label_alignment,
    load_nlp,
    mean_off_diagonal,
    score_step,
    step_partitions,
    unsupported_content,
)

pytest.importorskip("spacy")
try:
    NLP = load_nlp()
except OSError:  # pragma: no cover - the model is a documented install step
    pytest.skip("spaCy model en_core_web_sm is not installed", allow_module_level=True)

PROMPT = "The following is a cluster of banking questions. Write a question that summarizes the following cluster:"

TEXTS = [
    "how do i reset my pin", "i forgot my pin number", "change my pin please", "pin reset help",
    "where is my card", "my card has not arrived", "track my card delivery", "card delivery is late",
    "what is the exchange rate", "exchange rate for euros", "euro exchange rate today", "rate for dollars",
]  # fmt: skip
LABELS = np.repeat([0, 1, 2], 4)
GOLD = LABELS.copy()
CORPUS = analyze_corpus(TEXTS, NLP)


def embeddings() -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((12, 3), dtype=np.float32)
    X[np.arange(12), LABELS] = 1.0
    X[:, :] += 0.02 * np.arange(12)[:, None] % 0.05  # tiny deterministic jitter
    return X, np.eye(3, dtype=np.float32)


def run(summaries, S=None, **kw):
    X, eye = embeddings()
    return score_step(summaries, eye if S is None else S, X, LABELS, GOLD, CORPUS, NLP,
                      prompt=PROMPT, asks_question=True, **kw)  # fmt: skip


GOOD = ["How can I reset my pin?", "Where is my card delivery?", "What is the exchange rate today?"]


def test_good_summaries_have_clean_scores():
    rows, step = run(GOOD)
    assert [r["cluster"] for r in rows] == [0, 1, 2]
    for r in rows:
        assert r["coverage_nearest"] == 1.0 and r["coverage_intents"] == 1.0
        assert r["specificity_margin"] > 0.5
        assert r["unsupported_lemma_share"] == 0.0 and not r["flag_unsupported"]
        assert not (r["flag_generic"] or r["flag_low_coverage"] or r["flag_not_fluent"])
    assert step["genericity"] == pytest.approx(0.0, abs=1e-6)  # orthogonal summaries
    assert step["n_empty_clusters"] == 0


def test_duplicate_summary_is_flagged_and_loses_its_members():
    summaries = [GOOD[0], GOOD[0], GOOD[2]]
    S = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    rows, step = run(summaries, S)
    by = {r["cluster"]: r for r in rows}
    assert by[0]["flag_duplicate"] and by[1]["flag_duplicate"] and not by[2]["flag_duplicate"]
    assert by[0]["flag_near_duplicate"] and by[0]["flag_generic"] and not by[2]["flag_generic"]
    # The first copy is the representative of the tied group for both coverage scores; the later copy gets neither.
    assert by[0]["coverage_nearest"] == 1.0 and by[1]["coverage_nearest"] == 0.0
    assert by[0]["coverage_intents"] == 1.0 and by[1]["coverage_intents"] == 0.0
    assert by[1]["flag_low_coverage"]
    assert step["genericity"] == pytest.approx(1 / 3)  # one identical pair among three summaries


def test_summary_of_member_words_has_no_unsupported_content():
    rows, _ = run(["Where is my card delivery?", "Where is my card delivery?", "Where is my card delivery?"])
    # Only cluster 1's members contain these words; the same text is unsupported elsewhere.
    by = {r["cluster"]: r for r in rows}
    assert by[1]["unsupported_lemma_share"] == 0.0 and not by[1]["flag_unsupported"]
    assert by[0]["unsupported_lemma_share"] == 1.0 and by[0]["flag_unsupported"]


def test_invented_number_and_entity_are_unsupported():
    doc = NLP("Where is my card in Paris for 500 dollars?")
    supported = analyze_corpus(["where is my card", "card delivery to Paris"], NLP)
    blob_texts = " | " + " | ".join(supported.norm) + " | "
    out = unsupported_content([doc], set().union(*supported.lemmas), blob_texts)
    assert out["n_unsupported_numbers"] == 1  # 500 is in no member
    assert "paris" not in out["unsupported_items"].split()  # Paris is in a member
    assert "dollar" in out["unsupported_lemmas"] and out["flag_unsupported"]

    bare = unsupported_content([NLP("Where is my card in Paris?")], set().union(*supported.lemmas), blob_texts)
    assert bare["n_unsupported_numbers"] == 0 and bare["n_unsupported_entities"] == 0
    assert not bare["flag_unsupported"]

    invented = unsupported_content([NLP("Where is my card in Berlin?")], set().union(*supported.lemmas), blob_texts)
    assert invented["n_unsupported_entities"] == 1 and invented["flag_unsupported"]


def test_summary_nearer_a_neighbour_has_negative_margin():
    S = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)  # summaries of clusters 0 and 1 swapped
    rows, _ = run(GOOD, S)
    by = {r["cluster"]: r for r in rows}
    assert by[0]["specificity_margin"] < 0
    assert by[0]["flag_low_margin"] and by[0]["flag_generic"]
    assert by[0]["coverage_nearest"] == 0.0 and by[0]["coverage_intents"] == 0.0
    assert by[2]["specificity_margin"] > 0 and not by[2]["flag_low_margin"]


def test_coverage_in_a_cluster_mixing_two_intents():
    # Cluster 0 holds two intents: A near (1, 0), B near (0, 1). Its summary sits on A; cluster 1's summary is
    # nearer to B than cluster 0's summary is, so B's members are nearest to cluster 1's summary.
    X = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [-1, 0.2], [-1, 0.2]], dtype=np.float32)
    labels = np.array([0, 0, 0, 0, 1, 1])
    gold = np.array([0, 0, 1, 1, 2, 2])
    S = np.array([[1, 0], [-1, 0.2]], dtype=np.float32)
    corpus = analyze_corpus(["pay bill"] * 6, NLP)
    rows, _ = score_step(["pay bill?", "pay bill now?"], S, X, labels, gold, corpus, NLP,
                         prompt=PROMPT, asks_question=True)  # fmt: skip
    r = rows[0]
    assert r["n_intents"] == 2 and r["n_major_intents"] == 2 and r["n_major_covered"] == 1
    assert r["coverage_intents"] == 0.5 and r["coverage_nearest"] == 0.5
    assert not r["flag_low_coverage"]  # exactly one half is not below the threshold
    assert rows[1]["coverage_intents"] == 1.0


def test_members_kept_lost_and_gained_and_lost_in_uncovered_intents():
    X = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [-1, 0.2], [-1, 0.2]], dtype=np.float32)
    labels = np.array([0, 0, 0, 0, 1, 1])
    gold = np.array([0, 0, 1, 1, 2, 2])
    after = np.array([0, 0, 1, 1, 1, 1])  # the two intent-1 members move to cluster 1
    S = np.array([[1, 0], [-1, 0.2]], dtype=np.float32)
    corpus = analyze_corpus(["pay bill"] * 6, NLP)
    rows, _ = score_step(["pay bill?", "pay bill now?"], S, X, labels, gold, corpus, NLP,
                         prompt=PROMPT, asks_question=True, after=after)  # fmt: skip
    r0, r1 = rows
    assert (r0["n_kept"], r0["n_lost"], r0["n_gained"]) == (2, 2, 0)
    assert (r0["n_members_uncovered"], r0["n_lost_uncovered"]) == (2, 2)
    assert (r1["n_kept"], r1["n_lost"], r1["n_gained"]) == (2, 0, 2)
    assert r1["n_lost_uncovered"] == 0


def test_empty_cluster_and_none_summary_get_no_row():
    X, eye = embeddings()
    labels = np.array([0] * 8 + [1] * 4)  # cluster 2 is empty
    rows, step = score_step(GOOD, eye, X, labels, GOLD, CORPUS, NLP, prompt=PROMPT, asks_question=True)
    assert [r["cluster"] for r in rows] == [0, 1] and step["n_empty_clusters"] == 1
    rows, _ = score_step([GOOD[0], None, GOOD[2]], eye, X, LABELS, GOLD, CORPUS, NLP, prompt=PROMPT,
                         asks_question=True)  # fmt: skip
    assert [r["cluster"] for r in rows] == [0, 2]


@pytest.mark.parametrize(
    "summary, asks, failed",
    [
        ("How do I reset my pin?", True, set()),
        ("How do I reset my pin", True, {"flu_no_question"}),
        ("How do I reset my pin", False, set()),
        ('"How do I reset my pin?"', True, {"flu_markup"}),
        ("How do I reset my pin?\"", True, {"flu_markup"}),
        ("**How** do I reset my pin?", True, {"flu_markup"}),
        ("<think>hm</think> How do I reset my pin?", True, {"flu_markup"}),
        ("How do I reset my pin?\nAnd more?", True, {"flu_markup"}),
        ("How can I reset my pin 称呼?", True, {"flu_script"}),
        ("Café pin reset?", True, set()),
        ("pin reset help", False, {"flu_copy"}),
        ("Pin  reset, help!", False, {"flu_copy"}),
        ("Write a question that summarizes the following cluster?", True, {"flu_echo"}),
        ("", True, {"flu_empty", "flu_no_question"}),
    ],
)
def test_fluency_checks(summary, asks, failed):
    members = ["how can i reset my pin", "pin reset help"]
    got = fluency_checks(summary, members, PROMPT, asks)
    assert {k for k, v in got.items() if v} == failed


def test_prompt_ngram_that_a_member_contains_is_not_an_echo():
    members = ["i have online banking questions"]
    prompt = "A cluster of online banking questions"
    assert not fluency_checks("Online banking questions?", members, prompt, True)["flu_echo"]
    assert fluency_checks("Online banking questions?", ["x"], prompt, True)["flu_echo"]


def test_content_lemmas_drop_stop_words_numbers_and_function_words():
    assert content_lemmas(NLP("Where did the cards arrive in 2020?")) == {"card", "arrive"}


def test_cluster_overlap_majority_purity_and_jaccard():
    labels = np.array([0, 0, 0, 1, 1, 2])
    gold = np.array([0, 0, 1, 1, 1, 2])  # class sizes 2, 3, 1
    out = cluster_overlap(labels, gold, 4)
    assert out[0] == {"majority_intent": 0, "purity": pytest.approx(2 / 3), "jaccard_majority": pytest.approx(2 / 3)}
    assert out[1]["majority_intent"] == 1 and out[1]["jaccard_majority"] == pytest.approx(2 / 3)
    assert out[3] == {}  # empty cluster


def test_label_alignment_hungarian_matching():
    S = np.array([[1, 0], [0, 1]], dtype=np.float32)
    L = np.array([[0, 1], [1, 0]], dtype=np.float32)
    out = label_alignment(S, L, [1, 0])
    assert out == {"label_align_sim": pytest.approx(1.0), "label_align_acc": pytest.approx(1.0)}
    assert label_alignment(S, L, [0, 1])["label_align_acc"] == 0.0
    assert np.isnan(label_alignment(S, L, [None, None])["label_align_sim"])


def test_mean_off_diagonal():
    assert mean_off_diagonal(np.ones((3, 3))) == 1.0
    assert mean_off_diagonal(np.eye(3)) == 0.0
    assert np.isnan(mean_off_diagonal(np.ones((1, 1))))


def _record(**kw) -> RunRecord:
    base = dict(method="m", seed=0, summaries_evolution=[["a"], ["b"]])
    return RunRecord(**{**base, **kw})


def test_step_partitions_per_step_source_and_after():
    evo = [np.array([0, 1]), np.array([1, 0]), np.array([0, 0])]
    record = _record(assignments_evolution=evo, summary_source_steps=[0, 1], assignments=evo[-1])
    parts = list(step_partitions(record))
    assert [(p.step, p.source_step, p.approx) for p in parts] == [(0, 0, False), (1, 1, False)]
    assert parts[0].labels is evo[0] and parts[0].after is evo[1] and parts[1].after is evo[2]
    assert [p.step for p in step_partitions(record, final_only=True)] == [1]
    # A final step that stopped on tol has no partition after it.
    stopped = _record(assignments_evolution=evo[:2], summary_source_steps=[0, 1], assignments=evo[1])
    assert list(step_partitions(stopped))[-1].after is None


def test_step_partitions_old_run_is_last_step_against_final_assignments():
    final = np.array([0, 1])
    record = _record(assignments=final)
    (part,) = step_partitions(record)
    assert part.step == 1 and part.approx and part.labels is final and part.after is None
    assert list(step_partitions(_record(summaries_evolution=[]))) == []


def test_aggregate_scores_means_medians_and_shares():
    import pandas as pd

    df = pd.DataFrame(
        {
            "method": ["a", "a", "a", "b"],
            "coverage_nearest": [0.0, 0.5, 1.0, 0.2],
            "flag_generic": [True, False, False, True],
        }
    )
    out = aggregate_scores(df, by=["method"])
    assert out.loc["a", "n"] == 3
    assert out.loc["a", "coverage_nearest_mean"] == pytest.approx(0.5)
    assert out.loc["a", "coverage_nearest_median"] == 0.5
    assert out.loc["a", "flag_generic_share"] == pytest.approx(1 / 3)
    assert out.loc["b", "flag_generic_share"] == 1.0


def test_step_without_a_partition_after_has_no_link_columns():
    # A final summary step that stopped on tol has no partition after it (`after=None`).
    rows, _ = run(GOOD, after=None)
    assert all("n_kept" not in r and "purity_after" not in r for r in rows)


def test_purity_after_is_the_majority_share_of_the_cluster_after_the_step():
    after = LABELS.copy()
    after[3] = 1  # one cluster-0 member (gold 0) joins cluster 1, which then holds 4 gold-1 and 1 gold-0 members
    rows, _ = run(GOOD, after=after)
    by = {r["cluster"]: r for r in rows}
    assert (by[0]["n_after"], by[0]["purity_after"]) == (3, 1.0)
    assert (by[1]["n_after"], by[1]["purity_after"]) == (5, pytest.approx(4 / 5))
    assert by[1]["n_gained"] == 1 and by[0]["n_lost"] == 1


def test_multi_line_summary_is_lemmatized_line_by_line():
    # Lines taken from members are analysed like the members themselves, so nothing is unsupported.
    texts = ["hello ai", "change the ai name"]
    corpus = analyze_corpus(texts, NLP)
    X = np.eye(2, dtype=np.float32)[[0, 0]]
    rows, _ = score_step(["hello ai\nchange the ai name"], np.eye(2, dtype=np.float32)[:1], X, np.array([0, 0]),
                         np.array([0, 0]), corpus, NLP, prompt=PROMPT, asks_question=False)  # fmt: skip
    assert rows[0]["unsupported_lemma_share"] == 0.0
