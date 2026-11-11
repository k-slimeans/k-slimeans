"""`summary_guard` (#47): flagged summaries keep their cluster's k-means center as the start centroid.

Synthetic data, a summarizer that returns chosen texts per cluster index, an embedder that gives every
chosen text a fixed vector. `_Scripted` (from test_summarize_mask) records the `State` before each step, so
`states[j].centers` is the k-means center a guarded cluster must fall back to at step `j`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

from kslimeans import experiment
from kslimeans.aggregate import method_label
from kslimeans.config import load_config
from kslimeans.core import GUARD_COUNT_KEYS, SummaryGuard, _guarded_start, build_summary_guard, summary_kmeans
from kslimeans.embeddings import Embedder
from kslimeans.gpu_monitor import parse_gpu_check
from kslimeans.schedules import FixedSchedule, Step, build_schedule
from kslimeans.summarizers.base import Cluster, Summarizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core_equivalence import (  # noqa: E402,F401
    ClosestToMeanSummarizer,
    LookupEmbedder,
    _assert_matches,
    _load_fixture,
    synthetic,
)
from test_experiment import _FakeEmbedder, _FakeSummarizer, _base_config, _make_synthetic_dataset  # noqa: E402
from test_summarize_mask import _Scripted  # noqa: E402
from test_trace import TRACE_KEYS  # noqa: E402
from synth import N_BLOBS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DUP = "A very common question"
META = "What is the main topic of this cluster?"


class TextEmbedder(Embedder):
    """Documents come from the feature table; every other text gets a fixed vector derived from its bytes."""

    name = "fake"

    def __init__(self, texts: list[str], X: np.ndarray) -> None:
        self._docs = {t: X[i] for i, t in enumerate(texts)}
        self._dim = X.shape[1]

    def vector(self, text: str) -> np.ndarray:
        if text in self._docs:
            return self._docs[text]
        if text == "":
            return np.zeros(self._dim, dtype=np.float32)
        rng = np.random.default_rng(zlib.crc32(text.encode()))
        return (10 * rng.normal(size=self._dim)).astype(np.float32)

    def encode(self, texts):
        return np.stack([self.vector(t) for t in texts]).astype(np.float32)


@pytest.fixture(scope="module")
def separated():
    """Six blobs of ten documents, far enough apart that k-means recovers them: clusters == blobs at every step."""
    rng = np.random.default_rng(1)
    centers = 100.0 * np.eye(N_BLOBS, 8)
    X = np.concatenate([c + rng.normal(scale=0.5, size=(10, 8)) for c in centers]).astype(np.float32)
    return [f"doc{i}" for i in range(len(X))], X, None


class ChosenSummarizer(Summarizer):
    """`choose(cluster_index, call_number)` is the summary; call_number counts clusters summarized so far."""

    needs_sentences = False

    def __init__(self, choose) -> None:
        self.choose = choose
        self.calls = 0

    def summarize(self, cluster: Cluster) -> str:
        self.calls += 1
        return self.choose(cluster.index, self.calls)


class BlobEmbedder(TextEmbedder):
    """A summary ending in "blob N" embeds just off the mean of blob N's documents (a good centroid that is not the k-means center, as a real summary is)."""

    def vector(self, text: str) -> np.ndarray:
        if text.split()[-1:] and text.split()[-1].isdigit() and "blob" in text:
            b = int(text.split()[-1])
            return self._blob_means[b]
        return super().vector(text)

    def __init__(self, texts, X) -> None:
        super().__init__(texts, X)
        per = len(X) // N_BLOBS
        self._blob_means = [X[b * per : (b + 1) * per].mean(axis=0) + 0.3 for b in range(N_BLOBS)]


class BlobSummarizer(Summarizer):
    """Summary by the majority blob of the members (docN is in blob N // 10), so it follows the documents, not the
    cluster id; that matters because the memo cache is keyed by the members and a relabelled cluster reuses it."""

    needs_sentences = False

    def __init__(self, text_of_blob) -> None:
        self.text_of_blob = text_of_blob

    def summarize(self, cluster: Cluster) -> str:
        blobs = np.bincount([int(t[3:]) // 10 for t in cluster.texts], minlength=N_BLOBS)
        return self.text_of_blob(int(blobs.argmax()))


def plain(c: int, _call: int = 0) -> str:
    return f"Question about thing {c}"


def run(synthetic, choose, schedule, *, guard=None, tol=-1.0, embedder_cls=TextEmbedder):
    """`choose` is a function of the cluster index (one summary step) or a `Summarizer` (several)."""
    texts, X, _ = synthetic
    embedder = embedder_cls(texts, X)
    result = summary_kmeans(
        texts,
        N_BLOBS,
        choose if isinstance(choose, Summarizer) else ChosenSummarizer(choose),
        embedder,
        schedule,
        text_features=X,
        tol=tol,
        random_state=0,
        verbose=False,
        summary_guard=guard,
    )
    return result, embedder


def dup_01(c: int, _call: int = 0) -> str:
    return DUP if c in (0, 1) else plain(c)


BOTH = SummaryGuard(duplicate=True, mentions_cluster=True)


# --- off: byte-identical ---------------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [{}, {"summary_guard": None}], ids=["omitted", "none"])
def test_off_matches_fixture_and_logs_nothing_new(synthetic, kwargs):
    texts, X, embed = synthetic
    result = summary_kmeans(
        texts, N_BLOBS, ClosestToMeanSummarizer(embed), LookupEmbedder(embed), FixedSchedule(3, 30),
        text_features=X, tol=-1.0, random_state=0, init="k-means++", verbose=False, **kwargs,
    )  # fmt: skip
    _assert_matches(result, _load_fixture("synthetic_i1_f1.npz"))
    assert result.guarded_evolution is None
    assert all(list(row) == list(result.trace[0]) and set(row) == TRACE_KEYS for row in result.trace)
    assert list(result.trace[0]) == [
        "step", "inertia", "prev_inertia", "frac_changed", "summary_shift", "last_n_iter", "last_max_iter",
        "summarize", "summary_inertia", "n_calls", "n_summarized",
        "input_tokens", "output_tokens", "summarizer_seconds", "n_empty_summaries", "n_truncated_clusters",
        "n_dropped_texts",
    ]  # fmt: skip


def test_all_false_or_empty_guard_is_rejected_not_silently_off():
    for spec in ({}, {"duplicate": False, "mentions_cluster": False}):
        with pytest.raises(ValueError, match="no flag on"):
            build_summary_guard(spec)


def test_guard_that_flags_nothing_changes_no_output(separated):
    summarizer = lambda: BlobSummarizer(lambda b: f"Question about blob {b}")  # noqa: E731
    off, _ = run(separated, summarizer(), FixedSchedule(3, 30), embedder_cls=BlobEmbedder)
    on, _ = run(separated, summarizer(), FixedSchedule(3, 30), guard=BOTH, embedder_cls=BlobEmbedder)
    assert on.guarded_evolution == [[], [], []]
    assert np.array_equal(on.assignments, off.assignments)
    assert np.array_equal(on.centroids, off.centroids)
    for a, b in zip(on.centroids_evolution, off.centroids_evolution):
        assert np.array_equal(a, b)
    assert on.summaries_evolution == off.summaries_evolution
    volatile = ("inertia", "prev_inertia", "summarizer_seconds")  # not bit-reproducible run to run
    for a, b in zip(on.trace, off.trace):
        a = {k: v for k, v in a.items() if k not in volatile + GUARD_COUNT_KEYS}
        assert a == {k: v for k, v in b.items() if k not in volatile}


# --- full steps --------------------------------------------------------------------------------------------


def test_all_members_of_a_duplicate_group_get_their_kmeans_center(synthetic):
    def choose(c, _call=0):
        return DUP if c in (0, 1, 4) else plain(c)

    schedule = _Scripted([Step(True, 20)])
    result, embedder = run(synthetic, choose, schedule, guard=SummaryGuard(duplicate=True))
    centers = schedule.states[0].centers
    start = result.centroids_evolution[0]

    assert result.guarded_evolution == [[0, 1, 4]]
    for c in (0, 1, 4):
        assert np.array_equal(start[c], centers[c])
        assert not np.array_equal(start[c], embedder.vector(DUP))
    for c in (2, 3, 5):  # unflagged rows are the summary embeddings, exactly
        assert np.array_equal(start[c], embedder.vector(plain(c)))
    # Summaries are logged as generated; summary_embeddings (and so the metrics' summary side) are untouched.
    assert result.summaries_evolution[0] == [choose(c) for c in range(N_BLOBS)]
    assert np.array_equal(result.summary_embeddings, embedder.encode(result.summaries))


def test_unique_summaries_are_not_flagged_by_the_duplicate_rule(synthetic):
    result, _ = run(synthetic, plain, _Scripted([Step(True, 20)]), guard=SummaryGuard(duplicate=True))
    assert result.guarded_evolution == [[]]


def test_empty_summaries_are_never_duplicates(synthetic):
    result, embedder = run(synthetic, lambda c, _=0: "" if c < 2 else plain(c), _Scripted([Step(True, 20)]), guard=BOTH)
    assert result.guarded_evolution == [[]]
    assert np.array_equal(result.centroids_evolution[0][0], embedder.vector(""))


def test_mentions_cluster_is_flagged_case_insensitively(synthetic):
    def choose(c, _call=0):
        return {1: META, 3: "A SUMMARY of things", 5: DUP}.get(c, plain(c))

    schedule = _Scripted([Step(True, 20)])
    result, embedder = run(synthetic, choose, schedule, guard=SummaryGuard(mentions_cluster=True))
    assert result.guarded_evolution == [[1, 3]]  # DUP is unique here and the duplicate rule is off anyway
    centers = schedule.states[0].centers
    assert np.array_equal(result.centroids_evolution[0][1], centers[1])
    assert np.array_equal(result.centroids_evolution[0][3], centers[3])
    assert np.array_equal(result.centroids_evolution[0][5], embedder.vector(DUP))


def test_duplicate_off_leaves_duplicates_alone(synthetic):
    result, embedder = run(synthetic, dup_01, _Scripted([Step(True, 20)]), guard=SummaryGuard(mentions_cluster=True))
    assert result.guarded_evolution == [[]]
    assert np.array_equal(result.centroids_evolution[0][0], embedder.vector(DUP))


def test_trace_counts_and_overlap(synthetic):
    def choose(c, _call=0):
        return {0: META, 1: META, 2: "Another cluster question", 3: plain(3)}.get(c, DUP)  # 0,1 dup+meta; 2 meta; 4,5 dup

    result, _ = run(synthetic, choose, FixedSchedule(2, 20), guard=BOTH)
    first = next(r for r in result.trace if r["summarize"])
    assert first["n_guarded"] == 5 and first["n_guarded_duplicate"] == 4 and first["n_guarded_mentions"] == 3
    assert first["n_summarized"] == N_BLOBS  # still the number of clusters summarized
    assert result.guarded_evolution[0] == [0, 1, 2, 4, 5]
    for r in result.trace:
        assert set(r) == TRACE_KEYS | set(GUARD_COUNT_KEYS)
        assert list(r)[-3:] == list(GUARD_COUNT_KEYS)
    for r in result.trace:
        if not r["summarize"]:
            assert [r[k] for k in GUARD_COUNT_KEYS] == [0, 0, 0]  # step 0 and non-summary fits
    assert len(result.guarded_evolution) == len(result.summaries_evolution)


def test_summary_inertia_and_first_lloyd_start_use_the_guarded_centroids(synthetic):
    texts, X, _ = synthetic
    schedule = _Scripted([Step(True, 20)])
    result, _ = run(synthetic, dup_01, schedule, guard=BOTH)
    start = result.centroids_evolution[0].astype(np.float64)
    expected = float(((X[:, None, :].astype(np.float64) - start[None]) ** 2).sum(-1).min(axis=1).sum())
    assert result.trace[1]["summary_inertia"] == pytest.approx(expected, rel=1e-6)


def test_memo_cache_and_summaries_are_unaffected(synthetic):
    off, _ = run(synthetic, dup_01, _Scripted([Step(True, 20)]))
    on, _ = run(synthetic, dup_01, _Scripted([Step(True, 20)]), guard=BOTH)
    assert on.summaries_evolution == off.summaries_evolution
    assert on.n_calls == off.n_calls == N_BLOBS
    assert on.cost["n_empty_summaries"] == off.cost["n_empty_summaries"]


# --- tol ---------------------------------------------------------------------------------------------------


def test_tol_shift_uses_summary_embeddings_so_the_stopping_rule_is_unchanged(separated):
    # The summaries repeat step to step (same text per blob), so the summary-embedding shift of
    # step 2 is 0 and the run stops. A shift measured against the fallback centers would not be 0.
    # Blobs 0 and 1 get a summary that mentions the cluster, so the guard is active; its embedding is still a fine centroid.
    summarizer = lambda: BlobSummarizer(lambda b: f"This cluster is blob {b}" if b < 2 else f"Question about blob {b}")  # noqa: E731
    off, _ = run(separated, summarizer(), FixedSchedule(5, 20), tol=1e-4, embedder_cls=BlobEmbedder)
    on, _ = run(separated, summarizer(), FixedSchedule(5, 20), guard=BOTH, tol=1e-4, embedder_cls=BlobEmbedder)
    assert off.converged and on.converged
    assert len(on.summaries_evolution) == len(off.summaries_evolution) == 2
    assert len(on.guarded_evolution) == 2  # aligned with summaries_evolution, even for the step that stopped
    assert on.guarded_evolution[0] and on.guarded_evolution[1]  # the guard was active, and the run still converges
    assert len(on.trace) == 1 + len(on.centroids_evolution)
    # The first summary step compares the same ref (the step-0 centers) to the same summary embeddings.
    assert on.trace[1]["summary_shift"] == off.trace[1]["summary_shift"]
    assert np.array_equal(on.summary_embeddings, off.summary_embeddings)


# --- partial steps -----------------------------------------------------------------------------------------


def test_partial_steps_flag_only_active_clusters_and_keep_masked_off_centers(synthetic):
    def mask(*on):
        return [c in on for c in range(N_BLOBS)]

    def choose(c, _call=0):
        return DUP if c in (0, 1, 3) else plain(c)

    schedule = _Scripted([Step(True, 20, mask(0, 1, 2)), Step(True, 20, mask(3, 4))])
    result, embedder = run(synthetic, choose, schedule, guard=SummaryGuard(duplicate=True))
    first_centers, second_centers = schedule.states[0].centers, schedule.states[1].centers

    # Step 1: 0 and 1 duplicate each other; 2 is unique; clusters 3-5 are masked off (never flagged).
    assert result.guarded_evolution[0] == [0, 1]
    start = result.centroids_evolution[0]
    assert np.array_equal(start[0], first_centers[0]) and np.array_equal(start[1], first_centers[1])
    assert np.array_equal(start[2], embedder.vector(plain(2)))
    for c in (3, 4, 5):
        assert np.array_equal(start[c], first_centers[c])

    # Step 2: 3 repeats cluster 0's stale summary, so 3 is flagged and 0 (masked off) is not; 4 is unique.
    assert result.guarded_evolution[1] == [3]
    start = result.centroids_evolution[1]
    assert np.array_equal(start[3], second_centers[3])
    assert np.array_equal(start[4], embedder.vector(plain(4)))
    for c in (0, 1, 2, 5):
        assert np.array_equal(start[c], second_centers[c])

    # The embeddings of every summarized cluster, flagged or not, are still recorded and used for ref.
    assert np.array_equal(result.summary_embeddings[3], embedder.vector(DUP))
    assert np.array_equal(result.summary_embeddings[0], embedder.vector(DUP))
    assert np.isnan(result.summary_embeddings[5]).all()
    assert [r["n_guarded"] for r in result.trace] == [0, 2, 1]
    assert [r["n_summarized"] for r in result.trace] == [0, 3, 2]


def test_partial_step_with_empty_mask_flags_nothing(synthetic):
    result, _ = run(synthetic, plain, _Scripted([Step(True, 20, [False] * N_BLOBS)]), guard=BOTH)
    assert result.guarded_evolution == [[]]
    assert result.trace[1]["n_guarded"] == 0


# --- helpers and config --------------------------------------------------------------------------------------


def test_guarded_start_returns_the_embeddings_untouched_when_nothing_is_flagged():
    emb = np.ones((3, 2), dtype=np.float32)
    assert _guarded_start(emb, np.zeros((3, 2)), []) is emb


def test_guarded_start_keeps_the_wider_dtype_and_copies():
    emb = np.ones((3, 2), dtype=np.float32)
    centers = np.full((3, 2), 0.1, dtype=np.float64)
    start = _guarded_start(emb, centers, [1])
    assert start.dtype == np.float64 and start[1, 0] == 0.1 and start[0, 0] == 1.0
    assert emb[1, 0] == 1.0  # the summary embeddings (also `ref`) are not modified


def test_build_summary_guard_accepts_dicts_and_rejects_typos():
    assert build_summary_guard(None) is None
    assert build_summary_guard({"duplicate": True}) == SummaryGuard(duplicate=True)
    assert build_summary_guard(BOTH) is BOTH
    with pytest.raises(ValueError, match="Unknown summary_guard key"):
        build_summary_guard({"duplicates": True})
    with pytest.raises(ValueError, match="must be true or false"):
        build_summary_guard({"duplicate": "yes"})
    with pytest.raises(TypeError):
        build_summary_guard(["duplicate"])


def test_run_offline_rejects_a_bad_guard_before_loading_anything(tmp_path):
    with pytest.raises(ValueError, match="summary_guard"):
        experiment.run_offline(_base_config(tmp_path, summary_guard={"dupes": True}))


def _patch_fakes(monkeypatch):
    ds, X = _make_synthetic_dataset()
    embedder = _FakeEmbedder({t: X[i] for i, t in enumerate(ds.texts)})
    monkeypatch.setattr(experiment, "load_dataset", lambda name, **kwargs: ds)
    monkeypatch.setattr(experiment, "get_embedder", lambda name: embedder)
    monkeypatch.setattr(experiment, "build_summarizer", lambda cfg, prompt=None, text_type=None: _FakeSummarizer())


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_offline_logs_the_guard_only_when_on(tmp_path, monkeypatch):
    _patch_fakes(monkeypatch)
    off = experiment.run_offline(_base_config(tmp_path, seeds=[0], results_dir=str(tmp_path / "off")))
    # _FakeSummarizer's texts contain "summar", so the mentions rule flags every cluster.
    guard = {"duplicate": True, "mentions_cluster": True}
    on = experiment.run_offline(_base_config(tmp_path, seeds=[0], summary_guard=guard, results_dir=str(tmp_path / "on")))

    (off_sum,) = _read(off / "summaries.jsonl")
    assert "guarded_evolution" not in off_sum
    assert all(not (set(GUARD_COUNT_KEYS) & set(r)) for r in _read(off / "trace.jsonl"))

    (on_sum,) = _read(on / "summaries.jsonl")
    assert on_sum["guarded_evolution"] == [list(range(4))] * len(on_sum["summaries_evolution"])
    assert {k for k in on_sum} - {"guarded_evolution"} == set(off_sum)
    rows = [r for r in _read(on / "trace.jsonl") if r["summarize"]]
    assert rows and all(r["n_guarded"] == 4 and r["n_guarded_mentions"] == 4 for r in rows)
    assert yaml_guard(on) == guard

    off_rows, on_rows = _read(off / "metrics.jsonl"), _read(on / "metrics.jsonl")
    assert [set(r) for r in off_rows] == [set(r) for r in on_rows]  # metrics.jsonl keeps its keys


def yaml_guard(run_dir):
    import yaml

    return yaml.safe_load((run_dir / "config.yaml").read_text())["summary_guard"]


# --- the #47 config and runner ---------------------------------------------------------------------------------


V1 = ROOT / "configs" / "clinc150_summarizers.yaml"
GUARD = ROOT / "configs" / "clinc150_guard.yaml"


def test_guard_config_loads_and_validates():
    cfg = load_config(GUARD)
    experiment._check_keys(cfg)
    parse_gpu_check(cfg["gpu_check"])
    build_schedule(cfg["schedule"])
    assert build_summary_guard(cfg["summary_guard"]) == BOTH
    assert cfg["gpu_check"] == {"require_idle": "warn"}


def test_guard_config_is_the_31_protocol_for_the_4b_with_only_the_guard_and_names_changed():
    v1, guard = load_config(V1), load_config(GUARD)
    assert guard["summarizer"] == {**v1["summarizer"], "model_id": "Qwen/Qwen3-4B", "batch_size": 1}
    assert "prompt" not in guard["summarizer"]  # the #31 prompt
    for key in ("dataset", "embedder", "seeds", "force_context_length", "tol", "schedule", "cache_dir", "results_dir"):
        assert guard[key] == v1[key], key
    assert guard["seeds"] == [0, 1, 2, 3, 4]
    assert "baselines" not in guard  # k-means comes from the #31 runs
    assert set(guard) - set(v1) == {"summary_guard", "gpu_check"}
    assert guard["name"] != v1["name"] and guard["method_name"] == "k-SLMmeans-guard"
    assert method_label(guard) == "k-SLMmeans-guard (Qwen3-4B)"
    assert method_label(guard) != method_label({**v1, "summarizer": guard["summarizer"]})


def test_runner_is_valid_executable_bash_that_reads_the_31_rows():
    script = ROOT / "scripts" / "run_clinc150_guard.sh"
    assert script.stat().st_mode & 0o111
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in text
    assert "configs/clinc150_guard.yaml" in text and "configs/clinc150_summarizers.yaml" not in text
    assert "clinc150_lsa" in text and "clinc150_qwen3-4b" in text and "clinc150_guard_qwen3-4b" in text
