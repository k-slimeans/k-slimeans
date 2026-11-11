"""BBC News (#18): loader, preparation script, TF-IDF embedder, its summary counters and the config plumbing.

Everything runs on tiny synthetic corpora; no network, no models, no real clustering.
"""

from __future__ import annotations

import importlib.util
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kslimeans import data as data_module
from kslimeans import experiment
from kslimeans.cache import cached_embeddings
from kslimeans.core import summary_kmeans
from kslimeans.data import Dataset, load_dataset
from kslimeans.embeddings import TfidfEmbedder, _df_label, build_corpus_embedder, get_embedder
from kslimeans.schedules import FixedSchedule
from kslimeans.summarizers.base import Cluster, Summarizer
from kslimeans.summarizers.extractive import ExtractiveSummarizer
from kslimeans.summarizers.hf import truncate_words

CSV_PATH = data_module.DATA_DIR / "bbc500.csv"
IDS_PATH = data_module.DATA_DIR / "bbc500_ids.csv"
needs_csv = pytest.mark.skipif(not CSV_PATH.exists(), reason="data/bbc500.csv not built (python scripts/prepare_bbc.py)")
SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "prepare_bbc.py"
TOPICS = ("business", "entertainment", "politics", "sport", "tech")

# Per topic, a few shared words (so every one clears min_df) plus words unique to each document.
TOPIC_WORDS = {
    "business": "market shares profit bank trade economy",
    "entertainment": "film actor music award album concert",
    "politics": "minister election party government vote labour",
    "sport": "match team player season coach league",
    "tech": "software computer internet mobile network device",
}


def make_corpus(per_topic: int = 6) -> tuple[list[str], list[str]]:
    texts, topics = [], []
    for topic in TOPICS:
        for i in range(per_topic):
            texts.append(f"{TOPIC_WORDS[topic]} {topic}{i} the and of")
            topics.append(topic)
    return texts, topics


def tfidf(corpus: list[str], **kwargs) -> TfidfEmbedder:
    kwargs = {"min_df": 3, "min_terms": 10, **kwargs}
    return TfidfEmbedder(corpus, **kwargs)


@pytest.fixture(scope="module")
def prepare_bbc():
    spec = importlib.util.spec_from_file_location("prepare_bbc_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- loader


def test_bbc500_loader_on_a_fixture(tmp_path, monkeypatch):
    texts, topics = make_corpus()
    pd.DataFrame({"text": texts, "topic": topics, "file_id": range(len(texts))}).to_csv(
        tmp_path / "bbc500.csv", index=False
    )
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)

    ds = load_dataset("bbc500")

    assert ds.name == "bbc500"
    assert ds.texts == texts
    assert ds.n_clusters == 5
    assert ds.label_names == sorted(TOPICS)
    assert [ds.label_names[i] for i in ds.labels] == topics


def test_bbc500_prompt_avoids_cluster_articles_and_topics():
    prompt = data_module._BBC_PROMPT.lower()
    for word in ("cluster", "article", *TOPICS):
        assert word not in prompt
    assert "one sentence" in prompt


def test_bbc500_loader_names_the_prepare_script_when_the_csv_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match=r"python scripts/prepare_bbc\.py"):
        load_dataset("bbc500")


def test_committed_ids_are_stratified_unique_and_carry_a_checksum(prepare_bbc):
    ids, sha256 = prepare_bbc.read_ids(IDS_PATH)
    assert len(ids) == 500 and ids["file_id"].is_unique
    assert ids["topic"].value_counts().to_dict() == {topic: 100 for topic in TOPICS}
    assert (ids["file_id"].str.split("/").str[0] == ids["topic"]).all()
    assert len(sha256) == 64 and int(sha256, 16) >= 0
    assert (ids["topic"].tolist() != sorted(ids["topic"]))  # shuffled, not grouped by topic


@needs_csv
def test_generated_bbc500_csv_is_stratified_one_line_per_article_and_matches_the_ids(prepare_bbc):
    import hashlib

    ds = load_dataset("bbc500")
    assert len(ds.texts) == 500
    assert np.bincount(ds.labels).tolist() == [100] * 5
    assert all("\n" not in text for text in ds.texts)
    df = pd.read_csv(CSV_PATH)
    ids, sha256 = prepare_bbc.read_ids(IDS_PATH)
    assert df["file_id"].tolist() == ids["file_id"].tolist()
    assert hashlib.sha256(CSV_PATH.read_bytes()).hexdigest() == sha256


def test_prepare_resamples_then_rebuilds_from_the_ids_and_verifies_the_checksum(prepare_bbc, tmp_path, monkeypatch):
    archive = tmp_path / "bbc.zip"
    make_archive(archive)
    monkeypatch.setattr(prepare_bbc, "EXPECTED_ARTICLES", 150)
    out, ids_path = tmp_path / "out.csv", tmp_path / "ids.csv"

    sha256 = prepare_bbc.prepare(archive, out, ids_path, resample=True, per_topic=10)
    first = out.read_bytes()
    out.unlink()
    assert prepare_bbc.prepare(archive, out, ids_path) == sha256  # default: rebuild from the ids
    assert out.read_bytes() == first
    assert "Head" in pd.read_csv(out)["text"][0] and "Head" not in ids_path.read_text()  # ids carry no text

    ids_path.write_text(ids_path.read_text().replace(sha256, "0" * 64))
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_bbc.prepare(archive, out, ids_path)


# ---------------------------------------------------------------- prepare_bbc.py


def make_archive(path: Path, per_topic: int = 30) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("bbc/README.TXT", "readme")
        for topic in TOPICS:
            for i in range(per_topic):
                body = " ".join(f"{topic}{i}w{j}" for j in range(20))
                archive.writestr(f"bbc/{topic}/{i + 1:03d}.txt", f"Head {topic} {i}\n\nBody  one.\n\n{body}.\n")


def test_format_article_puts_headline_and_body_on_one_line(prepare_bbc):
    assert prepare_bbc.format_article("Ad sales boost profit\n\nFirst  para.\n\nSecond para.\n") == (
        "Ad sales boost profit. First para. Second para."
    )
    assert prepare_bbc.format_article("Is it over?\n\nNo.") == "Is it over? No."


def test_read_archive_and_sample_are_stratified_and_deterministic(prepare_bbc, tmp_path, monkeypatch):
    archive = tmp_path / "bbc.zip"
    make_archive(archive)
    monkeypatch.setattr(prepare_bbc, "EXPECTED_ARTICLES", 150)

    df = prepare_bbc.read_archive(archive)
    assert len(df) == 150  # README.TXT is skipped

    first = prepare_bbc.sample(df, per_topic=10, seed=0)
    again = prepare_bbc.sample(df, per_topic=10, seed=0)
    other = prepare_bbc.sample(df, per_topic=10, seed=1)
    assert first.equals(again)
    assert not first["file_id"].equals(other["file_id"])
    assert first["topic"].value_counts().to_dict() == {topic: 10 for topic in TOPICS}
    assert first["file_id"].is_unique
    assert first["topic"].tolist() != sorted(first["topic"])  # shuffled, not grouped by topic


def test_read_archive_rejects_a_wrong_article_count(prepare_bbc, tmp_path):
    archive = tmp_path / "bbc.zip"
    make_archive(archive, per_topic=2)
    with pytest.raises(ValueError, match="Expected 2225"):
        prepare_bbc.read_archive(archive)


def test_sample_fails_when_a_topic_is_too_small(prepare_bbc):
    df = pd.DataFrame({"text": ["a"], "topic": ["sport"], "file_id": ["sport/001"]})
    with pytest.raises(ValueError, match="business"):
        prepare_bbc.sample(df, per_topic=1)


def test_read_archive_refuses_oversized_members(prepare_bbc, tmp_path):
    archive = tmp_path / "bbc.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("bbc/sport/001.txt", "x" * 1_000_001)
    with pytest.raises(ValueError, match="refusing"):
        prepare_bbc.read_archive(archive)


def test_deduplicate_drops_exact_and_edited_copies_and_keeps_the_first_file_id(prepare_bbc):
    head = "Election vote for the labour party and minister in government " + " ".join(f"w{i}" for i in range(40))
    story = "Winger dropped from the squad after the cup defeat by Ireland " + " ".join(
        f"rugby{i}" for i in range(30)
    )
    df = pd.DataFrame(
        {
            "text": [
                story,
                "Tech firms launch new mobile software for network devices",
                story.replace("defeat", "loss"),  # edited copy: different prefix, high cosine
                "Tech firms launch new mobile software for network devices",  # exact copy
                head,
                head + " and then a long extra section about something else entirely, zebra yak giraffe tapir",
            ],
            "topic": ["sport", "tech", "sport", "tech", "politics", "politics"],
            "file_id": ["sport/344", "tech/002", "sport/332", "tech/001", "politics/001", "politics/002"],
        }
    )

    unique = prepare_bbc.deduplicate(df)

    # First file_id in sorted order wins; politics/002 is dropped by the shared 200-character prefix.
    assert unique["file_id"].tolist() == ["politics/001", "sport/332", "tech/001"]
    assert prepare_bbc.deduplicate(df).equals(unique)
    # With the cosine rule off, the edited copy survives; the exact copy and the shared prefix still go.
    assert prepare_bbc.deduplicate(df, threshold=1.01)["file_id"].tolist() == [
        "politics/001",
        "sport/332",
        "sport/344",
        "tech/001",
    ]


@needs_csv
def test_generated_csv_has_no_duplicate_articles():
    import re

    ds = load_dataset("bbc500")
    normalized = [re.sub(r"\W+", " ", t.lower()).strip() for t in ds.texts]
    assert len(set(normalized)) == 500
    assert len({n[:200] for n in normalized}) == 500
    spec = importlib.util.spec_from_file_location("prepare_bbc_dedup_check", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    df = pd.read_csv(CSV_PATH)
    assert len(module.deduplicate(df)) == 500  # TF-IDF cosine < 0.85 between all pairs


def test_deduplicate_does_not_depend_on_input_row_order(prepare_bbc):
    texts = [
        "Alpha beta gamma delta epsilon zeta eta theta " * 6,
        "Alpha beta gamma delta epsilon zeta eta theta " * 6,
        "Iota kappa lambda mu nu xi omicron pi " * 6,
        "Rho sigma tau upsilon phi chi psi omega " * 6,
    ]
    df = pd.DataFrame(
        {"text": texts, "topic": ["sport"] * 4, "file_id": ["sport/003", "sport/001", "sport/004", "sport/002"]}
    )
    expected = prepare_bbc.deduplicate(df)["file_id"].tolist()
    assert expected == ["sport/001", "sport/002", "sport/004"]
    for seed in range(5):
        shuffled = df.sample(frac=1, random_state=seed)
        assert prepare_bbc.deduplicate(shuffled)["file_id"].tolist() == expected


# ---------------------------------------------------------------- TF-IDF embedder


def test_tfidf_is_deterministic_and_l2_normalized():
    corpus, _ = make_corpus()
    a, b = tfidf(corpus), tfidf(corpus)
    X = a.encode(corpus)

    assert X.dtype == np.float32
    assert X.shape == (len(corpus), a.vocab_size)
    assert a._vectorizer.get_feature_names_out().tolist() == b._vectorizer.get_feature_names_out().tolist()
    assert np.array_equal(X, b.encode(corpus))
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-6)


def test_tfidf_removes_stop_words_and_rare_terms():
    corpus, _ = make_corpus()
    vocabulary = tfidf(corpus)._vocabulary
    assert "the" not in vocabulary and "and" not in vocabulary  # stop words
    assert "business0" not in vocabulary  # in one document only
    assert "market" in vocabulary


def test_tfidf_never_refits_on_summaries():
    corpus, _ = make_corpus()
    embedder = tfidf(corpus)
    vocabulary = dict(embedder._vocabulary)
    idf = embedder._vectorizer.idf_.copy()

    vector = embedder.encode(["market profit zebra zebra zebra"])
    embedder.encode(["zebra"] * 5)

    assert embedder._vocabulary == vocabulary
    assert np.array_equal(embedder._vectorizer.idf_, idf)
    assert vector.shape == (1, len(vocabulary))
    # A word outside the fit corpus contributes nothing: same vector as without it.
    assert np.allclose(vector, embedder.encode(["market profit"]))


def test_tfidf_encodes_the_empty_list():
    assert tfidf(make_corpus()[0]).encode([]).shape[0] == 0


def test_tfidf_fails_loudly_below_the_minimum_vocabulary():
    corpus, _ = make_corpus()
    with pytest.raises(ValueError, match="fewer than the required 700"):
        TfidfEmbedder(corpus, min_df=3)  # default min_terms=700


def test_tfidf_stats_count_oov_tokens_and_zero_vectors():
    embedder = tfidf(make_corpus()[0])
    assert embedder.stats() == {"embed_tokens": 0, "embed_oov_tokens": 0, "embed_zero_vectors": 0}

    # "the" is a stop word (not counted); "zebra" and "yak" are out of vocabulary; the second text is all OOV.
    X = embedder.encode(["market the profit zebra", "yak zebra"])

    assert embedder.stats() == {"embed_tokens": 5, "embed_oov_tokens": 3, "embed_zero_vectors": 1}
    assert not X[1].any() and X[0].any()


def test_tfidf_name_carries_every_setting():
    corpus, _ = make_corpus()
    names = {
        tfidf(corpus).name,
        tfidf(corpus, min_df=4).name,
        tfidf(corpus, max_df=0.9).name,
        tfidf(corpus, max_features=20).name,
        tfidf(corpus, sublinear_tf=True).name,
    }
    assert len(names) == 5


def test_tfidf_name_has_the_sklearn_version_and_one_spelling_per_setting():
    import sklearn

    corpus, _ = make_corpus()
    assert tfidf(corpus).name.endswith(f"-sk{sklearn.__version__}")
    assert _df_label(1) != _df_label(1.0)  # a count is not a proportion
    assert tfidf(corpus, max_df=1.0).name == tfidf(corpus, max_df=1.00).name
    assert "maxdf1p" in tfidf(corpus, max_df=1.0).name and "mindf3-" in tfidf(corpus).name


def test_cache_key_includes_the_tfidf_settings(tmp_path):
    corpus, _ = make_corpus()
    a, b = tfidf(corpus), tfidf(corpus, min_df=4)

    cached_embeddings("bbc", corpus, a, cache_dir=str(tmp_path))
    cached_embeddings("bbc", corpus, b, cache_dir=str(tmp_path))

    files = sorted(p.name for p in tmp_path.glob("*.npz"))
    assert files == sorted([f"bbc__{a.name}_embeddings.npz", f"bbc__{b.name}_embeddings.npz"])
    assert cached_embeddings("bbc", corpus, a, cache_dir=str(tmp_path)).shape[1] == a.vocab_size


def test_build_corpus_embedder_and_get_embedder_errors():
    corpus, _ = make_corpus()
    embedder = build_corpus_embedder({"name": "tfidf", "min_df": 3, "min_terms": 10}, corpus)
    assert isinstance(embedder, TfidfEmbedder) and embedder.min_df == 3
    with pytest.raises(ValueError, match="Unknown corpus-fitted embedder"):
        build_corpus_embedder({"name": "bm25"}, corpus)
    with pytest.raises(TypeError):
        build_corpus_embedder({"name": "tfidf", "bogus": 1}, corpus)
    with pytest.raises(ValueError, match="dict spec"):
        get_embedder("tfidf")
    with pytest.raises(TypeError, match=r"experiment\._resolve_embedder"):
        get_embedder({"name": "tfidf"})


# ---------------------------------------------------------------- summary_kmeans with TF-IDF


class FixedTextSummarizer(Summarizer):
    """One canned summary per cluster index, cycling through `outputs`."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs

    def summarize(self, cluster: Cluster) -> str:
        return self.outputs[cluster.index % len(self.outputs)]


def run_tfidf_kmeans(outputs: list[str]):
    corpus, _ = make_corpus()
    embedder = tfidf(corpus)
    result = summary_kmeans(
        corpus,
        5,
        FixedTextSummarizer(outputs),
        embedder,
        FixedSchedule(2, 10, True, True),
        text_features=embedder.encode(corpus),
        tol=-1.0,
        random_state=0,
        verbose=False,
    )
    return embedder, result


def test_summary_kmeans_logs_oov_and_zero_vector_counters_per_step():
    outputs = ["market profit bank", "film music zebra", "yak yak", "team match coach", "software network"]
    embedder, result = run_tfidf_kmeans(outputs)

    # 2 summary steps x 5 summaries (the clusters are renumbered between steps, so the outputs move around):
    # per step 13 tokens, 3 of them OOV ("zebra", "yak", "yak"), and "yak yak" is a zero vector.
    counters = ("embed_tokens", "embed_oov_tokens", "embed_zero_vectors")
    per_step = {"embed_tokens": 13, "embed_oov_tokens": 3, "embed_zero_vectors": 1}
    for key in counters:
        assert result.cost[key] == 2 * per_step[key]
    summary_rows = [row for row in result.trace if row["summarize"]]
    assert len(summary_rows) == 2
    for row in summary_rows:
        assert {key: row[key] for key in counters} == per_step
    # Step 0 and the summaries, only: the corpus encoding before the loop is not counted.
    assert all(row[key] == 0 for row in result.trace if not row["summarize"] for key in counters)
    assert embedder.stats()["embed_tokens"] > result.cost["embed_tokens"]  # the corpus encode is outside the window
    assert (~result.summary_embeddings.any(axis=1)).sum() == 1


class PlainEmbedder:
    """An embedder with no `stats()`, like the sentence encoders."""

    name = "plain"

    def encode(self, texts):
        return np.array([[len(t), sum(map(ord, t)) % 7, 1.0] for t in texts], dtype=np.float32)


def test_an_embedder_without_stats_leaves_trace_and_cost_exactly_as_before():
    texts = [f"doc{i}" for i in range(20)]
    embedder = PlainEmbedder()
    result = summary_kmeans(
        texts, 3, FixedTextSummarizer(["a b", "cc dd", "eee"]), embedder, FixedSchedule(2, 5, True, True),
        tol=-1.0, random_state=0, verbose=False,
    )
    old_keys = [
        "step", "inertia", "prev_inertia", "frac_changed", "summary_shift", "last_n_iter", "last_max_iter",
        "summarize", "summary_inertia", "n_calls", "n_summarized", "input_tokens", "output_tokens",
        "summarizer_seconds", "n_empty_summaries", "n_truncated_clusters", "n_dropped_texts",
    ]  # fmt: skip
    assert all(list(row) == old_keys for row in result.trace)
    assert list(result.cost) == old_keys[-6:]
    assert not any(key.startswith("embed_") for row in result.trace for key in row)


def test_zero_vector_summaries_do_not_crash_and_are_visible():
    _, result = run_tfidf_kmeans(["zebra"])  # every summary is out of vocabulary
    assert result.cost["embed_zero_vectors"] == 10
    # An empty cluster's "" summary has no tokens but is a zero vector too.
    assert result.cost["embed_oov_tokens"] == result.cost["embed_tokens"] > 0
    assert len(result.assignments) == 30


def test_the_summary_encoder_is_the_fitted_one():
    embedder, result = run_tfidf_kmeans(["market profit bank", "film music"])
    summary = embedder.encode([result.summaries_evolution[0][0]])
    assert np.allclose(result.centroids_evolution[0][0], summary[0])


def test_lsa_summarizer_runs_on_tfidf_sentence_embeddings():
    corpus, _ = make_corpus()
    embedder = tfidf(corpus)
    sentences = ["market shares profit bank", "bank trade economy zebra", "yak yak yak yak", "profit trade market"]
    cluster = Cluster(0, (0, 1), corpus[:2], sentences, embedder.encode(sentences))

    summary = ExtractiveSummarizer("lsa", top_k=2).summarize(cluster)

    assert len(summary.split("\n")) == 2
    assert "yak yak yak yak" not in summary.split("\n")[0]  # the all-zero sentence is not ranked first


# ---------------------------------------------------------------- config


def test_resolve_embedder_dict_and_str(monkeypatch):
    corpus, _ = make_corpus()
    embedder = experiment._resolve_embedder({"name": "tfidf", "min_df": 3, "min_terms": 10}, corpus)
    assert isinstance(embedder, TfidfEmbedder)

    sentinel = object()
    monkeypatch.setattr(experiment, "get_embedder", lambda name: sentinel)
    assert experiment._resolve_embedder("minilm", corpus) is sentinel


def test_run_offline_accepts_a_tfidf_embedder_dict(tmp_path, monkeypatch):
    corpus, topics = make_corpus()
    labels = np.array([TOPICS.index(t) for t in topics], dtype=np.int64)
    ds = Dataset("synth_bbc", corpus, labels, 5, prompt="", text_type="")
    monkeypatch.setattr(experiment, "load_dataset", lambda name, **kwargs: ds)
    monkeypatch.setattr(
        experiment, "build_summarizer", lambda cfg, prompt=None, text_type=None: FixedTextSummarizer(["market", "zebra"])
    )
    cfg = {
        "name": "bbc_synth",
        "dataset": "synth_bbc",
        "embedder": {"name": "tfidf", "min_df": 3, "min_terms": 10},
        "seeds": [0],
        "schedule": {"type": "fixed", "n_summaries": 2, "max_iter": 10},
        "summarizer": {"type": "lsa"},
        "cache_dir": str(tmp_path / "cache"),
        "results_dir": str(tmp_path / "results"),
    }

    run_dir = experiment.run_offline(cfg)

    import json

    row = json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[0])
    assert row["embed_zero_vectors"] > 0 and row["embed_oov_tokens"] > 0
    assert "embed_zero_vectors" in (run_dir / "summary.csv").read_text()
    assert list((tmp_path / "cache").glob("synth_bbc__tfidf-mindf3-*_embeddings.npz"))


def test_bbc_configs_load_and_pass_the_key_check():
    from kslimeans.config import load_config

    root = Path(__file__).resolve().parent.parent / "configs"
    for name in ("bbc_tfidf.yaml", "bbc_dense.yaml"):
        cfg = load_config(root / name)
        experiment._check_keys(cfg)
        assert cfg["dataset"] == "bbc500" and cfg["seeds"] == [0, 1, 2, 3, 4]
        assert cfg["schedule"]["n_summaries"] == 5 and cfg["summarizer"]["max_new_tokens"] == 64
    assert load_config(root / "bbc_tfidf.yaml")["embedder"] == {"name": "tfidf", "min_df": 20}
    assert load_config(root / "bbc_dense.yaml")["embedder"] == "minilm"


@needs_csv
def test_tfidf_config_gives_the_required_vocabulary_on_the_real_corpus():
    ds = load_dataset("bbc500")
    embedder = build_corpus_embedder({"name": "tfidf", "min_df": 20}, ds.texts)
    assert 700 <= embedder.vocab_size <= 1000
    assert embedder.encode(ds.texts).shape == (500, embedder.vocab_size)


# ---------------------------------------------------------------- member words


def test_truncate_words_keeps_leading_words_and_short_texts_untouched():
    assert truncate_words("a b c d e", 3) == "a b c"
    assert truncate_words("a  b\nc", 5) == "a  b\nc"
    assert truncate_words("a b c", 3) == "a b c"


def test_hf_render_cuts_members_to_max_words_before_the_token_budget():
    from kslimeans.summarizers.hf import HFSummarizer

    class Tokenizer:
        chat_template = "x"

        def apply_chat_template(self, messages, tokenize=False, **kwargs):
            return "\n".join(m["content"] for m in messages)

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}

    summarizer = HFSummarizer.__new__(HFSummarizer)
    summarizer.tokenizer = Tokenizer()
    summarizer.system_prompt, summarizer.prompt, summarizer.text_type = "sys", "ask", ""
    summarizer.chat_template_kwargs, summarizer._fold_system = {}, False
    summarizer.max_input_tokens, summarizer.model_id = 10_000, "m"
    summarizer._warned_trim, summarizer._stats = False, {"n_truncated_clusters": 0, "n_dropped_texts": 0}

    summarizer.max_words_per_text = 2
    assert summarizer._render(["a b c d", "e f g"]).endswith("a b\ne f")
    summarizer.max_words_per_text = None
    assert summarizer._render(["a b c d", "e f g"]).endswith("a b c d\ne f g")


def test_tfidf_run_dir_round_trips_through_runlog(tmp_path, monkeypatch):
    from kslimeans.runlog import CacheMissError, embed_summaries, load_run, load_run_dataset, load_run_embedder

    corpus, topics = make_corpus()
    labels = np.array([TOPICS.index(t) for t in topics], dtype=np.int64)
    ds = Dataset("synth_bbc", corpus, labels, 5, prompt="", text_type="")
    monkeypatch.setattr(experiment, "load_dataset", lambda name, **kwargs: ds)
    summaries = ["market profit bank", "film music award", "minister election vote", "team match coach", "zebra"]
    monkeypatch.setattr(
        experiment, "build_summarizer", lambda cfg, prompt=None, text_type=None: FixedTextSummarizer(summaries)
    )
    cfg = {
        "name": "bbc_synth",
        "dataset": "synth_bbc",
        "embedder": {"name": "tfidf", "min_df": 3, "min_terms": 10},
        "seeds": [0],
        "schedule": {"type": "fixed", "n_summaries": 2, "max_iter": 10},
        "summarizer": {"type": "lsa"},
        "cache_dir": str(tmp_path / "cache"),
        "results_dir": str(tmp_path / "results"),
    }
    run_dir = experiment.run_offline(cfg)

    log = load_run(run_dir)
    assert log.config["embedder"] == cfg["embedder"]
    record = log.record("k-NLPmeans", 0)
    assert record.summaries_evolution

    # The run cached its embeddings under the TF-IDF name, so loading hits and never resolves a model by name.
    monkeypatch.setattr("kslimeans.cache.get_embedder", lambda name: pytest.fail("resolved by name"))
    got_ds, X = load_run_dataset(log)
    embedder = load_run_embedder(log, got_ds)
    assert isinstance(embedder, TfidfEmbedder)
    assert (tmp_path / "cache" / f"synth_bbc__{embedder.name}_embeddings.npz").exists()
    np.testing.assert_array_equal(X, embedder.encode(corpus))

    # Summaries re-embed in the run's space (the fit is deterministic, so the refit equals the run's).
    first = record.summaries_evolution[0]
    assert embed_summaries(embedder, first).shape == (5, embedder.vocab_size)

    with pytest.raises(CacheMissError, match="tfidf-mindf3"):
        load_run_dataset(log, tmp_path / "other_cache")
    # A miss with allow_encode encodes with the fitted TF-IDF, not with an embedder resolved by name.
    _, X_new = load_run_dataset(log, tmp_path / "other_cache", allow_encode=True)
    np.testing.assert_array_equal(X_new, X)
