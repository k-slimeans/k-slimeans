"""Generate `.npz` fixtures from the ORIGINAL summaryCentroids.py.

Run once, offline, whenever the synthetic setup in `synth.py` changes, or
the upstream code is updated:

    .venv/bin/python tests/fixtures/make_fixtures.py \\
        --upstream /path/to/summaryCentroids

This imports the *unmodified* upstream module via `importlib` (so it
works regardless of the caller's cwd) and calls its real, unmodified
`kSummariesmeans` directly. For the synthetic cases, it monkeypatches the
module's `get_embeddings` and `summarize_cluster` globals with
deterministic fakes: a fixed lookup-table "embedder" (see
`synth.make_lookup_embedder`) and a "closest member to the cluster mean"
fake summarizer (`synth.closest_to_mean`). Because those fakes are pure
functions of their inputs, upstream's `kSummariesmeans` becomes fully
deterministic and its outputs are exact ground truth for
`tests/test_core_equivalence.py`, which re-implements the same fakes as
`Embedder`/`Summarizer` subclasses and checks `kslimeans.summary_kmeans`
against them with `np.array_equal`.

It also regenerates `bank77_lsa_seed0.npz`: the real distilbert
embeddings + LSA extractive summarizer run from `bank77_smoke.py`, on the
real Bank77 test CSV, run through the ORIGINAL, un-monkeypatched
`get_embeddings`/`summarize_cluster`. That fixture is what
`tests/test_bank77.py` checks the new package against, to prove exact
reproduction on real data (not just the synthetic corpus above).

Two more fixtures cover the mini-batch/streaming variant
(`tests/test_minibatch.py`): `minibatch_synthetic.npz` runs upstream's
ORIGINAL `miniBatchSummarymeans` over `synth.make_synthetic_minibatch_data`
with the same fakes as above (`miniBatchSummarymeans` calls
`text_sentences.items()` unconditionally, so a dummy per-document
sentence dict is built even though the fake summarizer ignores it); and
`bank77_minibatch_lsa_seed0.npz` runs upstream's ORIGINAL
`miniBatchKNLPmeans` on real Bank77 with real distilbert embeddings and a
real sentence corpus (built exactly like
`data_loaders/preprocess_offline_data.ipynb`: `build_sentence_corpus` +
`embed_sentences`, then a `doc -> [[sentence, embedding], ...]` dict).

A fifth, opt-in fixture (`--bank77-grid`) backs the README's k-NLPmeans
grid reproduction claim (`tests/test_bank77_grid.py`): `bank77_grid_seed0.npz`
runs upstream's ORIGINAL `kNLPmeans` over every (method, top_k, max_llm_iter)
combination in `bank77_grid_settings()` on real Bank77, with the same real
distilbert embeddings + sentence corpus built once and shared across all
settings (the same `build_sentence_corpus` + `embed_sentences` recipe as the
minibatch fixture above). It is off by default -- pass `--bank77-grid` -- and
never touches the other fixtures.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from synth import (  # noqa: E402
    closest_to_mean,
    make_lookup_embedder,
    make_synthetic_data,
    make_synthetic_minibatch_data,
)

N_CLUSTERS = 6  # matches synth.N_BLOBS

# The README's Bank77 k-NLPmeans grid claim: LSA/centroid/TextRank x top_k 3,5 x 1 or 5
# summary steps, max_iter 120, seed 0 -- 12 settings total.
BANK77_GRID_METHODS = ("lsa", "centroid", "textrank")
BANK77_GRID_TOP_K = (3, 5)
BANK77_GRID_MAX_LLM_ITER = (1, 5)


def bank77_grid_settings() -> list[tuple[str, int, int]]:
    """The 12 `(method, top_k, max_llm_iter)` settings the grid claim covers, in the
    order used both by the fixture writer below and its test (`test_bank77_grid.py`)."""
    return [
        (method, top_k, max_llm_iter)
        for method in BANK77_GRID_METHODS
        for top_k in BANK77_GRID_TOP_K
        for max_llm_iter in BANK77_GRID_MAX_LLM_ITER
    ]


def bank77_grid_key(method: str, top_k: int, max_llm_iter: int) -> str:
    """The `.npz` key prefix for one grid setting (e.g. `"lsa_top5_iter1_assignments"`)."""
    return f"{method}_top{top_k}_iter{max_llm_iter}"


def _load_module(name: str, path: Path):
    """Import a module from an explicit file path, independent of cwd or sys.path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fakes(sc, texts: list[str], X: np.ndarray) -> None:
    embed = make_lookup_embedder(texts, X)

    def fake_get_embeddings(texts_, model="text-embedding-3-small", emb_type="openai", instructor_prompt=""):
        return embed(list(texts_))

    def fake_summarize_cluster(
        texts_,
        prompt="",
        text_type="",
        model="gpt-4o",
        summary_type="LLM",
        emb_type="openai",
        top_k=5,
        sentence_with_embeds=(),
    ):
        return closest_to_mean(list(texts_), embed)

    sc.get_embeddings = fake_get_embeddings
    sc.summarize_cluster = fake_summarize_cluster


def _run_case(sc, texts, X, **kwargs) -> dict:
    defaults = dict(
        init="k-means++",
        force_context_length=0,
        emb_type="fake",
        text_features=X,
        instructor_prompt="",
        model="fake",
        summary_type="LLM",
        top_k=5,
        text_sentences=None,
    )
    defaults.update(kwargs)
    assignments, summaries, summary_embeddings, centroids, summaries_evolution, centroids_evolution = (
        sc.kSummariesmeans(list(texts), N_CLUSTERS, **defaults)
    )
    return {
        "assignments": np.asarray(assignments),
        "centroids": np.asarray(centroids, dtype=np.float32),
        "summary_embeddings": np.asarray(summary_embeddings, dtype=np.float32),
        "summaries": np.array(list(summaries), dtype=object),
        "n_summaries_evolution": np.array(len(summaries_evolution)),
        "n_centroids_evolution": np.array(len(centroids_evolution)),
    }


def _make_synthetic_fixtures(sc, texts, X, out_dir: Path) -> None:
    for initial_iter in (True, False):
        for final_iter in (True, False):
            result = _run_case(
                sc,
                texts,
                X,
                initial_iter=initial_iter,
                final_iter=final_iter,
                max_llm_iter=3,
                max_iter=30,
                tol=-1.0,
                random_state=0,
            )
            name = f"synthetic_i{int(initial_iter)}_f{int(final_iter)}.npz"
            np.savez(out_dir / name, **result)
            print("wrote", name)

    # tol-break case: must actually converge before exhausting max_llm_iter.
    result = _run_case(
        sc, texts, X, initial_iter=True, final_iter=True, max_llm_iter=10, max_iter=100,
        tol=1e-4, random_state=0,
    )
    converged = int(result["n_centroids_evolution"]) < int(result["n_summaries_evolution"])
    assert converged, "tol-break fixture did not actually converge early -- adjust the synthetic setup"
    result["converged"] = np.array(converged)
    np.savez(out_dir / "synthetic_tol_break.npz", **result)
    print("wrote synthetic_tol_break.npz (converged =", converged, ")")

    # force_context_length case: exercises the kmeans++ member-sampling path.
    result = _run_case(
        sc, texts, X, initial_iter=True, final_iter=True, max_llm_iter=3, max_iter=30,
        tol=-1.0, random_state=0, force_context_length=5,
    )
    np.savez(out_dir / "synthetic_force_context.npz", **result)
    print("wrote synthetic_force_context.npz")


def _run_minibatch_case(sc, texts, X, **kwargs) -> dict:
    # miniBatchSummarymeans calls `text_sentences.items()` unconditionally
    # (even for summary_type="LLM"), so it crashes without one -- see the
    # module docstring. The fake summarize_cluster ignores its contents.
    text_sentences = {i: [[texts[i], X[i]]] for i in range(len(texts))}
    defaults = dict(
        init="k-means++",
        force_context_length=0,
        emb_type="fake",
        text_features=X,
        model="fake",
        summary_type="LLM",
        top_k=5,
        text_sentences=text_sentences,
    )
    defaults.update(kwargs)
    summaries, centroids = sc.miniBatchSummarymeans(list(texts), N_CLUSTERS, **defaults)
    centroids = np.asarray(centroids, dtype=np.float64)
    lloyd1_assignments = sc.KMeans(n_clusters=N_CLUSTERS, init=centroids, max_iter=1).fit_predict(np.asarray(X))
    return {
        "centroids": centroids,
        "summaries": np.array(summaries, dtype=object),
        "lloyd1_assignments": np.asarray(lloyd1_assignments),
    }


def _make_minibatch_synthetic_fixture(sc, texts, X, out_dir: Path) -> None:
    """3 uneven batches (320 points / max_batch_size=120 -> 107/107/106)."""
    result = _run_minibatch_case(
        sc, texts, X, max_batch_size=120, max_llm_iter=3, max_iter=30, tol=-1.0, random_state=0
    )
    np.savez(out_dir / "minibatch_synthetic.npz", **result)
    print("wrote minibatch_synthetic.npz")


def _make_bank77_minibatch_fixture(upstream: Path, out_dir: Path) -> None:
    from sklearn.metrics import normalized_mutual_info_score

    # Fresh module instance, real (un-monkeypatched) get_embeddings/summarize_cluster.
    sc = _load_module("upstream_summaryCentroids_bank77_minibatch", upstream / "summaryCentroids.py")

    df = pd.read_csv(REPO_ROOT / "data" / "bank77_test.csv")  # cols: text, category
    categories = sorted(df["category"].unique())
    mapping = {c: i for i, c in enumerate(categories)}
    labels = np.array([mapping[c] for c in df["category"]])
    documents = list(df["text"])
    num_clusters = 77

    print("embedding bank77 with distilbert for the minibatch fixture...")
    X = np.asarray(sc.get_embeddings(documents, emb_type="distilbert", instructor_prompt=""))

    print("building the bank77 sentence corpus (spaCy + distilbert) for the minibatch fixture...")
    sents, index_map = sc.build_sentence_corpus(documents)
    sent_embeddings, _ = sc.embed_sentences(sents, emb_type="distilbert")
    text_sentences = {i: [] for i in range(len(documents))}
    for i in range(len(sents)):
        cur_index = index_map[i][0]
        text_sentences[cur_index].append([sents[i], sent_embeddings[i]])

    summaries, centroids = sc.miniBatchKNLPmeans(
        documents,
        num_clusters,
        max_batch_size=1000,
        init="k-means++",
        force_context_length=0,
        max_llm_iter=5,
        max_iter=100,
        tol=1e-4,
        random_state=0,
        emb_type="distilbert",
        text_features=X,
        nlp="lsa",
        top_k=5,
        text_sentences=text_sentences,
    )
    centroids = np.asarray(centroids, dtype=np.float64)
    lloyd1_assignments = sc.KMeans(n_clusters=num_clusters, init=centroids, max_iter=1).fit_predict(X)
    nmi = normalized_mutual_info_score(labels, lloyd1_assignments)
    print(f"bank77 minibatch lsa seed0: NMI={nmi:.4f}")

    np.savez(
        out_dir / "bank77_minibatch_lsa_seed0.npz",
        centroids=centroids,
        lloyd1_assignments=lloyd1_assignments,
        nmi=np.array(nmi),
    )
    print("wrote bank77_minibatch_lsa_seed0.npz")


def _make_bank77_fixture(upstream: Path, out_dir: Path) -> None:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # Fresh, un-monkeypatched module instance -- must not share state with the
    # synthetic-case module above, which has get_embeddings/summarize_cluster replaced.
    sc = _load_module("upstream_summaryCentroids_bank77", upstream / "summaryCentroids.py")

    df = pd.read_csv(REPO_ROOT / "data" / "bank77_test.csv")  # cols: text, category
    categories = sorted(df["category"].unique())
    mapping = {c: i for i, c in enumerate(categories)}
    labels = np.array([mapping[c] for c in df["category"]])
    documents = list(df["text"])
    num_clusters = 77

    print("embedding bank77 with distilbert (downloads the model on first run)...")
    X = np.asarray(sc.get_embeddings(documents, emb_type="distilbert", instructor_prompt=""))

    out = sc.kNLPmeans(
        documents,
        num_clusters,
        emb_type="distilbert",
        text_features=X,
        nlp="lsa",
        max_llm_iter=5,
        random_state=0,
    )
    assignments = np.asarray(out[0])

    nmi = normalized_mutual_info_score(labels, assignments)
    ari = adjusted_rand_score(labels, assignments)
    print(f"bank77 lsa seed0: NMI={nmi:.4f} ARI={ari:.4f}")

    np.savez(
        out_dir / "bank77_lsa_seed0.npz",
        assignments=assignments,
        nmi=np.array(nmi),
        ari=np.array(ari),
    )
    print("wrote bank77_lsa_seed0.npz")


def _make_bank77_grid_fixture(upstream: Path, out_dir: Path) -> None:
    """kNLPmeans over every `bank77_grid_settings()` combination, from the ORIGINAL,
    un-modified `summaryCentroids.py`. Backs `tests/test_bank77_grid.py`."""
    # Fresh, un-monkeypatched module instance -- must not share state with the
    # synthetic-case module above, which has get_embeddings/summarize_cluster replaced.
    sc = _load_module("upstream_summaryCentroids_bank77_grid", upstream / "summaryCentroids.py")

    df = pd.read_csv(REPO_ROOT / "data" / "bank77_test.csv")  # cols: text, category
    documents = list(df["text"])
    num_clusters = 77

    print("embedding bank77 with distilbert for the grid fixture...")
    X = np.asarray(sc.get_embeddings(documents, emb_type="distilbert", instructor_prompt=""))

    print("building the bank77 sentence corpus (spaCy + distilbert) for the grid fixture...")
    sents, index_map = sc.build_sentence_corpus(documents)
    sent_embeddings, _ = sc.embed_sentences(sents, emb_type="distilbert")
    text_sentences = {i: [] for i in range(len(documents))}
    for i in range(len(sents)):
        cur_index = index_map[i][0]
        text_sentences[cur_index].append([sents[i], sent_embeddings[i]])

    result: dict[str, np.ndarray] = {}
    for method, top_k, max_llm_iter in bank77_grid_settings():
        print(f"  kNLPmeans(nlp={method!r}, top_k={top_k}, max_llm_iter={max_llm_iter})...")
        assignments, summaries, summary_embeddings, *_ = sc.kNLPmeans(
            documents,
            num_clusters,
            force_context_length=0,
            max_llm_iter=max_llm_iter,
            max_iter=120,
            tol=1e-4,
            random_state=0,
            emb_type="distilbert",
            text_features=X,
            nlp=method,
            top_k=top_k,
            text_sentences=text_sentences,
        )
        key = bank77_grid_key(method, top_k, max_llm_iter)
        result[f"{key}_assignments"] = np.asarray(assignments, dtype=np.int16)
        result[f"{key}_summaries"] = np.array(list(summaries), dtype=object)
        result[f"{key}_summary_embeddings"] = np.asarray(summary_embeddings, dtype=np.float32)

    np.savez(out_dir / "bank77_grid_seed0.npz", **result)
    print("wrote bank77_grid_seed0.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, help="path to the summaryCentroids repo root")
    parser.add_argument(
        "--skip-bank77", action="store_true", help="skip the (slow) single-batch bank77 fixture"
    )
    parser.add_argument(
        "--skip-bank77-minibatch", action="store_true", help="skip the (slow) minibatch bank77 fixture"
    )
    parser.add_argument(
        "--bank77-grid",
        action="store_true",
        help="also (re)generate the bank77 kNLPmeans grid reproduction fixture (slow, off by "
        "default; never touches the other fixtures)",
    )
    args = parser.parse_args()
    upstream = Path(args.upstream).resolve()

    sc = _load_module("upstream_summaryCentroids_synthetic", upstream / "summaryCentroids.py")
    texts, X = make_synthetic_data()
    _install_fakes(sc, texts, X)
    _make_synthetic_fixtures(sc, texts, X, HERE)

    mb_texts, mb_X = make_synthetic_minibatch_data()
    _install_fakes(sc, mb_texts, mb_X)
    _make_minibatch_synthetic_fixture(sc, mb_texts, mb_X, HERE)

    if not args.skip_bank77:
        _make_bank77_fixture(upstream, HERE)
    if not args.skip_bank77_minibatch:
        _make_bank77_minibatch_fixture(upstream, HERE)
    if args.bank77_grid:
        _make_bank77_grid_fixture(upstream, HERE)


if __name__ == "__main__":
    main()
