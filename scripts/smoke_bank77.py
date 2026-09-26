"""First command to run in a fresh checkout: validates the environment end to end.

Loads Bank77, embeds it with distilbert, runs a k-means baseline and
k-NLPmeans(lsa) -- both deterministic, no model downloads or API keys
needed beyond distilbert/spaCy -- and compares NMI/ARI against known-good
reference values from a prior run on this project's dev machine. A few
bits of the embedding can differ across hardware and flip a handful of
cluster assignments, so an exact match isn't required, only a close one.

Usage:

    python scripts/smoke_bank77.py
"""

from __future__ import annotations

import sys

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from kslimeans.core import kNLPmeans
from kslimeans.data import load_dataset
from kslimeans.embeddings import get_embedder

# NMI/ARI from a known-good run of this exact pipeline on this project's dev machine.
REFERENCE = {
    "k-means": (0.6541, 0.3138),
    "k-NLPmeans": (0.6518, 0.2945),
}
CLOSE_TOL = 0.01


def _verdict(name: str, nmi: float, ari: float) -> str:
    ref_nmi, ref_ari = REFERENCE[name]
    if round(nmi, 4) == ref_nmi and round(ari, 4) == ref_ari:
        return "OK -- exact match with the reference run."
    if abs(nmi - ref_nmi) <= CLOSE_TOL and abs(ari - ref_ari) <= CLOSE_TOL:
        return (
            "OK -- close to the reference run (embeddings computed on different "
            "hardware can differ in the last bits and flip a few assignments)."
        )
    return (
        f"WARNING -- off from the reference run (expected NMI={ref_nmi}, ARI={ref_ari}); "
        "double-check your environment (embedder/model versions, data files)."
    )


def _report(name: str, y_true, y_pred) -> None:
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    ref_nmi, ref_ari = REFERENCE[name]
    print(f"{name:<12} NMI={nmi:.4f}  ARI={ari:.4f}   (reference: NMI={ref_nmi:.4f}  ARI={ref_ari:.4f})")
    print(f"  -> {_verdict(name, nmi, ari)}")


def main() -> int:
    print("Loading bank77...")
    ds = load_dataset("bank77")
    print(f"  {len(ds.texts)} documents, {ds.n_clusters} clusters")

    print("Embedding with distilbert (downloads the sentence-transformers model on first run)...")
    embedder = get_embedder("distilbert")
    X = embedder.encode(ds.texts)
    print(f"  embedding matrix: {X.shape}")

    print("\n--- baseline: k-means ---")
    km = KMeans(n_clusters=ds.n_clusters, init="k-means++", n_init=10, random_state=0).fit(X)
    _report("k-means", ds.labels, km.labels_)

    print("\n--- k-NLPmeans (lsa, deterministic -- no model download beyond distilbert/spaCy) ---")
    result = kNLPmeans(
        ds.texts,
        ds.n_clusters,
        emb_type="distilbert",
        text_features=X,
        nlp="lsa",
        max_llm_iter=5,
        random_state=0,
    )
    _report("k-NLPmeans", ds.labels, result.assignments)

    print("\nIf both lines above say OK, your environment is good to go.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
