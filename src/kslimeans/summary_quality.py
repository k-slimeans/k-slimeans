"""Summary quality: how well a cluster's summary stands for its members, scored apart from how well it reads.

Four score families per (summary step, cluster), never merged into one number (issue #34):

1. **Fluency proxies** (`fluency_checks`): cheap surface checks that say nothing about content.
2. **Coverage**: how much of the cluster the summary stands for, measured on the embeddings.
3. **Genericity**: whether the summary could describe a neighbouring cluster as well.
4. **Unsupported information**: content the members never say.

Inputs of step `j`. The members of cluster `c` are the documents the summary was *written from*, i.e.
`RunRecord.source_partition(j)`, not the partition after the step's Lloyd block (`step_partitions` gives both).
Runs logged before #30 have only their final assignments; for them the final summaries are scored against the final
partition, which is an approximation (the final summaries were written from the partition before the last Lloyd
block, which is not logged), flagged by `StepPartition.approx`.

Definitions (all thresholds are module constants so reports can quote them)

- *Distances and similarities.* Embeddings are the run's document embeddings `X` and the re-embedded summaries `S`.
  Coverage uses Euclidean distance, the metric of the assignment step the replacement feeds; the specificity
  margin and the summary-to-summary similarities use cosine. A cluster's "mean" is the mean of its members' rows
  of `X` (k-means' centroid).
- *Duplicates and ties.* Clusters whose summaries are identical text share one embedding and tie. The assignment step
  gives all the tied members to the lowest cluster index, so the first copy of a group of exact duplicates is its
  representative: it is the only copy that can win members (nearest-summary share), and the later copies are left
  out when it is compared with the other summaries (intent coverage). Later copies therefore score 0 on both.
- *Nearest-summary share* (coverage a): the share of the cluster's members whose nearest of the `k` summary
  embeddings (duplicates counted once, see above) is this cluster's.
- *Intent coverage* (coverage b): a gold intent present in the cluster is covered when its members, on average, are
  strictly closer to this summary than to every other summary (later copies of its own duplicates excluded). `coverage_intents` is the share of the cluster's
  major intents (at least `MAJOR_INTENT_SHARE` of its members) that are covered, `NaN` if none is major.
  `lost_uncovered` below counts members of any present intent that is not covered.
- *Specificity margin*: cosine(summary, own members' mean) minus the highest cosine(summary, another non-empty
  cluster's members' mean). Non-positive means a neighbour's members fit the summary at least as well.
- *Duplicates*: `flag_duplicate` is an exact text match with another cluster's summary at the same step (the
  `surface.duplicate_groups` rule); `flag_near_duplicate` is a summary embedding with cosine at least
  `NEAR_DUPLICATE_COS` to another cluster's. `flag_mentions` is `surface.mentions_cluster`.
  `flag_generic` is the union of duplicate, near-duplicate, mentions and a non-positive margin.
- *Method-level genericity* (#7): the mean off-diagonal cosine among the step's `k` summary embeddings; the same
  number for the members' means is given as a reference.
- *Unsupported information.* Content words are spaCy (`en_core_web_sm`) lemmas whose part of speech is NOUN, VERB,
  ADJ or PROPN, alphabetic, longer than one letter, neither a stop word nor with a stop-word lemma. A lemma is
  supported when it is the lemma of a content word of some member. Members are lemmatized one text at a time and
  summaries one line at a time, with the same pipeline, so an extractive summary is analysed exactly like the member
  sentences it is made of. Numbers are the summary's tokens containing a digit; named entities are the entities spaCy labels
  PERSON, ORG, GPE, LOC, FAC, NORP, PRODUCT, EVENT, WORK_OF_ART, LAW or LANGUAGE. A number or entity is supported when
  its normalised token sequence occurs in the normalised text of some member. `flag_unsupported`: the
  unsupported share of lemmas is at least `UNSUPPORTED_LEMMA_SHARE`, or any number or entity is unsupported.
  Paraphrase counts as unsupported here (a member saying "cancel" does not support "terminate"), so the share is
  an upper bound on invented content; the hand check measures how loose it is.
- *Fluency proxies.* Not asking a question where one was asked for (`asks_question`; false for extractive
  summaries); echoing the prompt (sharing a word 3-gram with the prompt that no member contains); copying a member
  verbatim (equal after normalisation); stray markup (`<...>` tags, `*`, `#`, backticks, a line break in a summary
  meant to be one question, a summary wrapped in or opening with a double quote, or with an odd number of them);
  letters outside the Latin script
  (`flu_script`, for example CJK characters a model slipped into an English question); and being empty.
  `flag_not_fluent` is the union.
- *Low coverage*: `flag_low_coverage` is a nearest-summary share below `LOW_COVERAGE_BELOW`.

Rows are produced for non-empty clusters with a summary; empty clusters and `None` summaries get none.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from spacy.lang.en.stop_words import STOP_WORDS

from .runlog import RunRecord, members_by_cluster
from .surface import duplicate_groups, mentions_cluster

NEAR_DUPLICATE_COS = 0.95
MAJOR_INTENT_SHARE = 0.10
LOW_COVERAGE_BELOW = 0.5
UNSUPPORTED_LEMMA_SHARE = 0.5
PROMPT_NGRAM = 3

CONTENT_POS = frozenset({"NOUN", "VERB", "ADJ", "PROPN"})
NAMED_ENTITY_LABELS = frozenset(
    {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"}
)

_WORD = re.compile(r"[^\W_]+")
_MARKUP = re.compile(r"<[^>]*>|[*#`]")
_NAN = float("nan")


def _has_non_latin_letters(text: str) -> bool:
    return any(ch.isalpha() and not unicodedata.name(ch, "").startswith("LATIN") for ch in text)


def normalize(text: str) -> str:
    """Lower-cased alphanumeric tokens joined by single spaces."""
    return " ".join(_WORD.findall(text.lower()))


def _ngrams(normalized: str, n: int) -> set[str]:
    words = normalized.split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _blob(norm_texts: Sequence[str]) -> str:
    """Normalised member texts joined so that `f" {phrase} " in blob` is a whole-token match inside one text."""
    return " | " + " | ".join(norm_texts) + " | "


def _in_blob(blob: str, phrase: str) -> bool:
    return bool(phrase) and f" {phrase} " in blob


# ------------------------------------------------------------------ text analysis


def load_nlp():
    """The spaCy pipeline used for lemmas and entities (`en_core_web_sm`, already a dependency of the repo)."""
    import spacy

    return spacy.load("en_core_web_sm")


def content_lemmas(doc) -> frozenset[str]:
    """Lower-cased content-word lemmas of a spaCy `doc` (see the module docstring)."""
    out = set()
    for token in doc:
        lemma = token.lemma_.lower()
        if (
            token.pos_ in CONTENT_POS
            and token.is_alpha
            and len(lemma) > 1
            and not token.is_stop
            and lemma not in STOP_WORDS
        ):
            out.add(lemma)
    return frozenset(out)


@dataclass(frozen=True)
class Corpus:
    """Per-document features shared by every step of a dataset: normalised text and member lemmas."""

    norm: list[str]
    lemmas: list[frozenset[str]]


def analyze_corpus(texts: Sequence[str], nlp, batch_size: int = 1000) -> Corpus:
    """Normalise and lemmatize every text once (parser and NER are skipped: they are not needed for members)."""
    docs = nlp.pipe(texts, batch_size=batch_size, disable=["parser", "ner"])
    return Corpus(norm=[normalize(t) for t in texts], lemmas=[content_lemmas(d) for d in docs])


# ------------------------------------------------------------------ fluency


def fluency_checks(
    summary: str,
    member_norm: Sequence[str],
    prompt: str,
    asks_question: bool,
    member_blob: str | None = None,
) -> dict[str, bool]:
    """Cheap well-formedness checks; `True` means the check failed. They say nothing about the content."""
    blob = member_blob if member_blob is not None else _blob(member_norm)
    norm = normalize(summary)
    echoed = any(f" {g} " not in blob for g in _ngrams(norm, PROMPT_NGRAM) & _ngrams(normalize(prompt), PROMPT_NGRAM))
    stripped = summary.strip()
    quotes = stripped.count('"')
    return {
        "flu_empty": not stripped,
        "flu_no_question": asks_question and not stripped.rstrip("\"'”’ ").endswith("?"),
        "flu_echo": echoed,
        "flu_copy": bool(norm) and norm in set(member_norm),
        "flu_script": _has_non_latin_letters(summary),
        "flu_markup": bool(_MARKUP.search(summary))
        or (asks_question and "\n" in stripped)
        or quotes % 2 == 1
        or (quotes >= 2 and stripped.startswith('"') and stripped.endswith('"')),
    }


# ------------------------------------------------------------------ unsupported information


def summary_lines(summary: str | None) -> list[str]:
    """The non-blank lines of a summary. Each is analysed on its own, as each member is, so that a multi-sentence
    extractive summary is lemmatized exactly like the member sentences it is made of."""
    return [line for line in (summary or "").split("\n") if line.strip()]


def unsupported_content(docs: Sequence, member_lemmas: frozenset[str] | set[str], member_blob: str) -> dict:
    """Content of a summary that no member supports (see the module docstring).

    `docs` are the spaCy docs of the summary's lines (`summary_lines`), analysed with the same settings as the members
    except that NER is on.
    """
    lemmas = frozenset().union(*(content_lemmas(d) for d in docs))
    numbers = sorted({normalize(t.text) for d in docs for t in d if any(ch.isdigit() for ch in t.text)} - {""})
    entities = sorted(
        {normalize(e.text) for d in docs for e in d.ents if e.label_ in NAMED_ENTITY_LABELS} - {""}
    )
    unsupported_lemmas = sorted(lemmas - set(member_lemmas))
    unsupported_numbers = [n for n in numbers if not _in_blob(member_blob, n)]
    unsupported_entities = [e for e in entities if not _in_blob(member_blob, e)]
    share = len(unsupported_lemmas) / len(lemmas) if lemmas else _NAN
    return {
        "n_content_lemmas": len(lemmas),
        "unsupported_lemma_share": share,
        "unsupported_lemmas": " ".join(unsupported_lemmas),
        "n_unsupported_numbers": len(unsupported_numbers),
        "n_unsupported_entities": len(unsupported_entities),
        "unsupported_items": " ".join(unsupported_numbers + unsupported_entities),
        "flag_unsupported": bool(
            (share >= UNSUPPORTED_LEMMA_SHARE) or unsupported_numbers or unsupported_entities
        ),
    }


def method_label(config: dict, method: str) -> str:
    """`k-NLPmeans` for extractive runs, `k-SLMmeans (Qwen3-4B)` for model runs: the method name alone is shared."""
    summarizer = config["summarizer"]
    model = summarizer.get("model_id")
    return f"{method} ({str(model).rsplit('/', 1)[-1]})" if model else method


# ------------------------------------------------------------------ partitions of a logged run


@dataclass(frozen=True)
class StepPartition:
    """The partitions around summary step `step` of a record."""

    step: int
    source_step: int | None
    """Trace step the summary was written from (`None` for an old run without per-step assignments)."""
    labels: np.ndarray
    """The members the summary was written from (the final assignments, if `approx`)."""
    after: np.ndarray | None
    """Labels after the step's Lloyd block, or `None` (old run, or a final step that stopped on `tol`)."""
    approx: bool
    """True when `labels` is the final partition standing in for the one the summary was written from."""


def step_partitions(record: RunRecord, final_only: bool = False) -> Iterator[StepPartition]:
    """Per summary step of `record`, its source and after partitions.

    A record without per-step assignments yields only its last step, scored against the final assignments and
    marked `approx`.
    """
    n = record.n_summary_steps
    if n == 0:
        return
    if record.assignments_evolution is None or record.summary_source_steps is None:
        if record.assignments is not None:
            yield StepPartition(n - 1, None, record.assignments, None, True)
        return
    for j in [n - 1] if final_only else range(n):
        try:
            after = record.partition_after(j)
        except IndexError:
            after = None
        yield StepPartition(j, record.summary_source_steps[j], record.source_partition(j), after, False)


# ------------------------------------------------------------------ cluster overlap with gold intents


def cluster_overlap(labels: np.ndarray, gold: np.ndarray, k: int) -> list[dict]:
    """Per cluster, its majority gold intent, purity and the Jaccard overlap with that intent.

    The majority intent is the one with the most members in the cluster (lowest id on ties), the max-count rule of
    `docs/results/prelim_bank77.md` section 6. Empty clusters give `None` entries.
    """
    gold = np.asarray(gold)
    class_sizes = np.bincount(gold, minlength=int(gold.max()) + 1)
    out: list[dict] = []
    for idx in members_by_cluster(labels, k):
        if len(idx) == 0:
            out.append({})
            continue
        counts = np.bincount(gold[idx], minlength=len(class_sizes))
        g = int(counts.argmax())
        n = int(counts[g])
        out.append(
            {
                "majority_intent": g,
                "purity": n / len(idx),
                "jaccard_majority": n / (len(idx) + int(class_sizes[g]) - n),
            }
        )
    return out


# ------------------------------------------------------------------ the per-step scores


def _unit(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.where(norms == 0, 1.0, norms)


def mean_off_diagonal(sim: np.ndarray) -> float:
    """Mean of the off-diagonal entries of a square matrix (`NaN` below 2 rows)."""
    n = len(sim)
    if n < 2:
        return _NAN
    return float((sim.sum() - np.trace(sim)) / (n * (n - 1)))


def label_alignment(S: np.ndarray, L: np.ndarray, majority: Sequence[int | None]) -> dict:
    """#8's label alignment at one step: Hungarian-match summaries to gold label names by cosine.

    `S` are the k summary embeddings, `L` the label-name embeddings, `majority[c]` the cluster's majority label
    (`None` for an empty cluster, which is left out). Returns `label_align_sim`, the mean similarity of the matched
    pairs, and `label_align_acc`, the share of matched non-empty clusters whose matched label is their majority.
    """
    sim = _unit(S) @ _unit(L).T
    rows, cols = linear_sum_assignment(-sim)
    pairs = [(r, c) for r, c in zip(rows, cols) if majority[r] is not None]
    if not pairs:
        return {"label_align_sim": _NAN, "label_align_acc": _NAN}
    return {
        "label_align_sim": float(np.mean([sim[r, c] for r, c in pairs])),
        "label_align_acc": float(np.mean([majority[r] == c for r, c in pairs])),
    }


def generic_flags(summaries: Sequence[str | None], S: np.ndarray, X: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Per-cluster boolean arrays `duplicate`, `near_duplicate`, `mentions`, `low_margin` and `generic` of one step.

    The genericity flags of `score_step` (the module docstring defines them) without its spaCy and coverage work, for
    callers that need only the flags (#47). `S[c]` is cluster `c`'s summary embedding, `labels` the partition the
    summaries were written from. Empty clusters and clusters without a summary get no row in `score_step`, so
    every flag is False for them here.
    """
    k = len(summaries)
    members = members_by_cluster(np.asarray(labels), k)
    nonempty = np.array([len(m) > 0 for m in members])
    has_summary = np.array([s is not None for s in summaries])
    means = np.vstack([X[m].mean(axis=0) if len(m) else np.zeros(X.shape[1]) for m in members])
    Sn, Mn = _unit(S), _unit(means)
    own_sim = np.einsum("ij,ij->i", Sn, Mn)
    to_means = Sn @ Mn.T
    np.fill_diagonal(to_means, -np.inf)
    to_means[:, ~nonempty] = -np.inf
    best_other_mean = to_means.max(axis=1)
    other_sum = Sn @ Sn.T
    np.fill_diagonal(other_sum, -np.inf)
    other_sum[:, ~has_summary] = -np.inf
    max_other_summary = other_sum.max(axis=1)
    margin = np.where(np.isfinite(best_other_mean), own_sim - best_other_mean, np.nan)
    scored = nonempty & has_summary
    duplicate = np.zeros(k, dtype=bool)
    for group in duplicate_groups(list(summaries)):
        duplicate[group] = True
    flags = {
        "duplicate": duplicate,
        "near_duplicate": max_other_summary >= NEAR_DUPLICATE_COS,
        "mentions": np.array([mentions_cluster(s) for s in summaries], dtype=bool),
        "low_margin": margin <= 0,
    }
    flags = {name: flag & scored for name, flag in flags.items()}
    flags["generic"] = np.logical_or.reduce(list(flags.values()))
    return flags


def score_step(
    summaries: Sequence[str | None],
    S: np.ndarray,
    X: np.ndarray,
    labels: np.ndarray,
    gold: np.ndarray,
    corpus: Corpus,
    nlp,
    *,
    prompt: str,
    asks_question: bool,
    after: np.ndarray | None = None,
    label_emb: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    """Score every summary of one step. Returns `(cluster_rows, step_row)`.

    `summaries[c]`, `S[c]` are cluster `c`'s summary and its embedding, `labels` the partition the summaries were
    written from, `gold` the gold intents, `X` the document embeddings and `corpus` the per-document features.
    With `after` (the partition after the step's Lloyd block) each row also counts members kept, lost and gained.
    With `label_emb` (embedded gold label names) rows and the step row carry #8's label alignment.
    """
    k = len(summaries)
    labels, gold = np.asarray(labels), np.asarray(gold)
    members = members_by_cluster(labels, k)
    nonempty = np.array([len(m) > 0 for m in members])
    has_summary = np.array([s is not None for s in summaries])
    overlap = cluster_overlap(labels, gold, k)

    # Cosine geometry: summaries against each other and against the members' means.
    means = np.vstack([X[m].mean(axis=0) if len(m) else np.zeros(X.shape[1]) for m in members])
    Sn, Mn = _unit(S), _unit(means)
    sum_sim = Sn @ Sn.T
    own_sim = np.einsum("ij,ij->i", Sn, Mn)
    to_means = Sn @ Mn.T
    np.fill_diagonal(to_means, -np.inf)
    to_means[:, ~nonempty] = -np.inf
    best_other_mean = to_means.max(axis=1)
    other_sum = sum_sim.copy()
    np.fill_diagonal(other_sum, -np.inf)
    other_sum[:, ~has_summary] = -np.inf
    max_other_summary = other_sum.max(axis=1)

    # Euclidean distances to the summaries: what the assignment step after the replacement sees.
    d2 = (X**2).sum(1)[:, None] - 2 * X @ S.T + (S**2).sum(1)[None, :]
    d2 = np.maximum(d2, 0.0)
    dist = np.sqrt(d2)
    groups = duplicate_groups(list(summaries))
    exact_dup = {i for group in groups for i in group}
    later_copies = {c: group[i + 1 :] for group in groups for i, c in enumerate(group)}
    not_representative = [c for group in groups for c in group[1:]]
    d2_rep = d2.copy()
    d2_rep[:, not_representative] = np.inf
    nearest = d2_rep.argmin(axis=1)
    label_sim = (Sn @ _unit(label_emb).T) if label_emb is not None else None
    lines = [summary_lines(s) for s in summaries]
    flat = list(nlp.pipe([line for ls in lines for line in ls], disable=["parser"]))
    offsets = np.cumsum([0] + [len(ls) for ls in lines])
    docs = [flat[offsets[c] : offsets[c + 1]] for c in range(k)]

    rows: list[dict] = []
    for c, idx in enumerate(members):
        if not nonempty[c] or summaries[c] is None:
            continue
        summary = summaries[c]
        member_norm = [corpus.norm[i] for i in idx]
        blob = _blob(member_norm)
        member_lemmas = set().union(*(corpus.lemmas[i] for i in idx))
        row: dict = {
            "cluster": c,
            "summary": summary,
            "n_members": len(idx),
            **overlap[c],
        }
        # Coverage
        share = float(np.mean(nearest[idx] == c))
        intents, counts = np.unique(gold[idx], return_counts=True)
        covered = {}
        for g in intents:
            mean_d = dist[idx[gold[idx] == g]].mean(axis=0)
            compared = np.ones(k, dtype=bool)
            compared[[c, *later_copies.get(c, [])]] = False
            covered[int(g)] = bool(not compared.any() or mean_d[c] < mean_d[compared].min())
        major = [int(g) for g, n in zip(intents, counts) if n >= MAJOR_INTENT_SHARE * len(idx)]
        row.update(
            coverage_nearest=share,
            n_intents=len(intents),
            n_major_intents=len(major),
            n_major_covered=sum(covered[g] for g in major),
            coverage_intents=(sum(covered[g] for g in major) / len(major)) if major else _NAN,
            flag_low_coverage=share < LOW_COVERAGE_BELOW,
        )
        # Genericity
        margin = float(own_sim[c] - best_other_mean[c]) if np.isfinite(best_other_mean[c]) else _NAN
        max_cos = float(max_other_summary[c]) if np.isfinite(max_other_summary[c]) else _NAN
        row.update(
            specificity_margin=margin,
            max_summary_cos=max_cos,
            flag_duplicate=c in exact_dup,
            flag_near_duplicate=bool(max_cos >= NEAR_DUPLICATE_COS),
            flag_mentions=mentions_cluster(summary),
            flag_low_margin=bool(margin <= 0),
        )
        row["flag_generic"] = bool(
            row["flag_duplicate"] or row["flag_near_duplicate"] or row["flag_mentions"] or row["flag_low_margin"]
        )
        # Unsupported information
        row.update(unsupported_content(docs[c], member_lemmas, blob))
        # Fluency
        flu = fluency_checks(summary, member_norm, prompt, asks_question, blob)
        row.update(flu)
        row["flag_not_fluent"] = any(flu.values())
        # Label alignment (#8), per cluster
        if label_sim is not None and "majority_intent" in row:
            g = row["majority_intent"]
            row["label_sim_majority"] = float(label_sim[c, g])
            row["label_top1"] = bool(label_sim[c].argmax() == g)
        # Link to the assignments
        if after is not None:
            kept = int(np.sum(after[idx] == c))
            uncovered = np.array([not covered[int(g)] for g in gold[idx]])
            lost_mask = after[idx] != c
            after_idx = np.flatnonzero(after == c)
            row.update(
                n_after=len(after_idx),
                purity_after=(np.bincount(gold[after_idx]).max() / len(after_idx)) if len(after_idx) else _NAN,
                n_kept=kept,
                n_lost=len(idx) - kept,
                n_gained=int(np.sum((after == c) & (labels != c))),
                n_members_uncovered=int(uncovered.sum()),
                n_lost_uncovered=int(np.sum(uncovered & lost_mask)),
            )
        rows.append(row)

    step = {
        "k": k,
        "n_empty_clusters": int((~nonempty).sum()),
        "n_summarized": int(has_summary.sum()),
        "genericity": mean_off_diagonal(sum_sim[np.ix_(has_summary, has_summary)]),
        "genericity_centroids": mean_off_diagonal((Mn @ Mn.T)[np.ix_(nonempty, nonempty)]),
        "summary_inertia_replayed": float(d2.min(axis=1).sum()),
    }
    if label_emb is not None:
        majority = [o.get("majority_intent") for o in overlap]
        step.update(label_alignment(S, label_emb, majority))
    return rows, step


# ------------------------------------------------------------------ aggregation

SCORE_COLUMNS = (
    "coverage_nearest",
    "coverage_intents",
    "specificity_margin",
    "max_summary_cos",
    "unsupported_lemma_share",
)
FLAG_COLUMNS = (
    "flag_not_fluent",
    "flu_no_question",
    "flu_echo",
    "flu_copy",
    "flu_markup",
    "flu_script",
    "flu_empty",
    "flag_low_coverage",
    "flag_generic",
    "flag_duplicate",
    "flag_near_duplicate",
    "flag_mentions",
    "flag_low_margin",
    "flag_unsupported",
)


def aggregate_scores(df, by: Sequence[str] = ("dataset", "method")):
    """Mean and median of the score columns and the share of each flag, per group of `by`, pooled over rows.

    `NaN` scores (for example intent coverage of a cluster with no major intent) are skipped. Returns a
    `DataFrame` indexed by `by` with `n` (rows), `<score>_mean`, `<score>_median` and `<flag>_share` columns.
    """
    grouped = df.groupby(list(by), sort=False)
    out = grouped.size().rename("n").to_frame()
    for col in SCORE_COLUMNS:
        if col in df:
            out[f"{col}_mean"] = grouped[col].mean()
            out[f"{col}_median"] = grouped[col].median()
    for col in FLAG_COLUMNS:
        if col in df:
            out[f"{col}_share"] = grouped[col].apply(lambda s: s.astype(float).mean())
    return out
