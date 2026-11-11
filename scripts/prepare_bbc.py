"""Build `data/bbc500.csv`: a stratified 500-article sample of the BBC News dataset.

The BBC News dataset (Greene & Cunningham, ICML 2006) has 2,225 articles in five topics (business,
entertainment, politics, sport, tech). The BBC owns the copyright in the articles, so the text is not in this
repository: `data/bbc500.csv` is generated and git-ignored, and only `data/bbc500_ids.csv` is committed. That
file lists the sampled articles (`file_id`, `topic`, in sample order, e.g. `business/001`, the article's path in
the archive) and, in its first line, the SHA-256 of the CSV they build.

    python scripts/prepare_bbc.py                  # download the archive, rebuild the CSV from the ids, verify it
    python scripts/prepare_bbc.py --zip PATH       # the same from a local bbc-fulltext.zip
    python scripts/prepare_bbc.py --resample       # redraw the ids (dedup, then the stratified sample) and
                                                   # rewrite data/bbc500_ids.csv with the new checksum

The CSV has `text`, `topic` and `file_id`. Each article's `text` is its headline and body on one line: the
headline gets a final full stop unless it already ends in punctuation, and all whitespace runs, paragraph
breaks included, become single spaces. One line per article keeps the member boundaries clear when a prompt
lists the articles one per line.

The sample takes `--per-topic` articles of each topic (100 by default, so 500) with a fixed seed and shuffles
them. The corpus contains duplicate and near-duplicate articles (the same story filed twice, or with a few
words edited, e.g. "Grand Slam" vs "grand slam"). They are removed from all 2,225 articles before sampling, so
they cannot distort the benchmark or the TF-IDF document frequencies. Articles are visited in sorted
`file_id` order and one is dropped when it matches an article already kept, so the first copy survives. It
matches when either
- its TF-IDF cosine similarity to it (English stop words removed, vectorizer fit on the full corpus) is at
  least `--dedup-threshold` (0.85), which catches edited and re-written copies of one story; or
- its first 200 characters are the same after lowercasing and collapsing non-word characters, which catches
  a story filed twice where one copy continues with extra material (same headline and lead, cosine ~0.76).

The cosine threshold sits in a gap: 0.95 would leave re-edited copies of one story in (cosine about 0.88),
while below 0.85 the matches are different stories on the same event. Exact match alone finds 103
duplicates of 2,225, so the two rules together are a superset of it.

The archive is read in place; nothing is extracted. Only members named `bbc/<topic>/<NNN>.txt` are read.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

URL = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
TOPICS = ("business", "entertainment", "politics", "sport", "tech")
EXPECTED_ARTICLES = 2225
DEDUP_THRESHOLD = 0.85
PREFIX_CHARS = 200
SEED = 0
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "bbc500.csv"
IDS_PATH = DATA_DIR / "bbc500_ids.csv"
_SHA_PREFIX = "# sha256 of bbc500.csv: "
_MEMBER = re.compile(rf"bbc/({'|'.join(TOPICS)})/(\d{{3}})\.txt")
_MAX_MEMBER_BYTES = 1_000_000  # the longest article is ~30 KB; refuse anything absurd


def format_article(raw: str) -> str:
    """Headline and body on one line (see the module docstring)."""
    headline, _, body = raw.strip().partition("\n")
    headline = " ".join(headline.split())
    if headline and headline[-1] not in ".!?:;\"'":
        headline += "."
    return f"{headline} {' '.join(body.split())}".strip()


def read_archive(path: str | Path) -> pd.DataFrame:
    """Every article of the archive as a row of `text`, `topic`, `file_id`, sorted by `file_id`."""
    rows = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            match = _MEMBER.fullmatch(info.filename)
            if match is None:
                continue
            if info.file_size > _MAX_MEMBER_BYTES:
                raise ValueError(f"{info.filename} is {info.file_size} bytes, refusing to read it")
            topic, number = match.groups()
            text = format_article(archive.read(info).decode("utf-8"))
            rows.append({"text": text, "topic": topic, "file_id": f"{topic}/{number}"})
    df = pd.DataFrame(rows, columns=["text", "topic", "file_id"]).sort_values("file_id", ignore_index=True)
    if len(df) != EXPECTED_ARTICLES:
        raise ValueError(f"Expected {EXPECTED_ARTICLES} articles in {path}, found {len(df)}")
    return df


def _prefix(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()[:PREFIX_CHARS]


def deduplicate(df: pd.DataFrame, threshold: float = DEDUP_THRESHOLD) -> pd.DataFrame:
    """`df` without the articles that match an earlier kept article (sorted `file_id` order): TF-IDF cosine
    of at least `threshold`, or the same normalized first `PREFIX_CHARS` characters. Deterministic; the
    first copy of each story survives."""
    df = df.sort_values("file_id", ignore_index=True)
    vectors = TfidfVectorizer(stop_words="english").fit_transform(df["text"])
    similarity = (vectors @ vectors.T).toarray()
    prefixes = df["text"].map(_prefix)
    kept: list[int] = []
    kept_prefixes: set[str] = set()
    for i in range(len(df)):
        if prefixes[i] in kept_prefixes or (kept and similarity[i, kept].max() >= threshold):
            continue
        kept.append(i)
        kept_prefixes.add(prefixes[i])
    return df.iloc[kept].reset_index(drop=True)


def sample(df: pd.DataFrame, per_topic: int = 100, seed: int = SEED) -> pd.DataFrame:
    """`per_topic` articles of each topic, drawn with `seed` and shuffled so rows are not grouped by topic."""
    rng = np.random.default_rng(seed)
    parts = []
    for topic in TOPICS:
        pool = df[df["topic"] == topic].sort_values("file_id")
        if len(pool) < per_topic:
            raise ValueError(f"Topic '{topic}' has {len(pool)} articles, need {per_topic}")
        parts.append(pool.iloc[np.sort(rng.choice(len(pool), size=per_topic, replace=False))])
    chosen = pd.concat(parts)
    return chosen.iloc[rng.permutation(len(chosen))].reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: str | Path) -> str:
    """Write the sample (`text`, `topic`, `file_id`) and return the SHA-256 of the file."""
    df[["text", "topic", "file_id"]].to_csv(path, index=False, lineterminator="\n")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_ids(df: pd.DataFrame, sha256: str, path: str | Path) -> None:
    """The committed ids file: a checksum comment line, then `file_id,topic` in sample order."""
    with open(path, "w", newline="") as f:
        f.write(f"{_SHA_PREFIX}{sha256}\n")
        df[["file_id", "topic"]].to_csv(f, index=False, lineterminator="\n")


def read_ids(path: str | Path) -> tuple[pd.DataFrame, str]:
    """The ids file's `file_id`/`topic` rows (sample order) and its recorded checksum."""
    first = Path(path).read_text(encoding="utf-8").split("\n", 1)[0]
    if not first.startswith(_SHA_PREFIX):
        raise ValueError(f"{path} must start with the line '{_SHA_PREFIX}<hex>'")
    return pd.read_csv(path, comment="#"), first[len(_SHA_PREFIX) :].strip()


def build_from_ids(articles: pd.DataFrame, ids: pd.DataFrame) -> pd.DataFrame:
    """The sample rows for `ids`, in `ids` order, from the archive's `articles`."""
    by_id = articles.set_index("file_id")
    missing = sorted(set(ids["file_id"]) - set(by_id.index))
    if missing:
        raise ValueError(f"{len(missing)} ids are not in the archive, e.g. {missing[:3]}")
    rows = by_id.loc[ids["file_id"]].reset_index()
    if not (rows["topic"].to_numpy() == ids["topic"].to_numpy()).all():
        raise ValueError("an id's topic differs between the ids file and the archive")
    return rows


def prepare(
    archive: str | Path,
    out: str | Path = OUT_PATH,
    ids_path: str | Path = IDS_PATH,
    resample: bool = False,
    per_topic: int = 100,
    seed: int = SEED,
    threshold: float = DEDUP_THRESHOLD,
) -> str:
    """Write the CSV and return its SHA-256.

    By default the sample is rebuilt from `ids_path` and the checksum must match the one recorded there
    (a mismatch raises, with the CSV left in place for inspection). With `resample` the ids are redrawn
    (dedup, then the stratified sample) and `ids_path` is rewritten with the new checksum.
    """
    articles = read_archive(archive)
    if resample:
        unique = deduplicate(articles, threshold)
        removed = articles["topic"].value_counts() - unique["topic"].value_counts()
        print(f"Removed {len(articles) - len(unique)} duplicate articles: {removed.sort_index().to_dict()}")
        df = sample(unique, per_topic, seed)
        sha256 = write_csv(df, out)
        write_ids(df, sha256, ids_path)
        return sha256
    ids, expected = read_ids(ids_path)
    sha256 = write_csv(build_from_ids(articles, ids), out)
    if sha256 != expected:
        raise ValueError(
            f"{out} has SHA-256 {sha256}, but {ids_path} records {expected}; the archive or the pandas version "
            "differs from the one the ids were made with"
        )
    return sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", help="path to bbc-fulltext.zip; downloaded from the original host if omitted")
    parser.add_argument("--resample", action="store_true", help="redraw the ids and rewrite the ids file")
    parser.add_argument("--per-topic", type=int, default=100, help="with --resample: articles per topic (100)")
    parser.add_argument("--seed", type=int, default=SEED, help=f"with --resample: sampling seed ({SEED})")
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=DEDUP_THRESHOLD,
        help="with --resample: TF-IDF cosine at which an article counts as a duplicate",
    )
    parser.add_argument("--out", default=str(OUT_PATH), help="output CSV (default data/bbc500.csv)")
    parser.add_argument("--ids", default=str(IDS_PATH), help="ids file (default data/bbc500_ids.csv)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        archive = args.zip
        if archive is None:
            archive = str(Path(tmp) / "bbc-fulltext.zip")
            print(f"Downloading {URL}")
            urllib.request.urlretrieve(URL, archive)
        sha256 = prepare(
            archive, args.out, args.ids, args.resample, args.per_topic, args.seed, args.dedup_threshold
        )
    print(f"Wrote {args.out} (SHA-256 {sha256})")


if __name__ == "__main__":
    main()
