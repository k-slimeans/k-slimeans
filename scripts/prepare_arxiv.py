"""Build the arXiv abstracts corpus used by the L4 / RQ2 scale test.

Source: the Hugging Face parquet mirror `librarian-bots/arxiv-metadata-snapshot`
(CC0), pinned to one revision. The pipeline has four resumable stages; each is
skipped when its output already exists and is complete, and writes into
`data/arxiv/` (git-ignored, or `$KSLIMEANS_DATA_DIR/arxiv/`):

1. `download`: the 11 raw parquet shards into `raw/`.
2. `filter`: stream the shards (only the needed columns) and write
   `arxiv.parquet` (id, text, label, archive, update_date). Drops withdrawn
   papers, abstracts under 50 words, and duplicate abstracts.
3. `embed`: MiniLM embeddings of `text` into `embeddings_minilm_fp16.npy`
   (float16 memmap), chunked with a `progress.json` checkpoint.
4. `subsample`: a seeded 200k random subsample, `subsample_200k.npy` and
   `arxiv_sub200k.parquet` (same columns plus `row`, the index into
   `arxiv.parquet` / the embeddings).

`manifest.json` records the source revision, the counts after each filter rule,
the label vocabulary, and the embedding and subsample settings.

Usage:

    python scripts/prepare_arxiv.py [--stages download filter embed subsample]
                                    [--force {filter,embed,subsample}] [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kslimeans.data import DATA_DIR

log = logging.getLogger("prepare_arxiv")

REPO_ID = "librarian-bots/arxiv-metadata-snapshot"
REVISION = "5733e66fde7e840bf1dcb2d9cffa0c64f4c5654d"
N_SHARDS = 11
LICENSE = "CC0-1.0"

MIN_WORDS = 50
WITHDRAWN_ABSTRACT_RE = r"(?i)\b(?:has been|is|was|been) withdrawn"
WITHDRAWN_COMMENTS_RE = r"^\W*(this )?(paper|article)?\s*(has been |is |was )?withdrawn"
_ABSTRACT_WITHDRAWN = re.compile(WITHDRAWN_ABSTRACT_RE)
_COMMENTS_WITHDRAWN = re.compile(WITHDRAWN_COMMENTS_RE, re.IGNORECASE)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
SUBSAMPLE_SIZE = 200_000
SUBSAMPLE_SEED = 0
BENCH_ROWS = 20_000
ROW_GROUP_ROWS = 100_000

STAGES = ("download", "filter", "embed", "subsample")
FORCEABLE = ("filter", "embed", "subsample")  # download keeps existing shards; delete raw/ to refetch
RAW_COLUMNS = ["id", "categories", "abstract", "comments", "update_date"]
OUT_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.string()),
        ("label", pa.string()),
        ("archive", pa.string()),
        ("update_date", pa.string()),
    ]
)

ARXIV_FILE = "arxiv.parquet"
EMBEDDINGS_FILE = "embeddings_minilm_fp16.npy"
PROGRESS_FILE = "progress.json"
SUBSAMPLE_FILE = "subsample_200k.npy"
SUBSAMPLE_PARQUET = "arxiv_sub200k.parquet"
MANIFEST_FILE = "manifest.json"

Encoder = Callable[[list[str]], np.ndarray]


def shard_names() -> list[str]:
    return [f"train-{i:05d}-of-{N_SHARDS:05d}.parquet" for i in range(N_SHARDS)]


# --- manifest ---------------------------------------------------------------


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def read_manifest(data_dir: Path) -> dict:
    path = data_dir / MANIFEST_FILE
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest.setdefault(
        "source",
        {"repo_id": REPO_ID, "revision": REVISION, "license": LICENSE, "shards": shard_names()},
    )
    return manifest


def update_manifest(data_dir: Path, **sections: object) -> dict:
    manifest = read_manifest(data_dir)
    for key, value in sections.items():
        if isinstance(value, dict) and isinstance(manifest.get(key), dict):
            manifest[key] = {**manifest[key], **value}
        else:
            manifest[key] = value
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_json(data_dir / MANIFEST_FILE, manifest)
    return manifest


# --- download ---------------------------------------------------------------


def download(data_dir: Path) -> list[Path]:
    """Fetch the raw shards at the pinned revision; shards already present are kept."""
    from huggingface_hub import hf_hub_download

    raw = data_dir / "raw"
    paths = []
    for name in shard_names():
        path = raw / "data" / name
        if not path.exists():
            log.info("downloading %s", name)
            hf_hub_download(
                repo_id=REPO_ID,
                filename=f"data/{name}",
                repo_type="dataset",
                revision=REVISION,
                local_dir=raw,
            )
        paths.append(path)
    update_manifest(data_dir)
    return paths


# --- filter -----------------------------------------------------------------


def is_withdrawn(abstract: str, comments: str) -> bool:
    return bool(_ABSTRACT_WITHDRAWN.search(abstract) or _COMMENTS_WITHDRAWN.search(comments))


def abstract_digest(text: str) -> bytes:
    return hashlib.blake2b(text.lower().encode("utf-8"), digest_size=12).digest()


def _shards(data_dir: Path) -> list[Path]:
    paths = sorted((data_dir / "raw" / "data").glob("train-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no raw shards under {data_dir / 'raw'} (run the download stage)")
    if len(paths) != N_SHARDS:
        log.warning("found %d raw shards, expected %d", len(paths), N_SHARDS)
    return paths


def filter_shards(
    data_dir: Path, shards: list[Path] | None = None, batch_size: int = 50_000
) -> dict:
    """Write `arxiv.parquet` from the raw shards and record the counts in the manifest."""
    out_path = data_dir / ARXIV_FILE
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    shards = shards if shards is not None else _shards(data_dir)
    # A rewritten corpus invalidates everything derived from the old one (row i must stay row i).
    _clear(data_dir, "filter")

    seen: set[bytes] = set()
    labels: Counter[str] = Counter()
    archives: set[str] = set()
    rows_in = dropped_withdrawn = dropped_short = dropped_duplicate = rows_out = 0

    with pq.ParquetWriter(tmp_path, OUT_SCHEMA) as writer:
        for shard in shards:
            log.info("filtering %s", shard.name)
            pf = pq.ParquetFile(shard)
            for batch in pf.iter_batches(batch_size=batch_size, columns=RAW_COLUMNS):
                cols = batch.to_pydict()
                dates = batch.column("update_date").cast(pa.date32()).cast(pa.string()).to_pylist()
                ids, texts, lbls, arcs, kept_dates = [], [], [], [], []
                for rid, cats, abstract, comments, date in zip(
                    cols["id"], cols["categories"], cols["abstract"], cols["comments"], dates
                ):
                    rows_in += 1
                    text = " ".join((abstract or "").split())
                    if is_withdrawn(text, " ".join((comments or "").split())):
                        dropped_withdrawn += 1
                        continue
                    if len(text.split()) < MIN_WORDS:
                        dropped_short += 1
                        continue
                    digest = abstract_digest(text)
                    if digest in seen:
                        dropped_duplicate += 1
                        continue
                    seen.add(digest)
                    label = cats.split()[0]
                    ids.append(rid)
                    texts.append(text)
                    lbls.append(label)
                    arcs.append(label.split(".")[0])
                    kept_dates.append(date)
                    labels[label] += 1
                    archives.add(arcs[-1])
                if ids:
                    writer.write_table(
                        pa.table([ids, texts, lbls, arcs, kept_dates], schema=OUT_SCHEMA),
                        row_group_size=ROW_GROUP_ROWS,
                    )
                    rows_out += len(ids)

    filter_info = {
        "rows_in": rows_in,
        "dropped_withdrawn": dropped_withdrawn,
        "dropped_short": dropped_short,
        "dropped_duplicate": dropped_duplicate,
        "rows_out": rows_out,
        "min_words": MIN_WORDS,
        "withdrawn_abstract_regex": WITHDRAWN_ABSTRACT_RE,
        "withdrawn_comments_regex": WITHDRAWN_COMMENTS_RE,
    }
    # Manifest first, then the rename: a crash in between leaves no arxiv.parquet, so
    # the stage reruns; the reverse order could leave a parquet with a stale manifest.
    update_manifest(
        data_dir,
        filter=filter_info,
        labels=dict(sorted(labels.items())),
        n_labels=len(labels),
        n_archives=len(archives),
    )
    tmp_path.replace(out_path)
    log.info("filter: %s", filter_info)
    return filter_info


def filter_done(data_dir: Path) -> bool:
    return (data_dir / ARXIV_FILE).exists() and "rows_out" in read_manifest(data_dir).get(
        "filter", {}
    )


# --- embed ------------------------------------------------------------------


def _iter_text(path: Path, chunk_rows: int) -> Iterator[list[str]]:
    for batch in pq.ParquetFile(path).iter_batches(batch_size=chunk_rows, columns=["text"]):
        yield batch.column("text").to_pylist()


def _load_progress(data_dir: Path, n_rows: int) -> dict:
    path = data_dir / PROGRESS_FILE
    emb = data_dir / EMBEDDINGS_FILE
    if path.exists() and emb.exists():
        progress = json.loads(path.read_text())
        if progress.get("n_rows") == n_rows:
            return progress
    return {"rows_done": 0, "n_rows": n_rows, "encode_seconds": 0.0}


def sentence_transformer_encoder(batch_size: int) -> tuple[Encoder, int]:
    """Return (encode, max_seq_length) for MiniLM: fp32, L2-normalized."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL, device=device)
    log.info("embedding on %s, max_seq_length=%d", device, model.max_seq_length)

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    return encode, int(model.max_seq_length)


def embed(
    data_dir: Path,
    encode: Encoder | None = None,
    batch_size: int = 384,
    chunk_rows: int = 16_384,
    max_rows: int | None = None,
    max_seq_length: int | None = None,
) -> dict:
    """Embed `arxiv.parquet` text into the fp16 memmap, resuming from `progress.json`.

    `encode(list[str]) -> float32 array (n, 384)` is injectable for tests. With
    `max_rows`, stop after about that many rows (a benchmark); the manifest's
    `complete` flag stays false until every row is done.
    """
    src = data_dir / ARXIV_FILE
    n_rows = pq.ParquetFile(src).metadata.num_rows
    emb_path = data_dir / EMBEDDINGS_FILE
    progress = _load_progress(data_dir, n_rows)
    done = progress["rows_done"]

    if done >= n_rows:  # every row is on disk but the manifest never got the flag
        info = {**read_manifest(data_dir).get("embedding", {}), "complete": True}
        update_manifest(data_dir, embedding=info)
        return info

    if encode is None:
        encode, max_seq_length = sentence_transformer_encoder(batch_size)

    mm = np.lib.format.open_memmap(
        emb_path,
        mode="r+" if done else "w+",
        dtype=np.float16,
        shape=(n_rows, EMBED_DIM),
    )
    target = n_rows if max_rows is None else min(n_rows, done + max_rows)
    start_done, run_start = done, time.perf_counter()
    logged_bench = False

    # Chunks are consumed from the start and the first `done` rows skipped: streaming
    # a parquet column is cheap next to encoding.
    offset = 0
    for texts in _iter_text(src, chunk_rows):
        end = offset + len(texts)
        if end <= done:
            offset = end
            continue
        if offset >= target:
            break
        lo = max(done - offset, 0)
        hi = min(len(texts), target - offset)
        t0 = time.perf_counter()
        vecs = np.asarray(encode(texts[lo:hi]), dtype=np.float32)
        progress["encode_seconds"] += time.perf_counter() - t0
        mm[offset + lo : offset + hi] = vecs.astype(np.float16)
        mm.flush()
        done = offset + hi
        offset = end
        progress["rows_done"] = done
        _write_json(data_dir / PROGRESS_FILE, progress)
        if not logged_bench and done - start_done >= BENCH_ROWS:
            logged_bench = True
            rate = (done - start_done) / (time.perf_counter() - run_start)
            log.info("embed throughput after %d rows: %.0f rows/s", done - start_done, rate)

    rows_per_s = done / progress["encode_seconds"] if progress["encode_seconds"] else 0.0
    log.info("embed: %d/%d rows, %.0f rows/s", done, n_rows, rows_per_s)
    info = {
        "model": EMBED_MODEL,
        "dim": EMBED_DIM,
        "dtype": "float16",
        "normalized": True,
        "text": "abstract",
        "max_seq_length": max_seq_length,
        "complete": done >= n_rows,
        "rows_per_s": round(rows_per_s, 1),
    }
    update_manifest(data_dir, embedding=info)
    return info


def embed_done(data_dir: Path) -> bool:
    emb = read_manifest(data_dir).get("embedding", {})
    return bool(emb.get("complete")) and (data_dir / EMBEDDINGS_FILE).exists()


# --- subsample --------------------------------------------------------------


def subsample(
    data_dir: Path,
    size: int = SUBSAMPLE_SIZE,
    seed: int = SUBSAMPLE_SEED,
    batch_size: int = 50_000,
) -> dict:
    """Write the seeded index subsample and the matching rows of `arxiv.parquet`."""
    src = data_dir / ARXIV_FILE
    n_rows = pq.ParquetFile(src).metadata.num_rows
    size = min(size, n_rows)
    idx = np.sort(
        np.random.default_rng(seed).choice(n_rows, size, replace=False).astype(np.int64)
    )

    _clear(data_dir, "subsample")  # a crash below then leaves the stage not done, never stale
    out_path = data_dir / SUBSAMPLE_PARQUET
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    schema = OUT_SCHEMA.append(pa.field("row", pa.int64()))
    offset = 0
    with pq.ParquetWriter(tmp_path, schema) as writer:
        for batch in pq.ParquetFile(src).iter_batches(batch_size=batch_size):
            lo, hi = np.searchsorted(idx, [offset, offset + batch.num_rows])
            if hi > lo:
                local = idx[lo:hi] - offset
                table = pa.Table.from_batches([batch]).take(pa.array(local))
                table = table.append_column("row", pa.array(idx[lo:hi], pa.int64()))
                writer.write_table(table.cast(schema), row_group_size=ROW_GROUP_ROWS)
            offset += batch.num_rows

    info = {"size": size, "seed": seed, "file": SUBSAMPLE_FILE}
    npy_tmp = data_dir / (SUBSAMPLE_FILE + ".tmp")
    with open(npy_tmp, "wb") as f:
        np.save(f, idx)
    npy_tmp.replace(data_dir / SUBSAMPLE_FILE)
    tmp_path.replace(out_path)
    update_manifest(data_dir, subsample=info)
    log.info("subsample: %s", info)
    return info


def subsample_done(data_dir: Path, size: int, seed: int = SUBSAMPLE_SEED) -> bool:
    info = read_manifest(data_dir).get("subsample", {})
    n_rows = pq.ParquetFile(data_dir / ARXIV_FILE).metadata.num_rows
    return (
        info.get("size") == min(size, n_rows)
        and info.get("seed") == seed
        and info.get("file") == SUBSAMPLE_FILE
        and (data_dir / SUBSAMPLE_FILE).exists()
        and (data_dir / SUBSAMPLE_PARQUET).exists()
    )


# --- driver -----------------------------------------------------------------


def _clear(data_dir: Path, stage: str) -> None:
    """Delete a stage's outputs (and those of stages that depend on it)."""
    files = {
        "filter": [ARXIV_FILE, EMBEDDINGS_FILE, PROGRESS_FILE, SUBSAMPLE_FILE, SUBSAMPLE_PARQUET],
        "embed": [EMBEDDINGS_FILE, PROGRESS_FILE],
        "subsample": [SUBSAMPLE_FILE, SUBSAMPLE_PARQUET],
    }[stage]
    for name in files:
        (data_dir / name).unlink(missing_ok=True)
    manifest = read_manifest(data_dir)
    drop = {
        "filter": ["filter", "labels", "n_labels", "n_archives", "embedding", "subsample"],
        "embed": ["embedding"],
        "subsample": ["subsample"],
    }.get(stage, [])
    if drop:
        for key in drop:
            manifest.pop(key, None)
        _write_json(data_dir / MANIFEST_FILE, manifest)


def run(
    data_dir: Path,
    stages: tuple[str, ...] = STAGES,
    force: str | None = None,
    batch_size: int = 384,
    chunk_rows: int = 16_384,
    max_rows: int | None = None,
    subsample_size: int = SUBSAMPLE_SIZE,
    encode: Encoder | None = None,
) -> None:
    if force and force not in FORCEABLE:
        raise ValueError(f"--force must be one of {list(FORCEABLE)}, got {force!r}")
    if force and force not in stages:
        raise ValueError(f"--force {force} requires {force} in --stages {list(stages)}")
    data_dir.mkdir(parents=True, exist_ok=True)
    if force:
        _clear(data_dir, force)
    for stage in STAGES:
        if stage not in stages:
            continue
        if stage == "download":
            download(data_dir)
        elif stage == "filter":
            if filter_done(data_dir):
                log.info("filter: %s already complete, skipping", ARXIV_FILE)
            else:
                filter_shards(data_dir)
        elif stage == "embed":
            if embed_done(data_dir):
                log.info("embed: already complete, skipping")
            else:
                embed(data_dir, encode, batch_size, chunk_rows, max_rows)
        elif stage == "subsample":
            if subsample_done(data_dir, subsample_size):
                log.info("subsample: already complete, skipping")
            else:
                subsample(data_dir, subsample_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument(
        "--force", choices=FORCEABLE, help="redo this stage (forcing filter also redoes the later ones)"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR / "arxiv")
    parser.add_argument("--batch-size", type=int, default=384, help="encoder batch size")
    parser.add_argument("--chunk-rows", type=int, default=16_384, help="rows per checkpoint")
    parser.add_argument(
        "--max-rows", type=int, help="embed at most this many more rows (throughput benchmark)"
    )
    parser.add_argument("--subsample-size", type=int, default=SUBSAMPLE_SIZE)
    args = parser.parse_args()
    if args.force and args.force not in args.stages:
        parser.error(f"--force {args.force} requires {args.force} in --stages")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(
        args.data_dir,
        tuple(args.stages),
        args.force,
        args.batch_size,
        args.chunk_rows,
        args.max_rows,
        args.subsample_size,
    )


if __name__ == "__main__":
    main()
