"""Scoring the human ratings of the summary hand check (issue #34): rater agreement, consensus, paraphrase check.

Everything here works on the filled sheets (one `DataFrame` per rater, rows matched by `row_id`) and the key made by
`scripts/make_audit_sheet.py --write-key`. `scripts/human_eval_report.py` is the command line around it. The
agreement code itself (2 x 2 tables, kappa, hand labels) is `kslimeans.summary_audit`.

The paraphrase check answers whether the unsupported-content flag of `summary_quality.py` wrongly flags reasonable
paraphrases. For each audited summary it recomputes the flag from the cluster's FULL member set with
`summary_quality.unsupported_content`, exactly as the scores do, so the words, numbers and entities that set it off
are known, and it must reproduce the key's flag for every row (`check_flags`). The raters see at most
`summary_audit.N_MEMBERS` members, so for every content lemma of the summary it also records whether it occurs in
the shown members, only in members that were not shown, or in no member.

The category `unshown_members` can only apply to the reverse table (a rater says unsupported, the flag is silent):
a flagged summary's triggers are absent from EVERY member by construction, shown or not. The flagged table's
`unsupported_lemmas` lists all the summary's lemmas absent from the members; the flag fired on them only if
`trigger_path` contains `lemma_share`, otherwise it fired on a number or an entity.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .summary_audit import HAND_LABELS, compare_raters, compare_with_key, hand_label
from .summary_quality import UNSUPPORTED_LEMMA_SHARE, _blob, analyze_corpus, content_lemmas, summary_lines, unsupported_content

CATEGORIES = ("paraphrase", "generalisation", "inflection_tokenisation", "unshown_members", "cluster_talk", "unsupported")
"""Hand categories for the paraphrase check. `cluster_talk`: the absent words describe the cluster or the task
("What is the main topic of the following cluster of questions?", an echoed instruction) rather than its members,
so they are neither a paraphrase nor invented content."""
RATING_COLUMN = {"low_coverage": "members_fit", "generic": "generic", "unsupported": "unsupported", "not_fluent": "fluent"}
"""The sheet column each hand label is read from."""

_MEMBER_PREFIX = re.compile(r"^\d+\. ")
FIXED_COLUMNS = ("summary", "cluster_size", "members_shown", "members")
"""The sheet columns a rater must not change."""
BINARY_COLUMNS = ("generic", "unsupported", "fluent")


def check_sheets(
    sheets: Mapping[str, pd.DataFrame], key: pd.DataFrame, blank: pd.DataFrame | None = None
) -> None:
    """Raise `ValueError` unless every rater's sheet has exactly the key's rows and the same fixed columns.

    The fixed columns (`FIXED_COLUMNS`) must equal those of `blank` (the committed blank sheet) when it is given,
    otherwise those of the first rater's sheet. The error names the rater, the column and the first row_ids.
    """
    expected = set(key["row_id"])
    for rater, sheet in sheets.items():
        if sheet["row_id"].duplicated().any() or set(sheet["row_id"]) != expected:
            raise ValueError(f"the sheet of rater {rater!r} does not have exactly the rows of the key")
    reference = (blank if blank is not None else next(iter(sheets.values()))).set_index("row_id")
    if blank is not None and set(reference.index) != expected:
        raise ValueError("the blank sheet does not have exactly the rows of the key")
    for rater, sheet in sheets.items():
        indexed = sheet.set_index("row_id").loc[reference.index]
        for column in FIXED_COLUMNS:
            a, b = indexed[column], reference[column]
            if column == "summary":
                a, b = a.fillna(""), b.fillna("")
            differs = ~((a == b) | (a.isna() & b.isna()))
            if differs.any():
                raise ValueError(
                    f"rater {rater!r} changed column {column!r} (rows {list(indexed.index[differs])[:5]}); "
                    "it must equal the blank sheet" if blank is not None else
                    f"column {column!r} of rater {rater!r} differs from the other raters' sheets "
                    f"(rows {list(indexed.index[differs])[:5]})"
                )


def _cell(value) -> float | None:
    """A rating cell as a number, `None` when empty; `ValueError` for text."""
    if value is None or pd.isna(value) or str(value).strip().lower() in ("", "nan"):
        return None
    return float(value)


def validate_ratings(sheets: Mapping[str, pd.DataFrame]) -> None:
    """Raise `ValueError` naming rater, row_id and column for a rating outside the rubric's values.

    `generic`, `unsupported` and `fluent` may hold 0, 1 or nothing; `members_fit` a whole number from 0 to the row's
    `members_shown` or nothing. `"1.0"` and surrounding spaces are accepted.
    """
    for rater, sheet in sheets.items():
        for r in sheet.to_dict("records"):
            for column in ("members_fit", *BINARY_COLUMNS):
                try:
                    v = _cell(r[column])
                except ValueError:
                    v = "text"
                if v is None:
                    continue
                if column == "members_fit":
                    shown = _cell(r["members_shown"])
                    ok = v != "text" and v == int(v) and 0 <= v <= (shown if shown is not None else v)
                    rule = f"a whole number from 0 to members_shown ({r['members_shown']})"
                else:
                    ok = v in (0, 1)
                    rule = "0 or 1"
                if not ok:
                    raise ValueError(
                        f"rater {rater!r}, row {r['row_id']}, column {column!r}: {r[column]!r} is not {rule} or empty"
                    )


# ------------------------------------------------------------------ agreement between raters


def pairwise_agreement(sheets: Mapping[str, pd.DataFrame]) -> dict[tuple[str, str], pd.DataFrame]:
    """`compare_raters` for every pair of raters, in the order the raters were given."""
    names = list(sheets)
    return {
        (a, b): compare_raters(sheets[a], sheets[b]) for i, a in enumerate(names) for b in names[i + 1 :]
    }


def members_fit_agreement(sheet_a: pd.DataFrame, sheet_b: pd.DataFrame) -> dict:
    """Agreement of the two raters' `members_fit` counts: n, Spearman correlation and mean absolute difference.

    Rows either rater left empty are dropped. The correlation is `NaN` when fewer than two rows are left or one
    rater gave the same count throughout.
    """
    a = sheet_a.set_index("row_id")["members_fit"]
    b = sheet_b.set_index("row_id")["members_fit"]
    both = pd.concat([pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")], axis=1).dropna()
    x, y = both.iloc[:, 0], both.iloc[:, 1]
    rho = float("nan")
    if len(both) >= 2 and x.nunique() > 1 and y.nunique() > 1:
        rho = float(np.corrcoef(x.rank(), y.rank())[0, 1])  # Spearman: Pearson on the average ranks
    return {
        "n": len(both),
        "spearman": rho,
        "mean_abs_diff": float((x - y).abs().mean()) if len(both) else float("nan"),
        "identical": int((x == y).sum()),
    }


def rater_disagreements(sheets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The rows on which the raters do not all give the same value, one line per row and property.

    For the four hand labels the value is the label as 0/1; `members_fit` is listed on its own with the raw counts.
    A row a rater left empty is not a disagreement. Columns: `row_id`, `property`, then one column per rater.
    """
    raters = list(sheets)
    records = {r: {row["row_id"]: row for row in sheets[r].to_dict("records")} for r in raters}
    row_ids = sorted(set.intersection(*(set(rec) for rec in records.values())))
    out = []
    for name in (*HAND_LABELS, "members_fit"):
        for row_id in row_ids:
            if name == "members_fit":
                values = [records[r][row_id].get("members_fit") for r in raters]
                values = [None if pd.isna(v) or str(v).strip() == "" else float(v) for v in values]
            else:
                labels = [hand_label(records[r][row_id], name) for r in raters]
                values = [None if v is None else int(v) for v in labels]
            present = [v for v in values if v is not None]
            if len(set(present)) > 1:
                out.append({"row_id": row_id, "property": name, **dict(zip(raters, values))})
    table = pd.DataFrame(out, columns=["row_id", "property", *raters])
    try:
        table[raters] = table[raters].astype("Int64")  # whole numbers, <NA> for an empty cell
    except TypeError:  # a fractional count typed into `members_fit`
        table[raters] = table[raters].astype("Float64")
    return table


def consensus_sheet(sheets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """One sheet holding, per property, the rating on which every rater agrees (empty where they differ or one is empty).

    Agreement is on the hand label, not on the raw cell: two `members_fit` counts that both mean "low coverage" are
    consensus and the first rater's count is kept. The result can go through `compare_with_key` like a rater's sheet.
    """
    raters = list(sheets)
    indexed = {r: sheets[r].set_index("row_id") for r in raters}
    base = indexed[raters[0]].copy()
    for name, column in RATING_COLUMN.items():
        keep = []
        for row_id in base.index:
            labels = [hand_label(indexed[r].loc[row_id].to_dict(), name) for r in raters]
            keep.append(None not in labels and len(set(labels)) == 1)
        base[column] = base[column].where(pd.Series(keep, index=base.index), np.nan)
    return base.reset_index()


# ------------------------------------------------------------------ unsupported content: the paraphrase check


def shown_members(cell: str) -> list[str]:
    """The member texts of a sheet's `members` cell (numbered lines)."""
    return [_MEMBER_PREFIX.sub("", line, count=1) for line in str(cell).split("\n") if line.strip()]


def analyse_unsupported(summary: str | None, shown: Sequence[str], members: Sequence[str], nlp) -> dict:
    """Recompute `unsupported_content` of `summary` against all `members` and place each content lemma.

    `members` is the cluster's full member set (what the flag reads), `shown` the subset the raters saw.
    Returns the fields of `unsupported_content` plus `lemmas_in_shown`, `lemmas_unshown_only` and
    `lemmas_in_no_member` (the summary's content lemmas, space-separated and sorted) and `trigger_path`: which
    rules fired, any of `lemma_share` (unsupported share at least `UNSUPPORTED_LEMMA_SHARE`), `number`, `entity`,
    joined by `+` (empty when the flag is 0).
    """
    docs = list(nlp.pipe(summary_lines(summary), disable=["parser"]))
    corpus = analyze_corpus(list(members), nlp)
    all_lemmas = set().union(*corpus.lemmas)
    result = unsupported_content(docs, all_lemmas, _blob(corpus.norm))
    shown_lemmas = set().union(*analyze_corpus(list(shown), nlp).lemmas)
    lemmas = sorted(frozenset().union(*(content_lemmas(d) for d in docs)))
    result["lemmas_in_shown"] = " ".join(w for w in lemmas if w in shown_lemmas)
    result["lemmas_unshown_only"] = " ".join(w for w in lemmas if w in all_lemmas and w not in shown_lemmas)
    result["lemmas_in_no_member"] = " ".join(w for w in lemmas if w not in all_lemmas)
    share = result["unsupported_lemma_share"]
    rules = [
        ("lemma_share", share >= UNSUPPORTED_LEMMA_SHARE),
        ("number", result["n_unsupported_numbers"] > 0),
        ("entity", result["n_unsupported_entities"] > 0),
    ]
    result["trigger_path"] = "+".join(name for name, fired in rules if fired)
    return result


def analyse_rows(
    sheet: pd.DataFrame, key: pd.DataFrame, members: Mapping[str, Sequence[str]], nlp
) -> pd.DataFrame:
    """`analyse_unsupported` for every row of one dataset; `members[row_id]` is the cluster's full member list.

    One row per `row_id` of the key with the key's `method` and the recomputed fields, plus the key's own
    `flag_unsupported` as `key_flag_unsupported` and `key_unsupported_lemma_share` for `check_flags`. Raises
    `ValueError` when a row's `cluster_size` (from the sheet) is not the number of members loaded for it, which
    means the wrong run, step or cluster was read.
    """
    merged = sheet.merge(key, on="row_id", how="inner", validate="one_to_one", suffixes=("", "_key"))
    rows = []
    for r in merged.to_dict("records"):
        summary = r["summary"] if isinstance(r["summary"], str) else ""
        if int(r["cluster_size"]) != len(members[r["row_id"]]):
            raise ValueError(
                f"row {r['row_id']}: the sheet says a cluster of {r['cluster_size']} but {len(members[r['row_id']])} "
                "members were loaded (wrong run, step or cluster?)"
            )
        res = analyse_unsupported(summary, shown_members(r["members"]), members[r["row_id"]], nlp)
        rows.append(
            {
                "dataset": r["dataset"],
                "row_id": r["row_id"],
                "method": r["method"],
                "summary": summary,
                "cluster_size": len(members[r["row_id"]]),
                "members_shown": r["members_shown"],
                "flag_unsupported": int(res["flag_unsupported"]),
                "unsupported_lemma_share": res["unsupported_lemma_share"],
                "unsupported_lemmas": res["unsupported_lemmas"],
                "trigger_path": res["trigger_path"],
                "trigger_numbers_entities": res["unsupported_items"],
                "n_trigger_numbers": res["n_unsupported_numbers"],
                "n_trigger_entities": res["n_unsupported_entities"],
                "lemmas_in_shown": res["lemmas_in_shown"],
                "lemmas_unshown_only": res["lemmas_unshown_only"],
                "lemmas_in_no_member": res["lemmas_in_no_member"],
                "key_flag_unsupported": int(bool(r["flag_unsupported"])),
                "key_unsupported_lemma_share": r["unsupported_lemma_share"],
            }
        )
    return pd.DataFrame(rows)


def check_flags(analysis: pd.DataFrame, tol: float = 1e-4) -> None:
    """Raise `ValueError` naming the rows where the recomputation disagrees with the key's flag or lemma share."""
    flag_bad = analysis["flag_unsupported"] != analysis["key_flag_unsupported"]
    a, b = analysis["unsupported_lemma_share"], analysis["key_unsupported_lemma_share"]
    share_bad = ~((a - b).abs() <= tol) & ~(a.isna() & b.isna())
    bad = analysis.loc[flag_bad | share_bad, "row_id"].tolist()
    if bad:
        raise ValueError(f"the recomputed unsupported flag or lemma share differs from the key on rows {bad}")


def _rater_columns(table: pd.DataFrame, sheets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    for rater, sheet in sheets.items():
        ratings = sheet.set_index("row_id")["unsupported"]
        table[f"unsupported_{rater}"] = table["row_id"].map(ratings)
    return table


def _with_category(table: pd.DataFrame) -> pd.DataFrame:
    table["category"] = ""
    return table


def flagged_table(analysis: pd.DataFrame, sheets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Every summary the automatic flag marks unsupported, with each rater's `unsupported` and an empty `category`."""
    table = analysis[analysis["flag_unsupported"] == 1].drop(columns=["key_flag_unsupported", "key_unsupported_lemma_share"])
    return _with_category(_rater_columns(table.reset_index(drop=True), sheets))


def reverse_table(analysis: pd.DataFrame, sheets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summaries at least one rater marks unsupported (1) although the flag is 0, same columns as `flagged_table`."""
    marked = pd.Series(False, index=analysis["row_id"])
    for sheet in sheets.values():
        marked |= sheet.set_index("row_id")["unsupported"].eq(1).reindex(marked.index, fill_value=False)
    table = analysis[(analysis["flag_unsupported"] == 0) & analysis["row_id"].map(marked)]
    table = table.drop(columns=["key_flag_unsupported", "key_unsupported_lemma_share"])
    return _with_category(_rater_columns(table.reset_index(drop=True), sheets))


def paraphrase_summary(flagged: pd.DataFrame, raters: Sequence[str]) -> dict:
    """The flagged summaries, how many every rater rated, and how many of those every rater judged supported (0)."""
    cols = [f"unsupported_{r}" for r in raters]
    rated = flagged[flagged[cols].notna().all(axis=1)]
    supported = int((rated[cols] == 0).all(axis=1).sum())
    return {
        "flagged": len(flagged),
        "rated_by_all": len(rated),
        "all_supported": supported,
        "share_all_supported": supported / len(rated) if len(rated) else float("nan"),
    }


def category_counts(table: pd.DataFrame) -> dict[str, int]:
    """Counts per `category` (hand-filled) of a paraphrase CSV, plus `unlabelled`; unknown values raise `ValueError`."""
    values = table["category"].fillna("").astype(str).str.strip()
    unknown = sorted(set(values) - set(CATEGORIES) - {""})
    if unknown:
        raise ValueError(f"unknown categories {unknown}; use one of {list(CATEGORIES)}")
    counts = {c: int((values == c).sum()) for c in CATEGORIES}
    counts["unlabelled"] = int((values == "").sum())
    return counts


# ------------------------------------------------------------------ report pieces


def vs_flags(sheets: Mapping[str, pd.DataFrame], key: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """`compare_with_key` per rater, plus the consensus sheet under the name `consensus`."""
    out = {r: compare_with_key(s, key) for r, s in sheets.items()}
    out["consensus"] = compare_with_key(consensus_sheet(sheets), key)
    return out
