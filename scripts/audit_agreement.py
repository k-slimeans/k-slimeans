"""Agreement between the hand ratings of the summary audit and the automatic flags (issue #34).

Reads a filled `summary_audit_sheet_<dataset>.csv` and its `summary_audit_key_<dataset>.csv` (which stores the
automatic flags of the same summaries) and prints, per property, the 2 x 2 table, the raw agreement and Cohen's
kappa. One rater. Empty rating cells are dropped. The hand labels and their rules are in
`docs/summary_audit_rubric.md`.

    python scripts/audit_agreement.py docs/results/summary_audit_sheet_clinc150_intent.csv \\
        docs/results/summary_audit_key_clinc150_intent.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from kslimeans.summary_audit import HAND_LABELS, agreement, hand_label

HEADER = ["property", "n", "hand yes", "auto yes", "both", "hand only", "auto only", "neither", "raw agreement", "kappa"]


def compare(sheet: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """One row per property: the agreement between the hand label and the automatic flag."""
    merged = sheet.merge(key, on="row_id", how="inner", validate="one_to_one", suffixes=("", "_key"))
    rows = []
    for name, flag in HAND_LABELS.items():
        pairs = [
            (h, bool(r[flag]))
            for r in merged.to_dict("records")
            if (h := hand_label(r, name)) is not None and pd.notna(r[flag])
        ]
        stats = agreement([h for h, _ in pairs], [a for _, a in pairs])
        rows.append({"property": name, **stats})
    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(HEADER) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(HEADER) - 1)) + " |"]
    for r in table.itertuples():
        lines.append(
            f"| {r.property} | {r.n} | {r.hand_pos} | {r.auto_pos} | {r.both} | {r.hand_only} | {r.auto_only} "
            f"| {r.neither} | {r.raw_agreement:.2f} | {r.kappa:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sheet", help="the filled sheet CSV")
    parser.add_argument("key", help="the matching key CSV")
    args = parser.parse_args()
    print(to_markdown(compare(pd.read_csv(args.sheet), pd.read_csv(args.key))))


if __name__ == "__main__":
    main()
