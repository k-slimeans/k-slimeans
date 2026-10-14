"""Export full CLINC150 (in-scope only) and its intent -> domain mapping into data/.

Writes two files:

* data/clinc150_full.csv     -- 22,500 rows: train + validation + test, `oos` dropped.
                                Same columns as clinc_small_test.csv: text, intent, intent_name.
* data/clinc150_domains.csv  -- 150 rows: intent_name, domain (10 domains).

Source is the original authors' repository, clinc/oos-eval (`data/data_full.json`
and `data/domains.json`, CC BY 3.0). The Hugging Face `clinc/clinc_oos` "plus"/"full"
configs repackage the same in-scope queries; to keep integer `intent` ids identical
to the Hugging Face ones (and to clinc_small_test.csv), the name -> id table is read
from clinc_small_test.csv, which was exported from Hugging Face.

Usage (from the repo root):  python scripts/export_clinc150.py
Only needs to be run once; the outputs are checked in.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _fetch_json(name: str) -> dict:
    with urllib.request.urlopen(BASE_URL + name, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    full = _fetch_json("data_full.json")
    domains = _fetch_json("domains.json")

    # Intent name -> Hugging Face integer id, taken from the existing small export.
    small = pd.read_csv(DATA_DIR / "clinc_small_test.csv")
    name_to_id = dict(zip(small["intent_name"], small["intent"]))

    rows = []
    for split in ("train", "val", "test"):  # in-scope splits only; oos_* are skipped
        for text, intent_name in full[split]:
            rows.append({"text": text, "intent": name_to_id[intent_name], "intent_name": intent_name})
    df = pd.DataFrame(rows, columns=["text", "intent", "intent_name"])

    assert len(df) == 22_500, len(df)
    assert df["intent_name"].nunique() == 150
    assert "oos" not in set(df["intent_name"])

    mapping = pd.DataFrame(
        [{"intent_name": intent, "domain": domain} for domain, intents in domains.items() for intent in intents]
    ).sort_values(["domain", "intent_name"])
    assert len(mapping) == 150 and mapping["domain"].nunique() == 10
    assert set(mapping["intent_name"]) == set(df["intent_name"])

    df.to_csv(DATA_DIR / "clinc150_full.csv", index=False)
    mapping.to_csv(DATA_DIR / "clinc150_domains.csv", index=False)
    print(f"wrote {len(df)} rows to clinc150_full.csv and {len(mapping)} rows to clinc150_domains.csv")


if __name__ == "__main__":
    main()
