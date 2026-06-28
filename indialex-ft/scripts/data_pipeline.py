"""data_pipeline.py — Load, clean, and split Indian legal data for SFT.

Input:  data/raw/  — JSONL or CSV files with fields: instruction, input, output
Output: data/splits/ — HuggingFace Dataset saved to disk (train / val / test)

Usage:
    python scripts/data_pipeline.py [--raw_dir data/raw] [--out_dir data/splits]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterator

from datasets import Dataset, DatasetDict

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Normalise unicode, collapse whitespace, strip control chars."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)  # control chars
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _format_record(rec: dict) -> dict | None:
    """Return a cleaned instruction-tuning record or None if invalid."""
    instruction = _clean(str(rec.get("instruction", "") or ""))
    inp = _clean(str(rec.get("input", "") or ""))
    output = _clean(str(rec.get("output", "") or ""))

    if not instruction or not output:
        return None

    return {"instruction": instruction, "input": inp, "output": output}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skipping malformed JSON line in %s", path.name)


def _load_csv(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


def load_raw(raw_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(raw_dir.iterdir()):
        if path.suffix == ".jsonl":
            loader = _load_jsonl(path)
        elif path.suffix == ".csv":
            loader = _load_csv(path)
        else:
            continue
        log.info("Loading %s …", path.name)
        for raw in loader:
            rec = _format_record(raw)
            if rec:
                records.append(rec)
    log.info("Total valid records: %d", len(records))
    return records


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_dataset(records: list[dict]) -> DatasetDict:
    ds = Dataset.from_list(records)
    # 80 / 10 / 10
    train_val = ds.train_test_split(test_size=0.2, seed=42)
    val_test = train_val["test"].train_test_split(test_size=0.5, seed=42)
    return DatasetDict({
        "train": train_val["train"],
        "val":   val_test["train"],
        "test":  val_test["test"],
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--out_dir", default=str(ROOT / "data" / "splits"))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_raw(raw_dir)
    if not records:
        log.error("No valid records found in %s — add .jsonl or .csv files first.", raw_dir)
        return

    splits = split_dataset(records)
    for split, ds in splits.items():
        log.info("  %s: %d samples", split, len(ds))

    splits.save_to_disk(str(out_dir))
    log.info("Dataset saved to %s", out_dir)


if __name__ == "__main__":
    main()
