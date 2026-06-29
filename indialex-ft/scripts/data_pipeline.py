"""data_pipeline.py — Load, clean, and split Indian legal data for SFT.

Input:  data/raw/  — JSONL, JSON, or CSV files with fields: instruction, input, output
Output: data/splits/ — HuggingFace Dataset saved to disk (train / val / test)

Usage:
    # Normal pipeline (existing raw files only)
    python scripts/data_pipeline.py

    # Generate 500 synthetic pairs first, then run pipeline
    python scripts/data_pipeline.py --synthetic

    # Custom paths
    python scripts/data_pipeline.py --raw_dir data/raw --out_dir data/splits
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import csv
import json
import logging
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Iterator

from datasets import Dataset, DatasetDict

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Synthetic data config
# ---------------------------------------------------------------------------

_SYNTHETIC_TOPICS = [
    "Section 80C deductions under Indian Income Tax Act",
    "Section 80D health insurance deductions",
    "GST (Goods and Services Tax) basics in India",
    "ITR filing types — ITR-1, ITR-2, ITR-3, ITR-4, ITR-5, ITR-6, ITR-7",
    "HRA (House Rent Allowance) exemption calculation",
    "Capital gains tax — short-term and long-term in India",
    "TDS (Tax Deducted at Source) rules and rates",
    "Advance tax computation and due dates",
    "Professional tax across Indian states",
    "Income tax slabs and rates for FY 2024-25 (AY 2025-26)",
]

_BATCH_SIZE  = 10   # pairs per API call
_ROUNDS      = 5    # calls per topic  → 10 topics × 5 rounds × 10 pairs = 500 total
_TOTAL_PAIRS = 500


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
    inp         = _clean(str(rec.get("input",       "") or ""))
    output      = _clean(str(rec.get("output",      "") or ""))

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
    val_test  = train_val["test"].train_test_split(test_size=0.5, seed=42)
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
    parser.add_argument("--raw_dir",  default=str(ROOT / "data" / "raw"))
    parser.add_argument("--out_dir",  default=str(ROOT / "data" / "splits"))
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
