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
# Synthetic data generation (Groq) — basic
# ---------------------------------------------------------------------------

def generate_synthetic_data(out_path: Path, total: int = _TOTAL_PAIRS) -> int:
    """Generate synthetic Indian tax/legal Q&A pairs via the Groq API.

    Makes _ROUNDS calls per topic (10 topics × 5 rounds × 10 pairs = 500).
    Uses response_format=json_object to force valid JSON output.

    Environment variable required:
        GROQ_API_KEY  — your Groq API key
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        log.error("GROQ_API_KEY is not set — set it in your .env file.")
        return 0

    try:
        from groq import Groq
    except ImportError:
        log.error("groq package not installed. Run: pip install groq")
        return 0

    client      = Groq(api_key=api_key)
    all_pairs: list[dict] = []
    call_num    = 0
    total_calls = len(_SYNTHETIC_TOPICS) * _ROUNDS

    for topic in _SYNTHETIC_TOPICS:
        for round_idx in range(1, _ROUNDS + 1):
            call_num += 1
            log.info(
                "Call %d/%d — %s (round %d/%d)",
                call_num, total_calls, topic, round_idx, _ROUNDS,
            )

            prompt = (
                f'Generate exactly {_BATCH_SIZE} unique Indian income tax / legal Q&A pairs '
                f'about "{topic}" (round {round_idx} of {_ROUNDS} — use different questions '
                f'each round).\n\n'
                "Rules:\n"
                "- Questions must be realistic — things an Indian taxpayer, salaried employee, "
                "or CA would actually ask.\n"
                "- Answers must be accurate, India-specific, and at least 3 sentences long.\n"
                "- Cover diverse sub-aspects: eligibility, limits, calculation, examples, "
                "common mistakes, recent changes.\n\n"
                "Return a JSON object with a single key 'pairs' whose value is an array:\n"
                '{"pairs": [\n'
                '  {"instruction": "<question>", "input": "", "output": "<detailed answer>"},\n'
                "  ...\n"
                "]}"
            )

            try:
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a JSON generator. Output ONLY valid JSON — "
                                "no markdown, no explanation, no text outside the JSON object."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                raw   = resp.choices[0].message.content.strip()
                data  = json.loads(raw)
                pairs = data.get("pairs", [])

                valid = []
                for p in pairs:
                    if not isinstance(p, dict):
                        continue
                    instruction = str(p.get("instruction", "")).strip()
                    output      = str(p.get("output", "")).strip()
                    if instruction and output:
                        valid.append({
                            "instruction": instruction,
                            "input":       str(p.get("input", "")),
                            "output":      output,
                        })

                all_pairs.extend(valid)
                log.info("  +%d pairs (total so far: %d)", len(valid), len(all_pairs))

            except json.JSONDecodeError as exc:
                log.warning("Call %d — JSON parse error: %s", call_num, exc)
            except Exception as exc:
                log.warning("Call %d — failed: %s", call_num, exc)

            if call_num < total_calls:
                time.sleep(3)

    if not all_pairs:
        log.error("No synthetic pairs were generated — check GROQ_API_KEY and network.")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(all_pairs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Saved %d synthetic pairs → %s", len(all_pairs), out_path)
    return len(all_pairs)


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


def _load_json(path: Path) -> Iterator[dict]:
    """Load a JSON file that contains either an array of records or a single record."""
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        log.warning("Skipping malformed JSON file: %s", path.name)
        return
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        yield data
    else:
        log.warning("Unexpected top-level type in %s — skipping", path.name)


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
        elif path.suffix == ".json":
            loader = _load_json(path)
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
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate 500 synthetic Q&A pairs via Groq before running the pipeline.",
    )
    parser.add_argument(
        "--synthetic_out",
        default=str(ROOT / "data" / "raw" / "synthetic.json"),
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        synthetic_path = Path(args.synthetic_out)
        if synthetic_path.exists():
            log.info("synthetic.json already exists — skipping generation.")
        else:
            n = generate_synthetic_data(synthetic_path)
            if n == 0:
                log.warning("Synthetic generation produced no pairs — continuing with existing data.")

    records = load_raw(raw_dir)
    if not records:
        log.error("No valid records found in %s — add .jsonl / .json / .csv files first.", raw_dir)
        return

    splits = split_dataset(records)
    for split, ds in splits.items():
        log.info("  %s: %d samples", split, len(ds))

    splits.save_to_disk(str(out_dir))
    log.info("Dataset saved to %s", out_dir)


if __name__ == "__main__":
    main()
