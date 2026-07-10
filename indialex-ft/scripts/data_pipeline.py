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
import sys
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

_BATCH_SIZE  = 10   # pairs per API call (small = reliable JSON + avoids token limits)
_ROUNDS      = 5    # calls per topic  → 10 topics × 5 rounds × 10 pairs = 500 total
_TOTAL_PAIRS = 500


# ---------------------------------------------------------------------------
# Synthetic data generation (Groq)
# ---------------------------------------------------------------------------

def _parse_retry_after(error_msg: str) -> float:
    """Extract seconds to wait from a Groq 429 error message.

    Groq embeds e.g. 'Please try again in 22m8.832s' in the error body.
    Returns the parsed seconds, or 65.0 as a safe default.
    """
    m = re.search(r"try again in\s+(?:(\d+)m)?([\d.]+)s", str(error_msg))
    if m:
        minutes = int(m.group(1) or 0)
        seconds = float(m.group(2) or 0)
        return minutes * 60 + seconds
    return 65.0


def _save(pairs: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(pairs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def generate_synthetic_data(out_path: Path, total: int = _TOTAL_PAIRS) -> int:
    """Generate synthetic Indian tax/legal Q&A pairs via the Groq API.

    Resume-aware: loads any pairs already saved in *out_path* and only
    generates what is still missing.  Writes the file after every successful
    API call so progress is never lost.

    On a 429 (token-per-day limit) the error message contains the exact
    retry delay; the function sleeps for that duration and retries the same
    call once before moving on.

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

    # ── Resume: load existing pairs ──────────────────────────────────────────
    all_pairs: list[dict] = []
    if out_path.exists():
        try:
            all_pairs = json.loads(out_path.read_text(encoding="utf-8"))
            log.info("Resuming — loaded %d existing pairs from %s", len(all_pairs), out_path)
        except Exception:
            log.warning("Could not read existing %s — starting fresh.", out_path)

    if len(all_pairs) >= total:
        log.info("Already have %d/%d pairs — nothing to generate.", len(all_pairs), total)
        return len(all_pairs)

    client      = Groq(api_key=api_key)
    total_calls = len(_SYNTHETIC_TOPICS) * _ROUNDS
    call_num    = 0

    for topic in _SYNTHETIC_TOPICS:
        for round_idx in range(1, _ROUNDS + 1):
            call_num += 1

            if len(all_pairs) >= total:
                log.info("Target of %d pairs reached — stopping early.", total)
                break

            needed = total - len(all_pairs)
            batch  = min(_BATCH_SIZE, needed)

            log.info(
                "Call %d/%d — %s (round %d/%d) | have %d/%d pairs",
                call_num, total_calls, topic, round_idx, _ROUNDS,
                len(all_pairs), total,
            )

            prompt = (
                f'Generate exactly {batch} unique Indian income tax / legal Q&A pairs '
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

            api_kwargs = dict(
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

            # One automatic retry on 429 after sleeping the advertised delay
            for attempt in (1, 2):
                try:
                    resp  = client.chat.completions.create(**api_kwargs)
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
                    _save(all_pairs, out_path)          # incremental save
                    log.info(
                        "  +%d pairs saved (total: %d/%d)",
                        len(valid), len(all_pairs), total,
                    )
                    break  # success — don't retry

                except json.JSONDecodeError as exc:
                    log.warning("Call %d attempt %d — JSON parse error: %s", call_num, attempt, exc)
                    break  # bad output, skip this call

                except Exception as exc:
                    err = str(exc)
                    if "429" in err and attempt == 1:
                        wait = _parse_retry_after(err)
                        log.warning(
                            "Call %d — rate limit hit. Sleeping %.0fs then retrying…",
                            call_num, wait,
                        )
                        time.sleep(wait + 5)   # +5 s buffer
                    else:
                        log.warning("Call %d attempt %d — failed: %s", call_num, attempt, exc)
                        break

            # Brief pause between successful calls to stay under per-minute limits
            if call_num < total_calls and len(all_pairs) < total:
                time.sleep(3)

        else:
            continue
        break  # inner loop hit the target — exit outer loop too

    if not all_pairs:
        log.error("No synthetic pairs were generated — check GROQ_API_KEY and network.")
        return 0

    log.info("Done — %d/%d pairs in %s", len(all_pairs), total, out_path)
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
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(records: list[dict], threshold: float = 0.92) -> list[dict]:
    """Remove near-duplicate records using sentence-transformer cosine similarity.

    Embeds each record as ``instruction + " " + output`` with
    all-MiniLM-L6-v2, then greedily keeps the first occurrence of any
    cluster where pairwise cosine similarity exceeds *threshold*.

    Args:
        records:   cleaned records from load_raw()
        threshold: cosine similarity cutoff — pairs above this are duplicates

    Returns:
        Deduplicated list (order preserved, first occurrence kept).
    """
    if not records:
        return records

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.warning(
            "sentence-transformers not installed — skipping deduplication. "
            "Run: pip install sentence-transformers"
        )
        return records

    import numpy as np

    log.info("Loading all-MiniLM-L6-v2 for deduplication …")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [r["instruction"] + " " + r["output"] for r in records]
    log.info("Embedding %d records …", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalised → dot product == cosine sim
    )

    kept_indices: list[int] = []
    kept_vecs: list = []          # grows as we accept records

    for i, emb in enumerate(embeddings):
        if not kept_vecs:
            kept_indices.append(i)
            kept_vecs.append(emb)
            continue

        # Cosine similarities against every already-kept embedding
        sims = np.stack(kept_vecs) @ emb   # shape (K,)
        if float(sims.max()) <= threshold:
            kept_indices.append(i)
            kept_vecs.append(emb)

    removed = len(records) - len(kept_indices)
    print(f"Deduplication: {removed} duplicates removed, {len(kept_indices)} records kept "
          f"(threshold={threshold}).")
    log.info("Deduplication complete — removed %d, kept %d.", removed, len(kept_indices))

    return [records[i] for i in kept_indices]


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
        help=(
            "Generate 500 synthetic Indian tax/legal Q&A pairs via Groq API "
            "(llama-3.3-70b-versatile) and save to data/raw/synthetic.json "
            "before running the cleaning and splitting pipeline. "
            "Requires GROQ_API_KEY env variable. Skipped if the file already exists."
        ),
    )
    parser.add_argument(
        "--synthetic_out",
        default=str(ROOT / "data" / "raw" / "synthetic.json"),
        help="Destination path for the generated synthetic data (default: data/raw/synthetic.json)",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=_TOTAL_PAIRS,
        help=f"Number of synthetic pairs to generate (default: {_TOTAL_PAIRS})",
    )
    parser.add_argument(
        "--dedup_threshold",
        type=float,
        default=0.92,
        help="Cosine similarity cutoff for deduplication (default: 0.92). "
             "Pairs above this value are considered duplicates.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Synthetic data generation (runs before load_raw) ---
    if args.synthetic:
        synthetic_path = Path(args.synthetic_out)
        # Always call generate — it resumes from existing pairs automatically
        # and exits early if the target is already met.
        n = generate_synthetic_data(synthetic_path, total=args.total)
        if n == 0:
            log.error("Synthetic generation produced no pairs — check GROQ_API_KEY and network.")
            sys.exit(1)

    # --- Load, clean, deduplicate, split ---
    records = load_raw(raw_dir)
    if not records:
        log.error("No valid records found in %s — add .jsonl / .json / .csv files first.", raw_dir)
        sys.exit(1)

    records = deduplicate(records, threshold=args.dedup_threshold)

    # Save deduplicated records before splitting so the clean set is inspectable
    deduped_path = out_dir / "deduped.json"
    deduped_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Deduplicated dataset saved to %s (%d records)", deduped_path, len(records))

    splits = split_dataset(records)
    for split, ds in splits.items():
        log.info("  %s: %d samples", split, len(ds))

    splits.save_to_disk(str(out_dir))
    log.info("Dataset saved to %s", out_dir)


if __name__ == "__main__":
    main()
