"""evaluate.py — Post-training evaluation of the fine-tuned IndiaLex model.

Loads the merged fine-tuned model, runs inference on the test split, computes
ROUGE-L and BERTScore F1, compares against baseline_results.json, and saves
a delta summary to evals/ft_results.json.

Usage:
    python scripts/evaluate.py [--config configs/train_config.yaml]
                               [--splits_dir data/splits]
                               [--baseline evals/baseline_results.json]
                               [--out evals/ft_results.json]
                               [--max_samples 50]
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import logging
from pathlib import Path

import yaml
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


def build_prompt(instruction: str, inp: str) -> str:
    if inp:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def compute_rouge_l(predictions: list[str], references: list[str]) -> list[float]:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]


def compute_bertscore(predictions: list[str], references: list[str]) -> list[float]:
    import evaluate
    metric = evaluate.load("bertscore")
    results = metric.compute(predictions=predictions, references=references, lang="en")
    return results["f1"]


def print_summary_table(baseline: dict, ft: dict):
    print("\n" + "=" * 55)
    print(f"{'Metric':<25} {'Baseline':>12} {'Fine-tuned':>12}")
    print("-" * 55)
    for key in ("mean_rouge_l", "mean_bertscore_f1"):
        b = baseline.get(key, 0.0)
        f = ft.get(key, 0.0)
        delta = f - b
        sign = "+" if delta >= 0 else ""
        print(f"{key:<25} {b:>12.4f} {f:>12.4f}  ({sign}{delta:.4f})")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default=str(ROOT / "configs" / "train_config.yaml"))
    parser.add_argument("--splits_dir",  default=str(ROOT / "data" / "splits"))
    parser.add_argument("--baseline",    default=str(ROOT / "evals" / "baseline_results.json"))
    parser.add_argument("--out",         default=str(ROOT / "evals" / "ft_results.json"))
    parser.add_argument("--max_samples", type=int, default=50)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    output_dir = cfg["output"]["output_dir"]
    merged_path = str((ROOT / output_dir / "merged").resolve())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load baseline for comparison
    baseline_data = {}
    baseline_path = Path(args.baseline)
    if baseline_path.exists():
        baseline_data = json.loads(baseline_path.read_text())
        log.info("Loaded baseline: ROUGE-L=%.4f  BERTScore=%.4f",
                 baseline_data.get("mean_rouge_l", 0),
                 baseline_data.get("mean_bertscore_f1", 0))
    else:
        log.warning("Baseline file not found at %s — skipping delta comparison.", args.baseline)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading fine-tuned model from %s", merged_path)
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        merged_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    log.info("Loading test split from %s", args.splits_dir)
    dataset = load_from_disk(args.splits_dir)
    test_ds = dataset["test"]
    if args.max_samples and args.max_samples < len(test_ds):
        test_ds = test_ds.select(range(args.max_samples))
    log.info("Running inference on %d samples …", len(test_ds))

    predictions, references, sample_results = [], [], []

    for i, row in enumerate(test_ds):
        prompt = build_prompt(row["instruction"], row["input"])
        pred = run_inference(model, tokenizer, prompt)
        predictions.append(pred)
        references.append(row["output"])
        if (i + 1) % 10 == 0:
            log.info("  %d / %d done", i + 1, len(test_ds))

    log.info("Computing ROUGE-L …")
    rouge_scores = compute_rouge_l(predictions, references)
    log.info("Computing BERTScore …")
    bert_scores = compute_bertscore(predictions, references)

    for i, row in enumerate(test_ds):
        sample_results.append({
            "instruction": row["instruction"],
            "input": row["input"],
            "reference": references[i],
            "prediction": predictions[i],
            "rouge_l": round(rouge_scores[i], 4),
            "bertscore_f1": round(bert_scores[i], 4),
        })

    ft_summary = {
        "model": merged_path,
        "num_samples": len(test_ds),
        "mean_rouge_l": round(sum(rouge_scores) / len(rouge_scores), 4),
        "mean_bertscore_f1": round(sum(bert_scores) / len(bert_scores), 4),
        "delta_vs_baseline": {},
        "samples": sample_results,
    }

    if baseline_data:
        ft_summary["delta_vs_baseline"] = {
            "rouge_l": round(ft_summary["mean_rouge_l"] - baseline_data.get("mean_rouge_l", 0), 4),
            "bertscore_f1": round(ft_summary["mean_bertscore_f1"] - baseline_data.get("mean_bertscore_f1", 0), 4),
        }

    out_path.write_text(json.dumps(ft_summary, indent=2, ensure_ascii=False))
    log.info("FT results saved to %s", out_path)

    print_summary_table(baseline_data, ft_summary)


if __name__ == "__main__":
    main()
