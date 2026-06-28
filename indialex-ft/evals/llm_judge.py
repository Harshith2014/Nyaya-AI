"""llm_judge.py — LLM-as-judge for IndiaLex fine-tuned model answers.

Scores model answers on four legal-domain dimensions (each 1-5):
  correctness     : factual accuracy vs. ground truth
  faithfulness    : answer grounded in context (no hallucination)
  helpfulness     : usefulness / clarity / actionability
  legal_accuracy  : correct interpretation of Indian law / statutes

Uses Ollama llama3.2 locally (same stack as ClauseIQ eval/llm_judge.py).
Lenient JSON parsing with regex fallback per dimension.

Usage:
    python evals/llm_judge.py --results evals/ft_results.json
                              [--out evals/judge_results.json]
                              [--model llama3.2]
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "llama3.2"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DimensionScore:
    score: int       # 1-5
    reasoning: str


@dataclass
class JudgeResult:
    instruction: str
    prediction: str
    correctness:    DimensionScore = field(default_factory=lambda: DimensionScore(0, ""))
    faithfulness:   DimensionScore = field(default_factory=lambda: DimensionScore(0, ""))
    helpfulness:    DimensionScore = field(default_factory=lambda: DimensionScore(0, ""))
    legal_accuracy: DimensionScore = field(default_factory=lambda: DimensionScore(0, ""))
    overall: float = 0.0
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "instruction":    self.instruction,
            "prediction":     self.prediction,
            "correctness":    {"score": self.correctness.score,    "reasoning": self.correctness.reasoning},
            "faithfulness":   {"score": self.faithfulness.score,   "reasoning": self.faithfulness.reasoning},
            "helpfulness":    {"score": self.helpfulness.score,    "reasoning": self.helpfulness.reasoning},
            "legal_accuracy": {"score": self.legal_accuracy.score, "reasoning": self.legal_accuracy.reasoning},
            "overall": round(self.overall, 3),
            "error":   self.error,
        }


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an expert evaluator of Indian legal AI systems.

Score the ANSWER on four dimensions, each from 1 (very poor) to 5 (excellent):

1. correctness     — Is the answer factually correct compared to the GROUND TRUTH?
2. faithfulness    — Is the answer grounded in the context with no hallucination?
3. helpfulness     — Is the answer clear, complete, and actionable?
4. legal_accuracy  — Does the answer correctly interpret Indian law, statutes, and precedents?

INSTRUCTION:
{instruction}

GROUND TRUTH:
{ground_truth}

ANSWER:
{answer}

Return ONLY valid JSON in this exact format (no extra text):
{{
  "correctness":    {{"score": <1-5>, "reasoning": "<one sentence>"}},
  "faithfulness":   {{"score": <1-5>, "reasoning": "<one sentence>"}},
  "helpfulness":    {{"score": <1-5>, "reasoning": "<one sentence>"}},
  "legal_accuracy": {{"score": <1-5>, "reasoning": "<one sentence>"}}
}}
JSON:"""


# ---------------------------------------------------------------------------
# LLM call (Ollama)
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 600},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["response"].strip()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_dim(obj: dict, key: str) -> DimensionScore:
    d = obj.get(key, {})
    score = max(1, min(5, int(d.get("score", 3))))
    return DimensionScore(score=score, reasoning=str(d.get("reasoning", "")).strip())


def _regex_fallback(raw: str, key: str) -> DimensionScore:
    pattern = rf'"{key}".*?"score"\s*:\s*(\d)'
    m = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
    score = int(m.group(1)) if m else 3
    return DimensionScore(score=score, reasoning="(parsed via regex fallback)")


# ---------------------------------------------------------------------------
# Judge function
# ---------------------------------------------------------------------------

def judge(
    instruction: str,
    prediction: str,
    ground_truth: str,
    model: str = DEFAULT_MODEL,
) -> JudgeResult:
    result = JudgeResult(instruction=instruction, prediction=prediction)
    prompt = _JUDGE_PROMPT.format(
        instruction=instruction[:400],
        ground_truth=ground_truth[:400],
        answer=prediction[:600],
    )
    try:
        raw = _call_ollama(prompt, model=model)
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                obj = json.loads(raw[start:end])
                result.correctness    = _parse_dim(obj, "correctness")
                result.faithfulness   = _parse_dim(obj, "faithfulness")
                result.helpfulness    = _parse_dim(obj, "helpfulness")
                result.legal_accuracy = _parse_dim(obj, "legal_accuracy")
            except json.JSONDecodeError:
                result.correctness    = _regex_fallback(raw, "correctness")
                result.faithfulness   = _regex_fallback(raw, "faithfulness")
                result.helpfulness    = _regex_fallback(raw, "helpfulness")
                result.legal_accuracy = _regex_fallback(raw, "legal_accuracy")
        else:
            result.error = "LLM returned no JSON block"
            return result

        scores = [
            result.correctness.score,
            result.faithfulness.score,
            result.helpfulness.score,
            result.legal_accuracy.score,
        ]
        result.overall = sum(scores) / len(scores)

    except Exception as exc:
        result.error = str(exc)

    return result


def judge_batch(samples: list[dict], model: str = DEFAULT_MODEL, verbose: bool = True) -> list[JudgeResult]:
    """Judge a list of samples. Each dict must have: instruction, prediction, reference."""
    results = []
    for i, s in enumerate(samples, 1):
        if verbose:
            print(f"  Judging [{i}/{len(samples)}]: {s['instruction'][:60]}…")
        r = judge(
            instruction=s["instruction"],
            prediction=s["prediction"],
            ground_truth=s.get("reference", ""),
            model=model,
        )
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(ROOT / "evals" / "ft_results.json"),
                        help="Path to ft_results.json (or baseline_results.json)")
    parser.add_argument("--out",     default=str(ROOT / "evals" / "judge_results.json"))
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run evaluate.py (or baseline_eval.py) first.")
        return

    data = json.loads(results_path.read_text())
    samples = data.get("samples", [])
    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Judging {len(samples)} samples with model '{args.model}' …")
    judge_results = judge_batch(samples, model=args.model)

    out_records = [r.as_dict() for r in judge_results]
    valid = [r for r in judge_results if not r.error]
    if valid:
        dims = ["correctness", "faithfulness", "helpfulness", "legal_accuracy"]
        means = {d: sum(getattr(r, d).score for r in valid) / len(valid) for d in dims}
        mean_overall = sum(r.overall for r in valid) / len(valid)

        print("\n--- LLM Judge Summary ---")
        for d, v in means.items():
            print(f"  {d:<18}: {v:.2f}")
        print(f"  {'overall':<18}: {mean_overall:.2f}")

        summary = {
            "model": args.model,
            "num_samples": len(samples),
            "num_valid": len(valid),
            "means": {k: round(v, 3) for k, v in means.items()},
            "mean_overall": round(mean_overall, 3),
            "samples": out_records,
        }
    else:
        summary = {"error": "All samples failed", "samples": out_records}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nJudge results saved to {out_path}")


if __name__ == "__main__":
    main()
