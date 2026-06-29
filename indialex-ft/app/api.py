"""api.py — FastAPI inference server for IndiaLex fine-tuned model.

Endpoints:
    POST /generate   — {instruction, input} → {output, model}
    POST /compare    — {instruction, input} → base vs FT outputs with latencies
    GET  /health     — liveness check

Usage:
    uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

log = logging.getLogger("uvicorn.error")

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "configs" / "train_config.yaml"

_feedback_lock = threading.Lock()
FEEDBACK_LOG = ROOT / "evals" / "feedback_log.jsonl"


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_state: dict = {}


def _load_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading model from %s …", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    output_dir = cfg["output"]["output_dir"]
    base_name  = cfg["model"]["model_name"]
    ft_path    = str(Path(output_dir) / "merged")

    # Always load base model
    _state["base_model"], _state["base_tokenizer"] = _load_model(base_name)
    _state["base_model_path"] = base_name
    log.info("Base model ready: %s", base_name)

    # Load fine-tuned model if available
    if Path(ft_path).exists():
        _state["ft_model"], _state["ft_tokenizer"] = _load_model(ft_path)
        _state["ft_model_path"] = ft_path
        log.info("Fine-tuned model ready: %s", ft_path)
    else:
        log.warning("Merged model not found at %s — FT model unavailable.", ft_path)
        _state["ft_model"] = _state["ft_tokenizer"] = _state["ft_model_path"] = None

    # /generate backward-compat: prefer FT, fall back to base
    _state["model"]      = _state["ft_model"]      or _state["base_model"]
    _state["tokenizer"]  = _state["ft_tokenizer"]  or _state["base_tokenizer"]
    _state["model_path"] = _state["ft_model_path"] or _state["base_model_path"]

    yield
    _state.clear()


app = FastAPI(title="IndiaLex-FT Inference API", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def _run_inference(model, tokenizer, instruction: str, inp: str, max_new_tokens: int) -> str:
    import torch

    inp = inp.strip()
    if inp:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

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


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    instruction: str
    input: Optional[str] = ""
    max_new_tokens: Optional[int] = 256


class GenerateResponse(BaseModel):
    output: str
    model: str


class CompareRequest(BaseModel):
    instruction: str
    input: Optional[str] = ""
    max_new_tokens: Optional[int] = 256


class CompareResponse(BaseModel):
    base_output: str
    base_model: str
    base_latency_s: float
    ft_output: Optional[str]
    ft_model: Optional[str]
    ft_latency_s: Optional[float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model": _state.get("model_path", "not loaded")}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    model     = _state.get("model")
    tokenizer = _state.get("tokenizer")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        text = _run_inference(model, tokenizer, req.instruction, req.input or "", req.max_new_tokens)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return GenerateResponse(output=text, model=_state["model_path"])


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    if _state.get("base_model") is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        t0 = time.perf_counter()
        base_out = _run_inference(
            _state["base_model"], _state["base_tokenizer"],
            req.instruction, req.input or "", req.max_new_tokens,
        )
        base_lat = round(time.perf_counter() - t0, 3)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Base model error: {exc}")

    ft_out = ft_lat = None
    if _state.get("ft_model"):
        try:
            t0 = time.perf_counter()
            ft_out = _run_inference(
                _state["ft_model"], _state["ft_tokenizer"],
                req.instruction, req.input or "", req.max_new_tokens,
            )
            ft_lat = round(time.perf_counter() - t0, 3)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"FT model error: {exc}")

    return CompareResponse(
        base_output=base_out,
        base_model=_state["base_model_path"],
        base_latency_s=base_lat,
        ft_output=ft_out,
        ft_model=_state.get("ft_model_path"),
        ft_latency_s=ft_lat,
    )
