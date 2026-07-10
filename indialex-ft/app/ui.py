"""ui.py — Streamlit UI for IndiaLex-FT inference.

Calls the FastAPI /generate endpoint (api.py) or runs inference directly
if the API is not reachable.

Usage:
    streamlit run app/ui.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import os

import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="IndiaLex-FT",
    page_icon="⚖️",
    layout="wide",
)

st.title("IndiaLex-FT — Indian Legal AI")
st.caption("Fine-tuned Llama 3.2 on Indian legal corpora")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value=API_URL)
    max_tokens = st.slider("Max new tokens", min_value=64, max_value=1024, value=256, step=64)
    use_api = st.toggle("Use API server", value=True)

    st.divider()
    st.markdown("**Model path** (direct inference only)")
    _outputs_root = ROOT / "outputs"
    model_path_input = st.text_input(
        "Model path",
        value=str(_outputs_root / "indialex-ft" / "merged"),
        help=f"Path to the merged fine-tuned model directory (must be under {_outputs_root})",
    )
    if not Path(model_path_input).resolve().is_relative_to(_outputs_root.resolve()):
        st.error("Model path must be inside the outputs/ directory.")
        model_path_input = str(_outputs_root / "indialex-ft" / "merged")

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def _load_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
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


def _infer_via_api(instruction: str, inp: str, max_new_tokens: int, url: str) -> str:
    import requests
    resp = requests.post(
        f"{url}/generate",
        json={"instruction": instruction, "input": inp, "max_new_tokens": max_new_tokens},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["output"]


def _infer_direct(instruction: str, inp: str, max_new_tokens: int, model_path: str) -> str:
    import torch
    model, tokenizer = _load_model(model_path)
    if inp:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
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
# Main UI
# ---------------------------------------------------------------------------

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    instruction = st.text_area(
        "Instruction",
        placeholder="Explain Section 420 of IPC and its punishment.",
        height=120,
    )
    context_input = st.text_area(
        "Context / Input (optional)",
        placeholder="Paste any relevant legal document or case details here…",
        height=150,
    )
    submit = st.button("Generate", type="primary", use_container_width=True)

with col2:
    st.subheader("Output")
    output_placeholder = st.empty()

if submit:
    if not instruction.strip():
        st.warning("Please enter an instruction.")
    else:
        with st.spinner("Generating…"):
            try:
                if use_api:
                    result = _infer_via_api(instruction, context_input, max_tokens, api_url)
                else:
                    result = _infer_direct(instruction, context_input, max_tokens, model_path_input)
                output_placeholder.text_area("Response", value=result, height=350)
            except Exception as exc:
                st.error(f"Error: {exc}")
                if use_api:
                    st.info("Tip: make sure the API server is running (`uvicorn app.api:app --reload`)")
