# IndiaLex-FT

Supervised fine-tuning (SFT) of Llama 3.2 on Indian legal corpora using LoRA/PEFT.
Complements the [ClauseIQ RAG system](../RAG/) with a parameter-efficient fine-tuned model
evaluated against ROUGE-L, BERTScore, and an LLM judge.

---

## Project Structure

```
indialex-ft/
├── data/
│   ├── raw/          # Place .jsonl or .csv files here
│   ├── processed/    # Intermediate cleaned data
│   └── splits/       # HuggingFace Dataset (train/val/test)
├── configs/
│   └── train_config.yaml
├── scripts/
│   ├── data_pipeline.py
│   ├── baseline_eval.py
│   ├── train.py
│   └── evaluate.py
├── evals/
│   ├── llm_judge.py
│   ├── baseline_results.json  # generated
│   ├── ft_results.json        # generated
│   └── judge_results.json     # generated
├── app/
│   ├── api.py        # FastAPI inference server
│   └── ui.py         # Streamlit UI
├── outputs/          # Saved adapter + merged model (generated)
└── requirements.txt
```

---

## Setup

```bash
cd indialex-ft
pip install -r requirements.txt
```

Requires a GPU with at least 16 GB VRAM for 4-bit training, or 8 GB for inference only.

---

## Data Format

Place `.jsonl` or `.csv` files in `data/raw/`. Each record must have:

```json
{"instruction": "Explain Section 302 IPC.", "input": "", "output": "Section 302 of IPC..."}
```

For CSV files, use columns: `instruction`, `input`, `output`.

---

## Usage

### 1. Prepare data
```bash
python scripts/data_pipeline.py
```
Splits data 80/10/10 into `data/splits/`.

### 2. Baseline evaluation (before fine-tuning)
```bash
python scripts/baseline_eval.py --max_samples 50
```
Saves results to `evals/baseline_results.json`.

### 3. Fine-tune with LoRA
```bash
python scripts/train.py
```
Saves LoRA adapter to `outputs/indialex-ft/adapter/` and merged model to `outputs/indialex-ft/merged/`.

### 4. Post-training evaluation
```bash
python scripts/evaluate.py --max_samples 50
```
Compares fine-tuned vs baseline, saves delta to `evals/ft_results.json`.

### 5. LLM judge scoring
```bash
# Requires Ollama running locally with llama3.2
python evals/llm_judge.py --results evals/ft_results.json
```
Saves per-sample judge scores to `evals/judge_results.json`.

### 6. Start inference API
```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

### 7. Launch Streamlit UI
```bash
streamlit run app/ui.py
```

---

## Configuration

All hyperparameters are in `configs/train_config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `model.model_name` | `meta-llama/Llama-3.2-3B-Instruct` | Base model |
| `lora.lora_r` | `16` | LoRA rank |
| `lora.lora_alpha` | `32` | LoRA scaling |
| `training.num_epochs` | `3` | Training epochs |
| `training.learning_rate` | `2e-4` | AdamW LR |
| `training.batch_size` | `4` | Per-device batch |
| `training.max_seq_length` | `2048` | Max token length |
| `output.output_dir` | `./outputs/indialex-ft` | Save path |
| `logging.wandb_project` | `indialex-ft` | W&B project name |

---

## Evaluation Metrics

| Metric | Tool | Description |
|--------|------|-------------|
| ROUGE-L | `rouge-score` | Longest common subsequence F1 |
| BERTScore F1 | `evaluate` | Semantic similarity via BERT embeddings |
| Correctness | LLM judge (Ollama) | Factual accuracy vs ground truth |
| Faithfulness | LLM judge | Grounded in context, no hallucination |
| Helpfulness | LLM judge | Clarity and actionability |
| Legal Accuracy | LLM judge | Correct interpretation of Indian law |
