"""train.py — LoRA SFT fine-tuning for IndiaLex-FT.

Loads the base model in 4-bit (bitsandbytes NF4), applies LoRA via PEFT,
and trains with TRL SFTTrainer. Logs to W&B and saves adapter + merged model.

Usage:
    python scripts/train.py [--config configs/train_config.yaml]
                            [--splits_dir data/splits]
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import logging
import os
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(row: dict) -> str:
    inp = row.get("input", "").strip()
    if inp:
        return (
            f"### Instruction:\n{row['instruction']}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{row['output']}"
        )
    return (
        f"### Instruction:\n{row['instruction']}\n\n"
        f"### Response:\n{row['output']}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default=str(ROOT / "configs" / "train_config.yaml"))
    parser.add_argument("--splits_dir", default=str(ROOT / "data" / "splits"))
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config file not found: %s", config_path)
        raise SystemExit(1)
    cfg = yaml.safe_load(config_path.read_text())

    model_name = cfg["model"]["model_name"]
    lora_r           = cfg["lora"]["lora_r"]
    lora_alpha       = cfg["lora"]["lora_alpha"]
    lora_dropout     = cfg["lora"]["lora_dropout"]
    batch_size       = cfg["training"]["batch_size"]
    grad_accum       = cfg["training"]["gradient_accumulation_steps"]
    num_epochs       = cfg["training"]["num_epochs"]
    lr               = cfg["training"]["learning_rate"]
    max_seq_len      = cfg["training"]["max_seq_length"]
    warmup_ratio     = cfg["training"]["warmup_ratio"]
    weight_decay     = cfg["training"]["weight_decay"]
    output_dir       = str((ROOT / cfg["output"]["output_dir"]).resolve())
    wandb_project    = cfg["logging"]["wandb_project"]

    os.environ.setdefault("WANDB_PROJECT", wandb_project)
    report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"

    import torch
    from datasets import load_from_disk
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    # --- 4-bit quantisation config ---
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    log.info("Loading base model: %s (4-bit NF4)", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # --- LoRA ---
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # --- Dataset ---
    log.info("Loading dataset from %s", args.splits_dir)
    dataset = load_from_disk(args.splits_dir)

    def _add_text(batch):
        batch["text"] = [format_prompt(
            {"instruction": i, "input": inp, "output": o}
        ) for i, inp, o in zip(batch["instruction"], batch["input"], batch["output"])]
        return batch

    train_ds = dataset["train"].map(_add_text, batched=True, remove_columns=dataset["train"].column_names)
    val_ds   = dataset["val"].map(_add_text,   batched=True, remove_columns=dataset["val"].column_names)

    # --- Training args (SFTConfig inherits TrainingArguments) ---
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=report_to,
        run_name="indialex-ft",
        max_seq_length=max_seq_len,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    log.info("Starting training …")
    trainer.train()

    # --- Save adapter ---
    adapter_path = (ROOT / output_dir / "adapter").resolve()
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    log.info("LoRA adapter saved to %s", adapter_path)

    # --- Merge and save full model ---
    log.info("Merging adapter into base model …")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM as AMCL
    base = AMCL.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    merged = PeftModel.from_pretrained(base, str(adapter_path))
    merged = merged.merge_and_unload()
    merged_path = (ROOT / output_dir / "merged").resolve()
    merged.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))
    log.info("Merged model saved to %s", merged_path)


if __name__ == "__main__":
    main()
