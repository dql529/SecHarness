#!/usr/bin/env python3
"""
train_qlora.py — QLoRA fine-tuning of Llama-3.2-3B-Instruct on UNSW-NB15.

Task: Given kv-format traffic text → predict JSON {verdict, attack_type, confidence, reasoning}
Uses: PEFT LoRA + bitsandbytes 4-bit NF4 quantization on Apple MPS.

ReAct-IDS config: rank=16, alpha=32, dropout=0.1, AdamW lr=5e-4, 3 epochs, NF4.
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_qlora")

# ---------------------------------------------------------------------------
# Prompt template — matches alpha_agent SYSTEM_PROMPT output format
# ---------------------------------------------------------------------------
SYSTEM_MSG = """\
You are a network intrusion detection expert specializing in behavioral analysis.

Your task: analyze a network traffic record (key=value format) and determine whether it represents benign activity or an attack.

Analysis approach — think step by step:
1. Parse each feature and its value (low/medium/high/very_high/zero/missing).
2. Identify behavioral anomalies: unusual protocol-state-service combinations, \
extreme byte/packet ratios, suspicious timing patterns, abnormal connection counts.
3. Consider what attack behaviors these features could indicate.
4. Make your verdict with a calibrated confidence score.

Respond ONLY with a JSON object (no markdown, no extra text):
{
  "verdict": "benign" or "attack",
  "attack_type": null or a short attack category name (e.g., "DoS", "Reconnaissance", "Exploits"),
  "confidence": a float between 0.0 and 1.0,
  "reasoning": "brief explanation of your analysis (1-3 sentences)"
}"""

USER_TEMPLATE = "Analyze this network traffic record:\n{traffic_text}"


def _label_to_response(label_name: str) -> str:
    """Convert ground-truth label to a target JSON response."""
    is_attack = label_name.lower() not in ("normal", "benign")
    if is_attack:
        # Map to attack type reasoning
        attack_reasonings = {
            # UNSW-NB15
            "exploits": "Features show exploit-like behavior: abnormal byte ratios and suspicious connection patterns suggest vulnerability exploitation.",
            "recon": "Traffic exhibits reconnaissance patterns: scanning-like connection behavior with high connection counts and low data transfer.",
            "reconnaissance": "Traffic exhibits reconnaissance patterns: scanning-like connection behavior with high connection counts and low data transfer.",
            "dos": "High-volume traffic with extreme packet rates and asymmetric byte patterns indicate a denial-of-service attack.",
            "backdoor": "Suspicious persistent connection with unusual service-protocol combination suggests backdoor activity.",
            "backdoors": "Suspicious persistent connection with unusual service-protocol combination suggests backdoor activity.",
            # CIC-IDS2017
            "ddos": "Distributed flood-based traffic with high packet rates, extreme flow volumes, and coordinated source patterns indicate a DDoS attack.",
            "bruteforce": "Repeated authentication attempts with rapid connection cycling, low data transfer per flow, and high failure rates suggest brute-force attack.",
            "portscan": "Systematic port probing with many short-lived connections, minimal payload, and sequential destination port patterns indicate port scanning.",
        }
        reasoning = attack_reasonings.get(
            label_name.lower(),
            f"Traffic patterns indicate {label_name} attack behavior based on anomalous feature combinations."
        )
        return json.dumps({
            "verdict": "attack",
            "attack_type": label_name,
            "confidence": round(random.uniform(0.80, 0.95), 2),
            "reasoning": reasoning,
        })
    else:
        return json.dumps({
            "verdict": "benign",
            "attack_type": None,
            "confidence": round(random.uniform(0.80, 0.95), 2),
            "reasoning": "Traffic features appear within normal ranges with expected protocol-service combinations and balanced byte ratios.",
        })


def build_chat_messages(traffic_text: str, label_name: str) -> list[dict]:
    """Build chat messages for a single training sample."""
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": USER_TEMPLATE.format(traffic_text=traffic_text)},
        {"role": "assistant", "content": _label_to_response(label_name)},
    ]


def prepare_dataset(csv_path: str, text_col: str, label_col: str,
                     max_samples: int = 0, seed: int = 42,
                     known_classes: set | None = None) -> Dataset:
    """Load CSV and convert to HF Dataset with chat-formatted text."""
    df = pd.read_csv(csv_path)
    log.info("Loaded %d rows from %s", len(df), csv_path)

    # Filter to known classes only
    if known_classes is None:
        known_classes = {"Normal", "Exploits", "Recon", "DoS", "Backdoor", "Backdoors", "Reconnaissance"}
    df = df[df[label_col].isin(known_classes)].reset_index(drop=True)
    log.info("After filtering to known classes: %d rows", len(df))

    if max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
        log.info("Subsampled to %d rows", len(df))

    # Balance classes: undersample majority to max 2x of minority
    class_counts = df[label_col].value_counts()
    min_count = class_counts.min()
    max_per_class = max(min_count * 3, 2000)  # at most 3x minority or 2000
    balanced_dfs = []
    for cls in class_counts.index:
        cls_df = df[df[label_col] == cls]
        if len(cls_df) > max_per_class:
            cls_df = cls_df.sample(n=max_per_class, random_state=seed)
        balanced_dfs.append(cls_df)
    df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    log.info("After balancing: %d rows, distribution: %s",
             len(df), df[label_col].value_counts().to_dict())

    random.seed(seed)
    records = []
    for _, row in df.iterrows():
        messages = build_chat_messages(row[text_col], row[label_col])
        records.append({"messages": messages})

    return Dataset.from_list(records)


def main():
    ap = argparse.ArgumentParser(description="QLoRA fine-tuning on UNSW-NB15")
    ap.add_argument("--train-csv", type=str,
                     default="project/data/processed/unsw_nb15/zeroday_split/train_known.csv")
    ap.add_argument("--val-csv", type=str,
                     default="project/data/processed/unsw_nb15/zeroday_split/val_known.csv")
    ap.add_argument("--output-dir", type=str, default="project/models/qlora_unsw")
    ap.add_argument("--base-model", type=str, default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--text-col", type=str, default="text",
                     help="Column name for text features")
    ap.add_argument("--label-col", type=str, default="label_name",
                     help="Column name for labels")
    ap.add_argument("--known-classes", type=str, default=None,
                     help="Comma-separated known classes (default: UNSW classes)")
    ap.add_argument("--max-train-samples", type=int, default=10000,
                     help="Max training samples (0=all, default=10000 for speed)")
    ap.add_argument("--max-val-samples", type=int, default=1000)
    ap.add_argument("--num-epochs", type=int, default=3, dest="epochs")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-4bit", action="store_true", default=True,
                     help="Use fp16/fp32 instead of 4-bit (default on MPS)")
    ap.add_argument("--use-4bit", action="store_true", help="Force 4-bit (CUDA only)")
    args = ap.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    train_csv = PROJECT_ROOT / args.train_csv.replace("project/", "")
    val_csv = PROJECT_ROOT / args.val_csv.replace("project/", "")
    output_dir = PROJECT_ROOT / args.output_dir.replace("project/", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Load tokenizer ---
    log.info("Loading tokenizer: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load model ---
    use_4bit = args.use_4bit and not args.no_4bit
    log.info("Loading model: %s (4bit=%s)", args.base_model, use_4bit)
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float32,
        )
    else:
        # fp32 on MPS for training stability (3B model ≈ 12GB, fits in 128GB)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            torch_dtype=torch.float32,
        )

    # --- LoRA config ---
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    log.info("Trainable params: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    # --- Parse known classes ---
    if args.known_classes:
        known_classes = set(c.strip() for c in args.known_classes.split(","))
    else:
        known_classes = None  # default UNSW classes

    # --- Prepare datasets ---
    log.info("Preparing training dataset...")
    train_ds = prepare_dataset(
        str(train_csv), args.text_col, args.label_col,
        max_samples=args.max_train_samples, seed=args.seed,
        known_classes=known_classes,
    )
    log.info("Training dataset: %d samples", len(train_ds))

    val_ds = None
    if val_csv.exists():
        log.info("Preparing validation dataset...")
        val_ds = prepare_dataset(
            str(val_csv), args.text_col, args.label_col,
            max_samples=args.max_val_samples, seed=args.seed,
            known_classes=known_classes,
        )
        log.info("Validation dataset: %d samples", len(val_ds))

    # --- Training args (trl 1.0 uses SFTConfig) ---
    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch" if val_ds else "no",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=False,
        fp16=False,
        dataloader_pin_memory=False,
        report_to="none",
        seed=args.seed,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        max_length=args.max_seq_len,
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    # --- Train ---
    log.info("Starting training: %d epochs, batch=%d, grad_accum=%d, lr=%s",
             args.epochs, args.batch_size, args.grad_accum, args.lr)
    t0 = time.time()
    train_result = trainer.train()
    train_time = time.time() - t0

    # --- Save adapter ---
    adapter_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    log.info("Adapter saved to: %s", adapter_path)

    # --- Log results ---
    metrics = train_result.metrics
    metrics["train_time_sec"] = round(train_time, 1)
    metrics["train_samples"] = len(train_ds)
    if val_ds:
        eval_metrics = trainer.evaluate()
        metrics.update(eval_metrics)

    log.info("=== Training Complete ===")
    for k, v in sorted(metrics.items()):
        log.info("  %s: %s", k, v)

    # Save training log
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info("Training log saved: %s", log_path)

    # Save loss history
    loss_history = [
        {"step": entry["step"], "loss": entry.get("loss"), "eval_loss": entry.get("eval_loss")}
        for entry in trainer.state.log_history
        if "loss" in entry or "eval_loss" in entry
    ]
    loss_path = output_dir / "loss_history.json"
    with open(loss_path, "w") as f:
        json.dump(loss_history, f, indent=2)
    log.info("Loss history saved: %s", loss_path)


if __name__ == "__main__":
    main()
