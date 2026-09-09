#!/usr/bin/env python3
"""
train_qlora_v2.py — QLoRA fine-tuning on tool-use trajectory data.

Unlike train_qlora.py (v1) which trains on direct JSON verdicts,
this script trains on multi-turn tool-use conversations extracted
from E3 audit logs by extract_trajectories.py.

Input: JSONL with {"messages": [...]} per line (already in chat format).
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_qlora_v2")


def load_trajectories(path: str, max_samples: int = 0, seed: int = 42) -> Dataset:
    """Load trajectory JSONL (pre-formatted messages) into HF Dataset."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    log.info("Loaded %d trajectories from %s", len(records), path)

    if max_samples > 0 and len(records) > max_samples:
        random.seed(seed)
        records = random.sample(records, max_samples)
        log.info("Subsampled to %d trajectories", len(records))

    # Validate
    valid = []
    for rec in records:
        msgs = rec.get("messages", [])
        if len(msgs) >= 5:
            # Check last assistant turn has classify
            last_assistant = [m for m in msgs if m["role"] == "assistant"][-1]
            try:
                obj = json.loads(last_assistant["content"])
                if obj.get("tool") == "classify":
                    valid.append(rec)
                    continue
            except (json.JSONDecodeError, KeyError):
                pass
        log.warning("Skipping invalid trajectory (%d messages)", len(msgs))

    log.info("Valid trajectories: %d / %d", len(valid), len(records))
    return Dataset.from_list(valid)


def main():
    ap = argparse.ArgumentParser(description="QLoRA v2: fine-tune on tool-use trajectories")
    ap.add_argument("--trajectories", type=str, required=True,
                     help="Path to trajectory JSONL from extract_trajectories.py")
    ap.add_argument("--output-dir", type=str, default="project/models/qlora_unsw_v2")
    ap.add_argument("--base-model", type=str, default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--max-train-samples", type=int, default=0,
                     help="Max training samples (0=all)")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--num-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / args.output_dir.replace("project/", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Detect device (CUDA on gpu-box, MPS on Mac)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        raise RuntimeError("Neither CUDA nor MPS available")
    log.info("Using device: %s", device)

    # --- Load tokenizer ---
    log.info("Loading tokenizer: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Patch chat template to add {% generation %} markers for assistant_only_loss.
    # Llama-3.2's default template lacks these; trl 1.0 requires them.
    # We inject them around the assistant content in the main message loop.
    original_template = tokenizer.chat_template
    old_assistant_block = (
        "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'"
        "+ message['content'] | trim + '<|eot_id|>' }}"
    )
    new_assistant_block = (
        "{%- if message['role'] == 'assistant' %}"
        "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
        "{% generation %}"
        "{{- message['content'] | trim + '<|eot_id|>' }}"
        "{% endgeneration %}"
        "{%- else %}"
        "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'"
        "+ message['content'] | trim + '<|eot_id|>' }}"
        "{%- endif %}"
    )
    patched_template = original_template.replace(old_assistant_block, new_assistant_block)
    if patched_template == original_template:
        log.warning("Could not patch chat template — assistant_only_loss may not work")
    else:
        tokenizer.chat_template = patched_template
        log.info("Patched chat template with {%% generation %%} markers")

    # --- Load model ---
    dtype = torch.float16 if device == "cuda" else torch.float32
    log.info("Loading model: %s (%s, %s)", args.base_model, dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map={"": device},
        torch_dtype=dtype,
    )

    # --- LoRA config ---
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    log.info("Trainable params: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    # --- Prepare dataset (already in messages format) ---
    train_ds = load_trajectories(
        args.trajectories,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    log.info("Training dataset: %d samples", len(train_ds))

    # --- Training config ---
    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="no",
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
        # Token masking: only compute loss on assistant responses (trl 1.0)
        assistant_only_loss=True,
    )
    log.info("Using assistant_only_loss=True (trl %s built-in masking)", SFTConfig.__module__)

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    # --- Train ---
    log.info("Starting training: %d epochs, batch=%d, grad_accum=%d, lr=%s, max_seq_len=%d",
             args.num_epochs, args.batch_size, args.grad_accum, args.lr, args.max_seq_len)
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
    metrics["max_seq_len"] = args.max_seq_len

    log.info("=== Training Complete ===")
    for k, v in sorted(metrics.items()):
        log.info("  %s: %s", k, v)

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
