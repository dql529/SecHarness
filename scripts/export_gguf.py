#!/usr/bin/env python3
"""
export_gguf.py — Merge LoRA adapter into base model and export for Ollama.

Steps:
1. Load base model + LoRA adapter
2. Merge adapter weights into base model
3. Save merged model as HF format
4. Convert to GGUF using llama.cpp's convert script (or manual)
5. Create Ollama Modelfile and register
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("export")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--adapter-path", default="project/models/qlora_unsw/adapter")
    ap.add_argument("--output-dir", default="project/models/qlora_unsw/merged")
    ap.add_argument("--ollama-name", default="secids-3b")
    args = ap.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    adapter_path = PROJECT_ROOT / args.adapter_path.replace("project/", "")
    output_dir = PROJECT_ROOT / args.output_dir.replace("project/", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base model in fp16
    log.info("Loading base model: %s", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    # Load adapter
    log.info("Loading LoRA adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(model, str(adapter_path))

    # Merge
    log.info("Merging adapter...")
    model = model.merge_and_unload()

    # Save
    log.info("Saving merged model to %s", output_dir)
    model.save_pretrained(str(output_dir), safe_serialization=True)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(output_dir))
    log.info("Merged model saved.")

    # Try to convert to GGUF using llama.cpp convert script
    # First check if convert_hf_to_gguf.py is available
    gguf_path = output_dir / f"model-f16.gguf"

    # Try using the convert script from llama-cpp-python or llama.cpp
    convert_scripts = [
        "convert_hf_to_gguf",  # llama-cpp-python
        "python -m llama_cpp.convert",
    ]

    converted = False
    for script in convert_scripts:
        try:
            cmd = f"{script} {output_dir} --outfile {gguf_path} --outtype f16"
            log.info("Trying: %s", cmd)
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                log.info("GGUF conversion successful: %s", gguf_path)
                converted = True
                break
            else:
                log.warning("Failed: %s", result.stderr[:200])
        except Exception as e:
            log.warning("Script not available: %s", e)

    if not converted:
        # Try installing llama-cpp-python for conversion
        log.info("Attempting pip install llama-cpp-python for conversion...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "llama-cpp-python"],
                capture_output=True, timeout=120,
            )
            cmd = f"{sys.executable} -m llama_cpp.convert {output_dir} --outfile {gguf_path} --outtype f16"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                log.info("GGUF conversion successful: %s", gguf_path)
                converted = True
        except Exception as e:
            log.warning("Could not convert to GGUF: %s", e)

    if not converted:
        log.info("GGUF conversion not available. Trying alternative: llama.cpp convert_hf_to_gguf.py")
        log.info("You can manually convert with:")
        log.info("  python convert_hf_to_gguf.py %s --outfile %s --outtype f16", output_dir, gguf_path)

    # Create Ollama Modelfile
    modelfile_path = output_dir / "Modelfile"
    if converted and gguf_path.exists():
        modelfile_content = f"""FROM {gguf_path}

TEMPLATE \"\"\"{{{{ if .System }}}}<|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>{{{{ end }}}}{{{{ if .Prompt }}}}<|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|>{{{{ end }}}}<|start_header_id|>assistant<|end_header_id|>

{{{{ .Response }}}}<|eot_id|>\"\"\"

PARAMETER temperature 0.1
PARAMETER num_predict 256
PARAMETER stop "<|eot_id|>"
"""
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)
        log.info("Modelfile written: %s", modelfile_path)

        # Register with Ollama
        log.info("Registering with Ollama as '%s'...", args.ollama_name)
        try:
            result = subprocess.run(
                ["ollama", "create", args.ollama_name, "-f", str(modelfile_path)],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                log.info("Model registered with Ollama: %s", args.ollama_name)
            else:
                log.warning("Ollama create failed: %s", result.stderr[:200])
        except Exception as e:
            log.warning("Could not register with Ollama: %s", e)
    else:
        # Create Modelfile pointing to merged HF dir (Ollama can load HF models)
        modelfile_content = f"""FROM {output_dir}

PARAMETER temperature 0.1
PARAMETER num_predict 256
"""
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)
        log.info("Modelfile written (HF format): %s", modelfile_path)
        log.info("Register with: ollama create %s -f %s", args.ollama_name, modelfile_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
