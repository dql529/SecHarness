"""
alpha_agent_hf.py — Agent-Alpha using HuggingFace transformers for local inference.

Loads a base model + LoRA adapter for fine-tuned IDS classification.
Supports both 4-bit quantized (QLoRA) and fp16 loading.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .base_agent import BaseAgent, AgentVerdict

logger = logging.getLogger(__name__)

# Same system prompt as alpha_agent.py (Ollama version)
SYSTEM_PROMPT = """\
You are a network intrusion detection expert specializing in behavioral analysis.

Your task: analyze a network traffic record (key=value format) and determine whether it represents benign activity or an attack.

Analysis approach — think step by step:
1. Parse each feature and its value (low/medium/high/very_high/zero/missing).
2. Identify behavioral anomalies: unusual protocol-state-service combinations, \
extreme byte/packet ratios, suspicious timing patterns, abnormal connection counts.
3. Consider what attack behaviors these features could indicate \
(e.g., high sbytes + zero dbytes → data exfiltration; many connections + low duration → scanning).
4. Make your verdict with a calibrated confidence score.

Respond ONLY with a JSON object (no markdown, no extra text):
{
  "verdict": "benign" or "attack",
  "attack_type": null or a short attack category name (e.g., "DoS", "Reconnaissance", "Exploits"),
  "confidence": a float between 0.0 and 1.0,
  "reasoning": "brief explanation of your analysis (1-3 sentences)"
}"""

USER_TEMPLATE = "Analyze this network traffic record:\n{traffic_text}"


def _parse_json_response(content: str) -> dict | None:
    """Extract JSON from model output, handling truncation."""
    # Try full JSON match
    json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: extract fields from truncated JSON
    start = content.find("{")
    if start < 0:
        return None
    fragment = content[start:]
    result = {}
    m = re.search(r'"verdict"\s*:\s*"(benign|attack)"', fragment, re.IGNORECASE)
    if m:
        result["verdict"] = m.group(1).lower()
    m = re.search(r'"attack_type"\s*:\s*"([^"]*)"', fragment)
    if m:
        result["attack_type"] = m.group(1)
    elif re.search(r'"attack_type"\s*:\s*null', fragment):
        result["attack_type"] = None
    m = re.search(r'"confidence"\s*:\s*([0-9.]+)', fragment)
    if m:
        result["confidence"] = m.group(1)
    m = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)', fragment)
    if m:
        result["reasoning"] = m.group(1)
    return result if "verdict" in result else None


class AlphaAgentHF(BaseAgent):
    """Agent-Alpha using HuggingFace transformers with optional LoRA adapter."""

    def __init__(
        self,
        base_model: str = "unsloth/Llama-3.2-3B-Instruct",
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = True,
        temperature: float = 0.1,
        max_new_tokens: int = 256,
        max_input_length: int = 512,
        device: str = "auto",
    ):
        super().__init__(name="Agent-Alpha-HF")
        self.base_model_name = base_model
        self.adapter_path = adapter_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_input_length = max_input_length

        # Load tokenizer
        tokenizer_source = adapter_path if adapter_path else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map=device,
                torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map=device,
                torch_dtype=torch.float16,
            )

        # Load LoRA adapter if provided (keep as PeftModel, don't merge to preserve quantization)
        if adapter_path:
            logger.info("Loading LoRA adapter from %s", adapter_path)
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            logger.info("LoRA adapter loaded (inference mode)")

        self.model.eval()
        self._device = next(self.model.parameters()).device
        logger.info("AlphaAgentHF ready: model=%s, adapter=%s, device=%s",
                     base_model, adapter_path, self._device)

    def _build_prompt(self, traffic_text: str) -> str:
        """Build chat prompt using tokenizer's chat template."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(traffic_text=traffic_text)},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.inference_mode()
    def _generate(self, traffic_text: str) -> tuple[str, int]:
        """Generate response for a single traffic record. Returns (text, token_count)."""
        prompt = self._build_prompt(traffic_text)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_length)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        prompt_len = inputs["input_ids"].shape[1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=max(self.temperature, 0.01),
            do_sample=self.temperature > 0,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated_ids = outputs[0][prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text, len(generated_ids)

    def analyze(self, traffic_text: str) -> AgentVerdict:
        """Analyze a single traffic record."""
        t0 = time.perf_counter()
        try:
            content, token_count = self._generate(traffic_text)
            parsed = _parse_json_response(content)

            if parsed and "verdict" in parsed:
                verdict = parsed["verdict"].lower().strip()
                if verdict not in ("benign", "attack"):
                    verdict = "benign"
                return AgentVerdict(
                    verdict=verdict,
                    attack_type=parsed.get("attack_type"),
                    confidence=float(parsed.get("confidence", 0.5)),
                    reasoning=str(parsed.get("reasoning", "")),
                    agent_name=self.name,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    token_count=token_count,
                )
            else:
                logger.warning("Failed to parse response: %s", content[:200])
                return AgentVerdict(
                    verdict="benign", confidence=0.0,
                    reasoning=f"[PARSE_FAIL] raw: {content[:300]}",
                    agent_name=self.name,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    token_count=token_count,
                )
        except Exception as e:
            logger.error("HF inference error: %s", e)
            return AgentVerdict(
                verdict="benign", confidence=0.0,
                reasoning=f"[ERROR] {e}",
                agent_name=self.name,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

    def analyze_batch(self, traffic_texts: List[str]) -> List[AgentVerdict]:
        """Sequential batch — HF generate doesn't easily batch variable-length chat prompts."""
        return [self.analyze(t) for t in traffic_texts]

    def __repr__(self) -> str:
        return f"<AlphaAgentHF model={self.base_model_name!r} adapter={self.adapter_path!r}>"
