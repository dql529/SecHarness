"""
llm_engine.py — Generic LLM inference engine for SecHarness v2.

Supports two backends:
  1. HuggingFace transformers (local inference, optional LoRA adapter)
  2. OpenAI-compatible API (vLLM, Ollama, etc.) — activated when
     base_model starts with "api://"

Not IDS-specific — system prompt is injected by the caller.

Heavy imports (torch, transformers, peft) are deferred to __init__ so that
modules importing LLMResponse for type hints don't trigger GPU library loading.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Result of a single LLM generation call."""
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMEngine:
    """Generic LLM inference engine.

    Supports:
    - HuggingFace local: variable system prompts, multi-turn, LoRA, 4-bit
    - OpenAI API: base_model="api://host:port/model_name"
      e.g. "api://localhost:8000/casperhansen/llama-3.3-70b-instruct-awq"
    """

    def __init__(
        self,
        base_model: str = "unsloth/Llama-3.2-3B-Instruct",
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = False,
        device: str = "auto",
        max_input_length: int = 2048,
        seed: Optional[int] = None,
    ):
        self.adapter_path = adapter_path
        self.max_input_length = max_input_length
        # Run-level sampling seed. API mode: sent per request (vLLM/Ollama honour
        # it, so identical prompts reproduce within a run and differ across seeds).
        # HF mode: torch.manual_seed once at construction. None = engine default.
        self.seed = seed
        self._hf_seeded = False
        self._api_mode = base_model.startswith("api://")

        if self._api_mode:
            self._init_api(base_model)
        else:
            self._init_hf(base_model, adapter_path, load_in_4bit, device)

    def _init_api(self, base_model: str) -> None:
        """Parse api://host:port/model_name and configure API backend."""
        # Format: api://host:port/model_name
        uri = base_model[len("api://"):]
        slash_idx = uri.find("/")
        if slash_idx == -1:
            raise ValueError(
                f"API model must be 'api://host:port/model_name', got: {base_model}"
            )
        self._api_base = f"http://{uri[:slash_idx]}"
        self._api_model = uri[slash_idx + 1:]
        self.base_model_name = self._api_model
        logger.info(
            "LLMEngine API mode: base=%s, model=%s",
            self._api_base, self._api_model,
        )

    def _init_hf(
        self,
        base_model: str,
        adapter_path: Optional[str],
        load_in_4bit: bool,
        device: str,
    ) -> None:
        """Load model via HuggingFace transformers."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.base_model_name = base_model

        # Load tokenizer
        tokenizer_source = adapter_path if adapter_path else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Resolve device_map: on Apple Silicon MPS, force all weights onto MPS
        # to avoid accelerate's disk offloading which is ~100x slower
        resolved_device_map = device
        if device == "auto" and torch.backends.mps.is_available() and not torch.cuda.is_available():
            resolved_device_map = {"": "mps"}
            logger.info("Apple Silicon detected: using device_map={'': 'mps'} to avoid disk offloading")

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
                device_map=resolved_device_map,
                torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map=resolved_device_map,
                torch_dtype=torch.float16,
            )

        # Load LoRA adapter if provided
        if adapter_path:
            from peft import PeftModel
            logger.info("Loading LoRA adapter from %s", adapter_path)
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            logger.info("LoRA adapter loaded")

        self.model.eval()
        self._device = next(self.model.parameters()).device
        logger.info(
            "LLMEngine ready: model=%s, adapter=%s, device=%s",
            base_model, adapter_path, self._device,
        )

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Generate a response given a list of chat messages.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": str}
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy-ish).

        Returns:
            LLMResponse with generated text and token counts.
        """
        if self._api_mode:
            return self._generate_api(messages, max_new_tokens, temperature)
        return self._generate_hf(messages, max_new_tokens, temperature)

    def _generate_api(
        self,
        messages: list[dict],
        max_new_tokens: int,
        temperature: float,
        max_retries: int = 3,
    ) -> LLMResponse:
        """Generate via OpenAI-compatible API (vLLM, Ollama, etc.)."""
        t0 = time.perf_counter()

        body = {
            "model": self._api_model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature if temperature == 0 else max(temperature, 0.01),
            "top_p": 0.9,
        }
        if self.seed is not None:
            body["seed"] = self.seed
        payload = json.dumps(body).encode()

        req = Request(
            f"{self._api_base}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        for attempt in range(max_retries):
            try:
                with urlopen(req, timeout=300) as resp:
                    result = json.loads(resp.read())
                break
            except (URLError, TimeoutError) as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "API call failed (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, max_retries, wait, e,
                )
                time.sleep(wait)

        if "error" in result:
            raise RuntimeError(f"API error: {result['error']}")

        choice = result["choices"][0]
        msg = choice["message"]
        text = msg.get("content") or ""
        # Thinking models (e.g. gemma4) put output in "reasoning" field
        if not text.strip() and msg.get("reasoning"):
            text = msg["reasoning"]
        usage = result.get("usage", {})
        latency_ms = (time.perf_counter() - t0) * 1000

        return LLMResponse(
            text=text,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
        )

    def _generate_hf(
        self,
        messages: list[dict],
        max_new_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Generate via local HuggingFace transformers."""
        import torch

        if self.seed is not None and not getattr(self, "_hf_seeded", False):
            torch.manual_seed(self.seed)
            self._hf_seeded = True

        t0 = time.perf_counter()

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=self.max_input_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_ids = outputs[0][input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        latency_ms = (time.perf_counter() - t0) * 1000

        return LLMResponse(
            text=text,
            input_tokens=input_len,
            output_tokens=len(generated_ids),
            latency_ms=latency_ms,
        )

    def __repr__(self) -> str:
        return f"<LLMEngine model={self.base_model_name!r} adapter={self.adapter_path!r}>"
