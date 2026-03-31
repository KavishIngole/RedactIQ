"""LLM-based PII detection using vLLM for inference.

This module handles context-dependent PII that regex cannot catch:
names, addresses, quasi-identifiers, and entities that require
understanding surrounding context. It uses a fine-tuned or prompted
LLM served via vLLM with Intel CPU optimizations.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx
from loguru import logger

from redactiq.utils.models import DetectionSource, PIIEntity, PIIEntityType


# Prompt template for PII extraction
_SYSTEM_PROMPT = """\
You are a PII detection specialist. Analyze the text and extract ALL personally \
identifiable information (PII). Return a JSON array of objects with these fields:
- "entity_type": one of PERSON, EMAIL, PHONE, SSN, CREDIT_CARD, ADDRESS, \
ORGANIZATION, DATE_OF_BIRTH, MEDICAL_RECORD, FINANCIAL_ACCOUNT, IP_ADDRESS, UNKNOWN
- "text": the exact PII text as it appears in the input
- "confidence": float between 0 and 1

Only return the JSON array, nothing else. If no PII is found, return [].
"""

_USER_TEMPLATE = "Detect all PII in the following text:\n\n{text}"


class LLMDetector:
    """Detects PII using a vLLM-served language model."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        det_cfg = cfg.get("detection", {}).get("llm", {})
        model_cfg = cfg.get("model", {})
        vllm_cfg = cfg.get("vllm", {})

        self.enabled = det_cfg.get("enabled", True)
        self.confidence_threshold = det_cfg.get("confidence_threshold", 0.7)
        self.max_tokens = det_cfg.get("max_tokens", 512)
        self.temperature = det_cfg.get("temperature", 0.1)
        self.model_name = model_cfg.get("name", "Qwen/Qwen3-8B")
        self.quantization = model_cfg.get("quantization", "bfloat16")
        self.lora_adapter_path = model_cfg.get("lora_adapter_path")
        self.device = vllm_cfg.get("device", "cpu")
        self.api_url = det_cfg.get(
            "api_url",
            "http://wiphackxlw49hx.cloudloka.com:8000",
        )
        self.use_remote_api = det_cfg.get("use_remote_api", True)

        self._llm = None
        self._sampling_params = None
        # Persistent HTTP client for connection pooling (reuse TCP connections)
        self._http_client = httpx.Client(timeout=120)

    def load_model(self):
        """Initialize the vLLM engine with optional LoRA adapter. Call once at startup."""
        if not self.enabled:
            logger.info("LLM detector is disabled, skipping model load")
            return

        # Try to apply Intel IPEX optimizations if available
        try:
            import intel_extension_for_pytorch as ipex
            logger.info(f"Intel IPEX {ipex.__version__} detected - CPU optimizations enabled")
        except ImportError:
            logger.debug("Intel IPEX not installed, using standard PyTorch")

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            logger.error(
                "vLLM is not installed. Install with: pip install vllm"
            )
            raise

        logger.info(f"Loading vLLM model: {self.model_name} on {self.device}")

        dtype_map = {"bfloat16": "bfloat16", "int8": "int8", "float16": "float16"}
        dtype = dtype_map.get(self.quantization, "float32")

        llm_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "dtype": dtype,
            "device": self.device,
            "enforce_eager": True,
            "max_model_len": 2048,
        }

        # Load LoRA adapter if configured
        if self.lora_adapter_path:
            from pathlib import Path
            if Path(self.lora_adapter_path).exists():
                llm_kwargs["enable_lora"] = True
                logger.info(f"LoRA adapter will be loaded from: {self.lora_adapter_path}")
            else:
                logger.warning(f"LoRA adapter path not found: {self.lora_adapter_path}")

        self._llm = LLM(**llm_kwargs)

        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=0.95,
        )

        logger.info("vLLM model loaded successfully")

    def detect(self, text: str) -> list[PIIEntity]:
        """Detect PII entities using the LLM.

        Routes to remote API if use_remote_api is set, otherwise uses
        local vLLM engine.
        """
        if not self.enabled:
            return []

        if self.use_remote_api:
            return self.detect_with_api(text, self.api_url)

        if self._llm is None:
            logger.warning("LLM not loaded. Call load_model() first.")
            return []

        prompt = self._build_prompt(text)
        start = time.perf_counter()

        outputs = self._llm.generate([prompt], self._sampling_params)
        raw_output = outputs[0].outputs[0].text.strip()

        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(f"LLM inference took {elapsed:.1f}ms")

        return self._parse_output(raw_output, text)

    def detect_with_api(self, text: str, api_url: str | None = None) -> list[PIIEntity]:
        """Detect PII using a remote OpenAI-compatible chat completions API."""
        url = api_url or self.api_url
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(text=text)},
        ]

        start = time.perf_counter()

        try:
            response = self._http_client.post(
                f"{url}/v1/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error(f"LLM API request failed: {e}")
            return []

        try:
            raw_output = response.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Unexpected API response format: {e}")
            return []

        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(f"LLM API call took {elapsed:.1f}ms")

        return self._parse_output(raw_output, text)

    def _build_prompt(self, text: str) -> str:
        """Construct the prompt for the LLM using Qwen3 ChatML format."""
        return (
            f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{_USER_TEMPLATE.format(text=text)}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _parse_output(self, raw_output: str, original_text: str) -> list[PIIEntity]:
        """Parse LLM JSON output into PIIEntity list."""
        # Try to extract JSON array from the output
        json_match = re.search(r"\[.*\]", raw_output, re.DOTALL)
        if not json_match:
            logger.warning(f"Could not parse LLM output as JSON: {raw_output[:200]}")
            return []

        try:
            items = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from LLM: {json_match.group()[:200]}")
            return []

        entities: list[PIIEntity] = []
        lower_text = original_text.lower()  # compute once for case-insensitive lookups
        for item in items:
            if not isinstance(item, dict):
                continue

            entity_type_str = item.get("entity_type", "UNKNOWN")
            try:
                entity_type = PIIEntityType(entity_type_str)
            except ValueError:
                entity_type = PIIEntityType.UNKNOWN

            pii_text = item.get("text", "")
            confidence = float(item.get("confidence", 0.8))

            if confidence < self.confidence_threshold:
                continue

            # Find the span in the original text
            start = original_text.find(pii_text)
            if start == -1:
                # Case-insensitive fallback (lower_text computed once above)
                start = lower_text.find(pii_text.lower())

            if start == -1:
                logger.debug(f"LLM-detected entity not found in text: {pii_text}")
                continue

            entities.append(PIIEntity(
                entity_type=entity_type,
                text=pii_text,
                start=start,
                end=start + len(pii_text),
                confidence=confidence,
                source=DetectionSource.LLM,
            ))

        return entities
