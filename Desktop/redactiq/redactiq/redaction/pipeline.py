"""Main redaction pipeline that orchestrates all components.

Pipeline flow:
  text -> Rule-based detection -> LLM detection -> Merge ->
  Redaction -> Anomaly Detection -> Final output + flags
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from loguru import logger

from redactiq.anomaly.detector import AnomalyDetector
from redactiq.detection.hybrid import HybridDetector
from redactiq.redaction.engine import RedactionEngine
from redactiq.utils.models import (
    AnomalyFlag,
    DetectionSource,
    PIIEntity,
    PIIEntityType,
    RedactionResult,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class RedactionPipeline:
    """End-to-end PII redaction pipeline with anomaly detection."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.detector = HybridDetector(config)
        self.redactor = RedactionEngine(config)
        self.anomaly_detector = AnomalyDetector(config)

    def load_models(self):
        """Load all ML models. Call once at startup."""
        self.detector.load_models()
        self.anomaly_detector.load_embedder()

        # Auto-load anomaly model if it exists on disk
        anomaly_path = _PROJECT_ROOT / "models" / "anomaly_model.pkl"
        if anomaly_path.exists() and not self.anomaly_detector._is_trained:
            try:
                self.anomaly_detector.load(anomaly_path)
            except Exception as e:
                logger.warning(f"Could not load anomaly model: {e}")

        logger.info("All pipeline models loaded")

    def process(
        self,
        text: str,
        mode: str | None = None,
        detect_anomalies: bool = True,
        entity_types: list[str] | None = None,
    ) -> RedactionResult:
        """Process a single document through the full pipeline.

        Args:
            text: Input text to redact.
            mode: Override redaction mode (mask/pseudonymize/hash).
            detect_anomalies: Whether to run anomaly detection.
            entity_types: Optional filter - only redact these entity types.

        Returns:
            RedactionResult with redacted text, entities, and anomaly flags.
        """
        start_time = time.perf_counter()

        # Use per-request mode without mutating shared engine state
        effective_mode = mode or self.redactor.mode

        # Step 1: Detect PII entities (hybrid: rules + LLM)
        entities = self.detector.detect(text)

        # Step 1.5: Filter by entity types if requested
        if entity_types:
            allowed = set()
            for t in entity_types:
                try:
                    allowed.add(PIIEntityType(t))
                except ValueError:
                    pass
            entities = [e for e in entities if e.entity_type in allowed]

        rule_count = sum(1 for e in entities if e.source == DetectionSource.RULE)
        llm_count = sum(1 for e in entities if e.source == DetectionSource.LLM)

        logger.info(
            f"Detected {len(entities)} PII entities "
            f"(rules={rule_count}, llm={llm_count})"
        )

        # Step 2: Apply redaction (thread-safe: pass mode as argument)
        redacted_text = self.redactor.redact(text, entities, mode=effective_mode)

        # Step 3: Anomaly detection on redacted text
        anomaly_flags: list[AnomalyFlag] = []
        if detect_anomalies and self.anomaly_detector.enabled:
            segments = self._segment_text(redacted_text)
            if segments:
                texts = [s["text"] for s in segments]
                offsets = [s["start"] for s in segments]
                anomaly_flags = self.anomaly_detector.detect(texts, offsets)

        elapsed = (time.perf_counter() - start_time) * 1000

        # Reset pseudonymization cache between documents
        self.redactor.reset_cache()

        return RedactionResult(
            original_text=text,
            redacted_text=redacted_text,
            entities=entities,
            anomaly_flags=anomaly_flags,
            processing_time_ms=round(elapsed, 2),
            rule_detections=rule_count,
            llm_detections=llm_count,
        )

    def process_batch(
        self,
        texts: list[str],
        mode: str | None = None,
        detect_anomalies: bool = True,
    ) -> list[RedactionResult]:
        """Process multiple documents with per-document error isolation."""
        results: list[RedactionResult] = []
        for text in texts:
            try:
                results.append(
                    self.process(text, mode=mode, detect_anomalies=detect_anomalies)
                )
            except Exception as e:
                logger.error(f"Error processing document ({len(text)} chars): {e}")
                results.append(RedactionResult(
                    original_text=text,
                    redacted_text=text,
                    entities=[],
                    anomaly_flags=[],
                    processing_time_ms=0.0,
                    rule_detections=0,
                    llm_detections=0,
                ))
        return results

    @staticmethod
    def _segment_text(text: str, max_segment_len: int = 200) -> list[dict]:
        """Split text into segments for anomaly detection."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        segments = []
        current_pos = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            idx = text.find(sentence, current_pos)
            if idx == -1:
                idx = current_pos

            if len(sentence) > max_segment_len:
                for i in range(0, len(sentence), max_segment_len):
                    chunk = sentence[i : i + max_segment_len]
                    segments.append({"text": chunk, "start": idx + i})
            else:
                segments.append({"text": sentence, "start": idx})

            current_pos = idx + len(sentence)

        return segments
