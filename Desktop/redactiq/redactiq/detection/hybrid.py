"""Hybrid PII detection that merges rule-based and LLM detections.

The merger supports multiple strategies:
- union: Keep all detections from both sources
- intersection: Only keep detections found by both
- llm_priority: Use LLM results, fall back to rules for types LLM missed
- rule_priority: Use rule results, augment with LLM for context-dependent types
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from redactiq.detection.llm_detector import LLMDetector
from redactiq.detection.rule_engine import RuleBasedDetector
from redactiq.utils.models import DetectionSource, PIIEntity


class HybridDetector:
    """Combines rule-based and LLM-based PII detection."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.merge_strategy = cfg.get("detection", {}).get("merge_strategy", "union")
        self.rule_detector = RuleBasedDetector(config)
        self.llm_detector = LLMDetector(config)

    def load_models(self):
        """Load LLM model (rules don't need loading)."""
        self.llm_detector.load_model()

    def detect(self, text: str) -> list[PIIEntity]:
        """Run both detectors and merge results."""
        rule_entities = self.rule_detector.detect(text)
        llm_entities = self.llm_detector.detect(text)

        logger.debug(
            f"Rule detections: {len(rule_entities)}, LLM detections: {len(llm_entities)}"
        )

        merged = self._merge(rule_entities, llm_entities)

        return sorted(merged, key=lambda e: e.start)

    def _merge(
        self,
        rule_entities: list[PIIEntity],
        llm_entities: list[PIIEntity],
    ) -> list[PIIEntity]:
        """Merge two entity lists based on the configured strategy."""
        if self.merge_strategy == "union":
            return self._merge_union(rule_entities, llm_entities)
        elif self.merge_strategy == "intersection":
            return self._merge_intersection(rule_entities, llm_entities)
        elif self.merge_strategy == "llm_priority":
            return self._merge_llm_priority(rule_entities, llm_entities)
        elif self.merge_strategy == "rule_priority":
            return self._merge_rule_priority(rule_entities, llm_entities)
        else:
            logger.warning(f"Unknown merge strategy: {self.merge_strategy}, using union")
            return self._merge_union(rule_entities, llm_entities)

    def _merge_union(
        self,
        rule_entities: list[PIIEntity],
        llm_entities: list[PIIEntity],
    ) -> list[PIIEntity]:
        """Keep all detections; for overlapping spans, keep higher confidence."""
        all_entities = rule_entities + llm_entities
        return self._resolve_overlaps(all_entities)

    def _merge_intersection(
        self,
        rule_entities: list[PIIEntity],
        llm_entities: list[PIIEntity],
    ) -> list[PIIEntity]:
        """Only keep entities detected by both systems (overlapping spans)."""
        result: list[PIIEntity] = []
        for rule_ent in rule_entities:
            for llm_ent in llm_entities:
                if self._spans_overlap(rule_ent, llm_ent):
                    # Take the one with higher confidence
                    best = rule_ent if rule_ent.confidence >= llm_ent.confidence else llm_ent
                    best.source = DetectionSource.MERGED
                    result.append(best)
                    break
        return self._resolve_overlaps(result)

    def _merge_llm_priority(
        self,
        rule_entities: list[PIIEntity],
        llm_entities: list[PIIEntity],
    ) -> list[PIIEntity]:
        """Use LLM results as primary; add rule results for uncovered spans."""
        covered = set()
        for llm_ent in llm_entities:
            for i in range(llm_ent.start, llm_ent.end):
                covered.add(i)

        result = list(llm_entities)
        for rule_ent in rule_entities:
            span = set(range(rule_ent.start, rule_ent.end))
            if not span & covered:
                result.append(rule_ent)

        return self._resolve_overlaps(result)

    def _merge_rule_priority(
        self,
        rule_entities: list[PIIEntity],
        llm_entities: list[PIIEntity],
    ) -> list[PIIEntity]:
        """Use rule results as primary; add LLM results for uncovered spans."""
        covered = set()
        for rule_ent in rule_entities:
            for i in range(rule_ent.start, rule_ent.end):
                covered.add(i)

        result = list(rule_entities)
        for llm_ent in llm_entities:
            span = set(range(llm_ent.start, llm_ent.end))
            if not span & covered:
                result.append(llm_ent)

        return self._resolve_overlaps(result)

    @staticmethod
    def _spans_overlap(a: PIIEntity, b: PIIEntity) -> bool:
        """Check if two entity spans overlap."""
        return a.start < b.end and b.start < a.end

    @staticmethod
    def _resolve_overlaps(entities: list[PIIEntity]) -> list[PIIEntity]:
        """For overlapping spans, keep the one with higher confidence."""
        if not entities:
            return []

        sorted_ents = sorted(entities, key=lambda e: (e.start, -e.confidence))
        result: list[PIIEntity] = [sorted_ents[0]]

        for ent in sorted_ents[1:]:
            prev = result[-1]
            if ent.start < prev.end:
                # Overlap: keep the higher-confidence one
                if ent.confidence > prev.confidence:
                    result[-1] = ent
            else:
                result.append(ent)

        return result
