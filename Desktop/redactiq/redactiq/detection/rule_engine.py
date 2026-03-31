"""Rule-based PII detection using regex patterns.

This module provides fast, deterministic detection for structured PII types
like emails, phone numbers, SSNs, and credit card numbers. It handles the
'low-hanging fruit' that regex excels at, while the LLM module handles
context-dependent entities.
"""

from __future__ import annotations

import re
from typing import Any

from redactiq.utils.models import DetectionSource, PIIEntity, PIIEntityType


# ---------------------------------------------------------------------------
# Compiled regex patterns for each PII type
# ---------------------------------------------------------------------------

_PATTERNS: dict[str, tuple[PIIEntityType, re.Pattern]] = {
    "email": (
        PIIEntityType.EMAIL,
        re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
        ),
    ),
    "phone_us": (
        PIIEntityType.PHONE,
        re.compile(
            r"(?<!\d)"
            r"(?:\+?1[\s\-.]?)?"
            r"(?:\(?\d{3}\)?[\s\-.]?)"
            r"\d{3}[\s\-.]?"
            r"\d{4}"
            r"(?!\d)"
        ),
    ),
    "phone_intl": (
        PIIEntityType.PHONE,
        re.compile(
            r"\+\d{1,3}[\s\-.]?\(?\d{1,4}\)?[\s\-.]?\d{2,4}[\s\-.]?\d{2,4}(?:[\s\-.]?\d{2,4})?"
        ),
    ),
    "ssn": (
        PIIEntityType.SSN,
        # Require a context keyword before the number to avoid false positives
        re.compile(
            r"(?:SSN|social\s*security(?:\s*(?:no|number|#))?|SS#)[:\s]*"
            r"\d{3}[\-\s]?\d{2}[\-\s]?\d{4}\b",
            re.IGNORECASE,
        ),
    ),
    "ssn_standalone": (
        PIIEntityType.SSN,
        # Standalone SSN: requires dashes (XXX-XX-XXXX) for specificity
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    "credit_card": (
        PIIEntityType.CREDIT_CARD,
        re.compile(
            r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))"
            r"[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{1,4}\b"
        ),
    ),
    "ip_address": (
        PIIEntityType.IP_ADDRESS,
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
    ),
    "date_of_birth": (
        PIIEntityType.DATE_OF_BIRTH,
        re.compile(
            r"\b(?:DOB|Date of Birth|Born|Birthday)[:\s]*"
            r"(?:\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\w+ \d{1,2},?\s*\d{4})\b",
            re.IGNORECASE,
        ),
    ),
    "passport": (
        PIIEntityType.PASSPORT,
        # Require the keyword "passport" to avoid matching arbitrary IDs
        re.compile(
            r"\bpassport\s*(?:no|number|#)?[:\s]*[A-Z]{1,2}\d{6,9}\b",
            re.IGNORECASE,
        ),
    ),
    "drivers_license": (
        PIIEntityType.DRIVERS_LICENSE,
        re.compile(
            r"\b(?:DL|driver'?s?\s*(?:license|licence)\s*(?:no|number|#)?)[:\s]*"
            r"[A-Z0-9\-]{5,15}\b",
            re.IGNORECASE,
        ),
    ),
}


def _luhn_check(number_str: str) -> bool:
    """Validate a number string using the Luhn algorithm.

    Used to verify credit card numbers and reduce false positives.
    """
    digits = [int(d) for d in number_str if d.isdigit()]
    if len(digits) < 13:
        return False

    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


class RuleBasedDetector:
    """Detects PII using compiled regex patterns."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = (config or {}).get("detection", {}).get("rules", {})
        self.enabled = cfg.get("enabled", True)
        self.confidence_threshold = cfg.get("confidence_threshold", 0.85)

        # Select only the configured patterns (or all)
        active_patterns = cfg.get("patterns", list(_PATTERNS.keys()))
        self.patterns = {
            name: (etype, pat)
            for name, (etype, pat) in _PATTERNS.items()
            if name in active_patterns
        }

    def detect(self, text: str) -> list[PIIEntity]:
        """Scan text for PII using all active patterns.

        Returns a list of PIIEntity objects with span information.
        Overlapping matches are deduplicated by keeping the longest match.
        """
        if not self.enabled:
            return []

        raw_entities: list[PIIEntity] = []

        for pattern_name, (entity_type, pattern) in self.patterns.items():
            for match in pattern.finditer(text):
                confidence = self._score_confidence(pattern_name, match, text)
                if confidence < self.confidence_threshold:
                    continue

                raw_entities.append(PIIEntity(
                    entity_type=entity_type,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                    source=DetectionSource.RULE,
                ))

        return self._deduplicate(raw_entities)

    def _score_confidence(self, pattern_name: str, match: re.Match, text: str) -> float:
        """Assign a confidence score using both pattern type and match quality."""
        matched_text = match.group()

        if pattern_name == "email":
            return 0.97 if "." in matched_text.split("@")[0] else 0.95

        if pattern_name == "credit_card":
            return 0.97 if _luhn_check(matched_text) else 0.70

        if pattern_name == "ssn":
            return 0.96

        if pattern_name == "ssn_standalone":
            digits = matched_text.replace("-", "")
            if digits[:3] == "000" or digits[3:5] == "00" or digits[5:] == "0000":
                return 0.3
            return 0.90

        if pattern_name == "ip_address":
            start = max(0, match.start() - 15)
            prefix = text[start:match.start()].lower()
            if "version" in prefix or prefix.strip().endswith("v"):
                return 0.4
            return 0.93

        if pattern_name in ("phone_us", "phone_intl"):
            return 0.92 if len(matched_text) >= 12 else 0.87

        if pattern_name in ("date_of_birth", "passport", "drivers_license"):
            return 0.93

        return 0.75

    @staticmethod
    def _deduplicate(entities: list[PIIEntity]) -> list[PIIEntity]:
        """Remove overlapping detections, keeping the higher-confidence or longer span."""
        if not entities:
            return []

        sorted_ents = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))

        deduped: list[PIIEntity] = [sorted_ents[0]]
        for ent in sorted_ents[1:]:
            prev = deduped[-1]
            if ent.start < prev.end:
                if ent.confidence > prev.confidence:
                    deduped[-1] = ent
                continue
            deduped.append(ent)

        return deduped
