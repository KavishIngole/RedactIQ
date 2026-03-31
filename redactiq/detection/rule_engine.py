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

    # -----------------------------------------------------------------------
    # India-specific PII patterns
    # -----------------------------------------------------------------------

    # Aadhaar: 12 digits, optionally separated by spaces/dashes in 4-4-4 groups
    "aadhaar": (
        PIIEntityType.AADHAAR,
        re.compile(
            r"(?:aadhaar|aadhar|UID)[\s#:]*"
            r"([2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4})\b",
            re.IGNORECASE,
        ),
    ),
    "aadhaar_standalone": (
        PIIEntityType.AADHAAR,
        # Standalone 12-digit with mandatory separators (4-4-4) to avoid false positives
        re.compile(r"\b[2-9]\d{3}[\s\-]\d{4}[\s\-]\d{4}\b"),
    ),

    # PAN Card: 5 upper letters + 4 digits + 1 upper letter (e.g., ABCDE1234F)
    # 4th char encodes holder type: C,P,H,F,A,T,B,L,J,G
    "pan_card": (
        PIIEntityType.PAN,
        re.compile(
            r"\b[A-Z]{3}[CPHFATBLJG][A-Z]\d{4}[A-Z]\b"
        ),
    ),

    # Indian Passport: Letter followed by 7 digits (e.g., J8369854, K2345678)
    # Also matches with "passport" keyword for higher confidence
    "indian_passport": (
        PIIEntityType.INDIAN_PASSPORT,
        re.compile(
            r"\bpassport\s*(?:no|number|#)?[:\s]*[A-Z]\d{7}\b",
            re.IGNORECASE,
        ),
    ),

    # IFSC Code: 4 letters + 0 + 6 alphanumeric (e.g., SBIN0001234)
    "ifsc_code": (
        PIIEntityType.IFSC,
        re.compile(
            r"\b[A-Z]{4}0[A-Z0-9]{6}\b"
        ),
    ),

    # Indian Mobile: +91 or 0 prefix + 10 digits starting with 6-9
    "phone_indian": (
        PIIEntityType.INDIAN_PHONE,
        re.compile(
            r"(?<!\d)"
            r"(?:\+91[\s\-.]?|0)?"
            r"[6-9]\d{4}[\s\-.]?\d{5}"
            r"(?!\d)"
        ),
    ),

    # Voter ID / EPIC: 3 uppercase letters + 7 digits (e.g., ABC1234567)
    "voter_id": (
        PIIEntityType.VOTER_ID,
        re.compile(
            r"\b(?:voter\s*(?:id|card)|EPIC)[:\s#]*"
            r"([A-Z]{3}\d{7})\b",
            re.IGNORECASE,
        ),
    ),
    "voter_id_standalone": (
        PIIEntityType.VOTER_ID,
        re.compile(r"\b[A-Z]{3}\d{7}\b"),
    ),

    # Indian Driving License: state code (2 letters) + optional dash + 2 digits + year (4 digits) + 7 digits
    # e.g., MH02 20190001234, DL-0420150000001, KA0120171234567
    "indian_dl": (
        PIIEntityType.INDIAN_DL,
        re.compile(
            r"\b(?:DL|driving\s*licen[cs]e|license\s*no)[:\s#]*"
            r"([A-Z]{2}[\-\s]?\d{2}[\-\s]?\d{4}\d{7})\b",
            re.IGNORECASE,
        ),
    ),

    # GSTIN: 2 digits (state) + PAN (10 chars) + 1 digit + Z + 1 alphanumeric
    # e.g., 27AAPFU0939F1ZV
    "gstin": (
        PIIEntityType.GSTIN,
        re.compile(
            r"\b\d{2}[A-Z]{3}[CPHFATBLJG][A-Z]\d{4}[A-Z]\d[Z][A-Z0-9]\b"
        ),
    ),

    # UPI ID: alphanumeric + @ + bank handle (e.g., user@paytm, name@okicici)
    "upi_id": (
        PIIEntityType.UPI_ID,
        re.compile(
            r"\b[A-Za-z0-9._\-]+@(?:ok(?:icici|sbi|axis|hdfc)|paytm|ybl|upi|"
            r"apl|ibl|axl|sbi|icici|hdfcbank|kotak|indus|"
            r"federal|rbl|idbi|boi|pnb|bob|union|citi|"
            r"jupiteraxis|freecharge|gpay|phonepe)\b",
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


# Verhoeff algorithm tables for Aadhaar validation
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def _verhoeff_check(number_str: str) -> bool:
    """Validate a number string using the Verhoeff algorithm.

    Used to verify Aadhaar numbers and reduce false positives.
    """
    digits = [int(d) for d in number_str if d.isdigit()]
    if len(digits) != 12:
        return False
    c = 0
    for i, d in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][d]]
    return c == 0


class RuleBasedDetector:
    """Detects PII using compiled regex patterns."""

    # O(1) dispatch table for confidence scoring (avoids long if/elif chains)
    _SCORERS: dict[str, Any] = {
        "email": lambda m, t: 0.97 if "." in m.group().split("@")[0] else 0.95,
        "credit_card": lambda m, t: 0.97 if _luhn_check(m.group()) else 0.70,
        "ssn": lambda m, t: 0.96,
        "ssn_standalone": lambda m, t: (
            0.3 if (
                (d := m.group().replace("-", ""))[:3] == "000"
                or d[3:5] == "00"
                or d[5:] == "0000"
            ) else 0.90
        ),
        "ip_address": lambda m, t: (
            0.4 if (
                "version" in t[max(0, m.start() - 15):m.start()].lower()
                or t[max(0, m.start() - 15):m.start()].strip().lower().endswith("v")
            ) else 0.93
        ),
        "phone_us": lambda m, t: 0.92 if len(m.group()) >= 12 else 0.87,
        "phone_intl": lambda m, t: 0.92 if len(m.group()) >= 12 else 0.87,
        "date_of_birth": lambda m, t: 0.93,
        "passport": lambda m, t: 0.93,
        "drivers_license": lambda m, t: 0.93,
        # India-specific
        "aadhaar": lambda m, t: (
            0.97 if _verhoeff_check("".join(c for c in m.group() if c.isdigit())) else 0.92
        ),
        "aadhaar_standalone": lambda m, t: (
            0.95 if _verhoeff_check(m.group()) else 0.70
        ),
        "pan_card": lambda m, t: 0.96,
        "indian_passport": lambda m, t: 0.95,
        "ifsc_code": lambda m, t: 0.94,
        "phone_indian": lambda m, t: 0.93 if len(m.group()) >= 12 else 0.88,
        "voter_id": lambda m, t: 0.94,
        "voter_id_standalone": lambda m, t: (
            0.93 if any(
                kw in t[max(0, m.start() - 30):m.start()].lower()
                for kw in ("voter", "epic", "election", "id")
            ) else 0.80
        ),
        "indian_dl": lambda m, t: 0.94,
        "gstin": lambda m, t: 0.97,
        "upi_id": lambda m, t: 0.96,
    }

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
        scorer = self._SCORERS.get(pattern_name)
        if scorer is not None:
            return scorer(match, text)
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
