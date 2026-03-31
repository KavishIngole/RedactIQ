"""Redaction engine that masks, pseudonymizes, or hashes detected PII entities.

This module takes the list of detected PII entities and produces
sanitized text. It supports three modes:
- mask: Replace PII with asterisks (e.g., "John" -> "****")
- pseudonymize: Replace PII with realistic fake data
- hash: Replace PII with a deterministic hash
"""

from __future__ import annotations

import hashlib
from typing import Any

from faker import Faker

from redactiq.utils.models import PIIEntity, PIIEntityType


class RedactionEngine:
    """Applies redaction to text based on detected PII entities."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = (config or {}).get("redaction", {})
        self.mode = cfg.get("mode", "mask")
        self.mask_char = cfg.get("mask_char", "*")  # ASCII-safe mask character
        self.preserve_format = cfg.get("preserve_format", True)
        self.tag_format = cfg.get("tag_format", "[{entity_type}]")

        self._faker = Faker()
        Faker.seed(42)

        # Cache for pseudonymization consistency within a document
        self._pseudo_cache: dict[str, str] = {}

        # Fake data generators keyed by entity type (built once)
        self._generators: dict[PIIEntityType, Any] = {
            PIIEntityType.PERSON: self._faker.name,
            PIIEntityType.EMAIL: self._faker.email,
            PIIEntityType.PHONE: self._faker.phone_number,
            PIIEntityType.SSN: lambda: self._faker.ssn(),
            PIIEntityType.CREDIT_CARD: self._faker.credit_card_number,
            PIIEntityType.ADDRESS: self._faker.address,
            PIIEntityType.IP_ADDRESS: self._faker.ipv4,
            PIIEntityType.ORGANIZATION: self._faker.company,
            PIIEntityType.DATE_OF_BIRTH: lambda: self._faker.date_of_birth().isoformat(),
        }

    def redact(self, text: str, entities: list[PIIEntity], mode: str | None = None) -> str:
        """Apply redaction to all detected entities in the text.

        Args:
            text: Original text.
            entities: Detected PII entities.
            mode: Override redaction mode for this call (thread-safe).
        """
        if not entities:
            return text

        effective_mode = mode or self.mode

        # Single-pass forward build: collect text fragments and replacements,
        # then join once — O(T) instead of O(N*T) slice-and-concat.
        sorted_entities = sorted(entities, key=lambda e: e.start)
        parts: list[str] = []
        prev_end = 0
        for entity in sorted_entities:
            parts.append(text[prev_end:entity.start])
            parts.append(self._get_replacement(entity, effective_mode))
            prev_end = entity.end
        parts.append(text[prev_end:])
        return "".join(parts)

    def _get_replacement(self, entity: PIIEntity, mode: str | None = None) -> str:
        """Generate replacement text based on redaction mode."""
        effective_mode = mode or self.mode
        if effective_mode == "mask":
            return self._mask(entity)
        elif effective_mode == "pseudonymize":
            return self._pseudonymize(entity)
        elif effective_mode == "hash":
            return self._hash(entity)
        else:
            return self._mask(entity)

    def _mask(self, entity: PIIEntity) -> str:
        """Replace PII text with mask characters or a typed tag."""
        if self.preserve_format:
            # Keep the same length with mask characters
            return self.mask_char * len(entity.text)
        else:
            return self.tag_format.format(entity_type=entity.entity_type.value)

    def _pseudonymize(self, entity: PIIEntity) -> str:
        """Replace PII with realistic fake data of the same type."""
        cache_key = entity.text.lower()
        if cache_key in self._pseudo_cache:
            return self._pseudo_cache[cache_key]

        gen = self._generators.get(entity.entity_type)
        replacement = gen() if gen else self._faker.word()
        self._pseudo_cache[cache_key] = replacement
        return replacement

    def _hash(self, entity: PIIEntity) -> str:
        """Replace PII with a deterministic SHA-256 hash prefix."""
        hash_val = hashlib.sha256(entity.text.encode()).hexdigest()[:12]
        return f"[HASH:{hash_val}]"

    def reset_cache(self):
        """Clear the pseudonymization cache between documents."""
        self._pseudo_cache.clear()
