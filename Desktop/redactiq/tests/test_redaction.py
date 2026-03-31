"""Tests for the redaction engine."""

import pytest

from redactiq.redaction.engine import RedactionEngine
from redactiq.utils.models import PIIEntity, PIIEntityType, DetectionSource


@pytest.fixture
def engine():
    return RedactionEngine()


def _make_entity(text: str, start: int, entity_type: PIIEntityType = PIIEntityType.EMAIL):
    return PIIEntity(
        entity_type=entity_type,
        text=text,
        start=start,
        end=start + len(text),
        confidence=0.95,
        source=DetectionSource.RULE,
    )


class TestMaskMode:
    def test_single_entity(self, engine):
        text = "Email: john@test.com"
        entity = _make_entity("john@test.com", 7)
        result = engine.redact(text, [entity])
        assert "john@test.com" not in result
        assert "\u2588" in result

    def test_preserves_length(self, engine):
        text = "SSN: 123-45-6789"
        entity = _make_entity("123-45-6789", 5, PIIEntityType.SSN)
        result = engine.redact(text, [entity])
        assert len(result) == len(text)

    def test_multiple_entities(self, engine):
        text = "john@a.com and jane@b.com"
        entities = [
            _make_entity("john@a.com", 0),
            _make_entity("jane@b.com", 15),
        ]
        result = engine.redact(text, entities)
        assert "john@a.com" not in result
        assert "jane@b.com" not in result

    def test_empty_entities(self, engine):
        text = "No PII here."
        result = engine.redact(text, [])
        assert result == text


class TestPseudonymizeMode:
    def test_replaces_with_fake_data(self):
        engine = RedactionEngine({"redaction": {"mode": "pseudonymize"}})
        text = "Name: John Smith"
        entity = _make_entity("John Smith", 6, PIIEntityType.PERSON)
        result = engine.redact(text, [entity])
        assert "John Smith" not in result
        assert result.startswith("Name: ")

    def test_consistent_within_document(self):
        engine = RedactionEngine({"redaction": {"mode": "pseudonymize"}})
        text = "John Smith said John Smith again."
        entities = [
            _make_entity("John Smith", 0, PIIEntityType.PERSON),
            _make_entity("John Smith", 16, PIIEntityType.PERSON),
        ]
        result = engine.redact(text, entities)
        parts = result.split(" said ")
        # Both replacements should be the same (consistent pseudonymization)
        assert parts[0] == parts[1].replace(" again.", "")


class TestHashMode:
    def test_produces_hash(self):
        engine = RedactionEngine({"redaction": {"mode": "hash"}})
        text = "Email: john@test.com"
        entity = _make_entity("john@test.com", 7)
        result = engine.redact(text, [entity])
        assert "[HASH:" in result
        assert "john@test.com" not in result


class TestModeOverride:
    def test_per_call_mode_override(self, engine):
        """Engine defaults to mask, but per-call mode='hash' should produce hash."""
        text = "Email: john@test.com"
        entity = _make_entity("john@test.com", 7)
        result = engine.redact(text, [entity], mode="hash")
        assert "[HASH:" in result

    def test_per_call_mode_does_not_mutate_engine(self, engine):
        """Using mode override should not change the engine's default mode."""
        text = "Email: john@test.com"
        entity = _make_entity("john@test.com", 7)
        engine.redact(text, [entity], mode="hash")
        # Engine default should still be mask
        assert engine.mode == "mask"
