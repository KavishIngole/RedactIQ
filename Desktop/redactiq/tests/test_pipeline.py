"""Tests for the full redaction pipeline (rules-only mode)."""

import pytest

from redactiq.redaction.pipeline import RedactionPipeline


@pytest.fixture
def pipeline():
    """Pipeline with LLM disabled for fast unit tests."""
    config = {
        "detection": {
            "rules": {"enabled": True},
            "llm": {"enabled": False},
            "merge_strategy": "union",
        },
        "redaction": {"mode": "mask", "preserve_format": True},
        "anomaly": {"enabled": False},
    }
    return RedactionPipeline(config)


class TestPipelineIntegration:
    def test_basic_redaction(self, pipeline):
        text = "Contact john@example.com or call 555-123-4567."
        result = pipeline.process(text)
        assert "john@example.com" not in result.redacted_text
        assert len(result.entities) >= 1

    def test_no_pii(self, pipeline):
        text = "This text has no personal information."
        result = pipeline.process(text)
        assert result.redacted_text == text
        assert len(result.entities) == 0

    def test_multiple_pii_types(self, pipeline):
        text = (
            "John's email is john@test.com, SSN 123-45-6789, "
            "card 4532-1234-5678-9012, IP 192.168.1.1."
        )
        result = pipeline.process(text)
        assert len(result.entities) >= 3

    def test_mode_override(self, pipeline):
        text = "Email: john@test.com"
        result = pipeline.process(text, mode="hash")
        assert "[HASH:" in result.redacted_text

    def test_processing_time(self, pipeline):
        result = pipeline.process("Test text with john@example.com")
        assert result.processing_time_ms > 0

    def test_batch_processing(self, pipeline):
        texts = [
            "Email: a@b.com",
            "Phone: 555-123-4567",
            "No PII here.",
        ]
        results = pipeline.process_batch(texts)
        assert len(results) == 3
        assert len(results[0].entities) >= 1
        assert len(results[2].entities) == 0

    def test_entity_types_filter(self, pipeline):
        """Only redact specified entity types when entity_types is provided."""
        text = "Email john@test.com and IP 192.168.1.100."
        result = pipeline.process(text, entity_types=["EMAIL"])
        # Should have at least one email entity
        assert len(result.entities) >= 1
        # All returned entities should be EMAIL
        assert all(
            e.entity_type.value == "EMAIL" for e in result.entities
        )

    def test_entity_types_filter_none_means_all(self, pipeline):
        """When entity_types is None, all types should be detected."""
        text = "Email john@test.com and IP 192.168.1.100."
        result = pipeline.process(text, entity_types=None)
        types = {e.entity_type.value for e in result.entities}
        assert len(types) >= 2
        assert "EMAIL" in types
        assert "IP_ADDRESS" in types


class TestTextSegmentation:
    def test_sentence_segmentation(self):
        text = "First sentence. Second sentence. Third sentence."
        segments = RedactionPipeline._segment_text(text)
        assert len(segments) == 3

    def test_long_text_chunking(self):
        text = "A" * 500
        segments = RedactionPipeline._segment_text(text, max_segment_len=200)
        assert len(segments) >= 2
