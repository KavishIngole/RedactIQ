"""Tests for the rule-based PII detection engine."""

import pytest

from redactiq.detection.rule_engine import RuleBasedDetector, _luhn_check
from redactiq.utils.models import PIIEntityType


@pytest.fixture
def detector():
    return RuleBasedDetector()


class TestEmailDetection:
    def test_standard_email(self, detector):
        text = "Contact me at john.doe@example.com for details."
        entities = detector.detect(text)
        emails = [e for e in entities if e.entity_type == PIIEntityType.EMAIL]
        assert len(emails) == 1
        assert emails[0].text == "john.doe@example.com"

    def test_multiple_emails(self, detector):
        text = "Send to a@b.com and c@d.org please."
        entities = detector.detect(text)
        emails = [e for e in entities if e.entity_type == PIIEntityType.EMAIL]
        assert len(emails) == 2

    def test_no_false_positive_on_at_sign(self, detector):
        text = "The meeting is at 3pm."
        entities = detector.detect(text)
        emails = [e for e in entities if e.entity_type == PIIEntityType.EMAIL]
        assert len(emails) == 0


class TestPhoneDetection:
    def test_us_phone_with_dashes(self, detector):
        text = "Call 555-123-4567 now."
        entities = detector.detect(text)
        phones = [e for e in entities if e.entity_type == PIIEntityType.PHONE]
        assert len(phones) >= 1

    def test_us_phone_with_parens(self, detector):
        text = "Call (555) 123-4567 now."
        entities = detector.detect(text)
        phones = [e for e in entities if e.entity_type == PIIEntityType.PHONE]
        assert len(phones) >= 1

    def test_international_phone(self, detector):
        text = "Reach me at +44 20 7946 0958."
        entities = detector.detect(text)
        phones = [e for e in entities if e.entity_type == PIIEntityType.PHONE]
        assert len(phones) >= 1


class TestSSNDetection:
    def test_ssn_with_keyword_and_dashes(self, detector):
        text = "SSN: 123-45-6789"
        entities = detector.detect(text)
        ssns = [e for e in entities if e.entity_type == PIIEntityType.SSN]
        assert len(ssns) == 1

    def test_ssn_with_keyword_no_dashes(self, detector):
        text = "SSN: 123456789"
        entities = detector.detect(text)
        ssns = [e for e in entities if e.entity_type == PIIEntityType.SSN]
        assert len(ssns) == 1

    def test_ssn_standalone_dashed_format(self, detector):
        """Standalone SSN in XXX-XX-XXXX format should be detected."""
        text = "Number is 123-45-6789 here."
        entities = detector.detect(text)
        ssns = [e for e in entities if e.entity_type == PIIEntityType.SSN]
        assert len(ssns) >= 1

    def test_ssn_no_false_positive_on_bare_digits(self, detector):
        """Bare 9-digit numbers without dashes or context should NOT match."""
        text = "The code is 123456789 for reference."
        entities = detector.detect(text)
        ssns = [e for e in entities if e.entity_type == PIIEntityType.SSN]
        assert len(ssns) == 0

    def test_ssn_social_security_keyword(self, detector):
        text = "Social Security Number: 123-45-6789"
        entities = detector.detect(text)
        ssns = [e for e in entities if e.entity_type == PIIEntityType.SSN]
        assert len(ssns) >= 1


class TestCreditCardDetection:
    def test_visa(self, detector):
        text = "Card: 4532-0151-1283-0366"
        entities = detector.detect(text)
        cards = [e for e in entities if e.entity_type == PIIEntityType.CREDIT_CARD]
        assert len(cards) >= 1

    def test_mastercard(self, detector):
        text = "Card: 5425 2334 3010 9903"
        entities = detector.detect(text)
        cards = [e for e in entities if e.entity_type == PIIEntityType.CREDIT_CARD]
        assert len(cards) >= 1

    def test_invalid_luhn_filtered(self, detector):
        """Card numbers failing Luhn check should be filtered out."""
        text = "Card: 4532-1234-5678-9012"
        entities = detector.detect(text)
        cards = [e for e in entities if e.entity_type == PIIEntityType.CREDIT_CARD]
        assert len(cards) == 0


class TestLuhnValidation:
    def test_valid_card(self):
        assert _luhn_check("4532015112830366") is True

    def test_invalid_card(self):
        assert _luhn_check("1234567890123456") is False

    def test_too_short(self):
        assert _luhn_check("1234") is False


class TestIPAddressDetection:
    def test_valid_ip(self, detector):
        text = "Server at 192.168.1.100."
        entities = detector.detect(text)
        ips = [e for e in entities if e.entity_type == PIIEntityType.IP_ADDRESS]
        assert len(ips) == 1
        assert ips[0].text == "192.168.1.100"

    def test_version_string_low_confidence(self, detector):
        """Version-like strings should get low confidence and be filtered."""
        text = "version 1.2.3.4 is released."
        entities = detector.detect(text)
        ips = [e for e in entities if e.entity_type == PIIEntityType.IP_ADDRESS]
        # Should be filtered out due to "version" context
        assert len(ips) == 0


class TestPassportDetection:
    def test_passport_with_keyword(self, detector):
        text = "Passport number: AB1234567"
        entities = detector.detect(text)
        passports = [e for e in entities if e.entity_type == PIIEntityType.PASSPORT]
        assert len(passports) == 1

    def test_no_false_positive_without_keyword(self, detector):
        """Bare alphanumeric strings should NOT match as passports."""
        text = "Product code A1234567 is available."
        entities = detector.detect(text)
        passports = [e for e in entities if e.entity_type == PIIEntityType.PASSPORT]
        assert len(passports) == 0


class TestDeduplication:
    def test_overlapping_entities(self, detector):
        text = "Contact: 555-123-4567"
        entities = detector.detect(text)
        for i, a in enumerate(entities):
            for b in entities[i + 1 :]:
                assert a.end <= b.start or b.end <= a.start


class TestConfiguration:
    def test_disabled_detector(self):
        config = {"detection": {"rules": {"enabled": False}}}
        det = RuleBasedDetector(config)
        entities = det.detect("john@example.com 555-123-4567")
        assert len(entities) == 0

    def test_selective_patterns(self):
        config = {"detection": {"rules": {"enabled": True, "patterns": ["email"]}}}
        det = RuleBasedDetector(config)
        entities = det.detect("john@example.com 555-123-4567")
        assert all(e.entity_type == PIIEntityType.EMAIL for e in entities)
