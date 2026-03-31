"""Tests for the rule-based PII detection engine."""

import pytest

from redactiq.detection.rule_engine import RuleBasedDetector, _luhn_check, _verhoeff_check
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


# -----------------------------------------------------------------------
# India-specific PII detection tests
# -----------------------------------------------------------------------


class TestVerhoeffValidation:
    def test_valid_aadhaar(self):
        assert _verhoeff_check("987654321012") is True

    def test_invalid_aadhaar(self):
        assert _verhoeff_check("123456789012") is False

    def test_wrong_length(self):
        assert _verhoeff_check("12345") is False


class TestAadhaarDetection:
    def test_aadhaar_with_keyword(self, detector):
        text = "Aadhaar: 9876 5432 1012"
        entities = detector.detect(text)
        aadhaar = [e for e in entities if e.entity_type == PIIEntityType.AADHAAR]
        assert len(aadhaar) >= 1

    def test_aadhaar_with_keyword_dashes(self, detector):
        text = "UID: 2994-5612-3038"
        entities = detector.detect(text)
        aadhaar = [e for e in entities if e.entity_type == PIIEntityType.AADHAAR]
        assert len(aadhaar) >= 1

    def test_aadhaar_standalone_with_separators(self, detector):
        text = "Number is 9876 5432 1012 for reference."
        entities = detector.detect(text)
        aadhaar = [e for e in entities if e.entity_type == PIIEntityType.AADHAAR]
        assert len(aadhaar) >= 1

    def test_aadhaar_no_false_positive_bare_digits(self, detector):
        """Bare 12-digit number without separators and no keyword should NOT match."""
        text = "Code 123456789012 is invalid."
        entities = detector.detect(text)
        aadhaar = [e for e in entities if e.entity_type == PIIEntityType.AADHAAR]
        assert len(aadhaar) == 0

    def test_aadhaar_starts_with_0_or_1_rejected(self, detector):
        """Aadhaar cannot start with 0 or 1."""
        text = "Aadhaar: 0123 4567 8901"
        entities = detector.detect(text)
        aadhaar = [e for e in entities if e.entity_type == PIIEntityType.AADHAAR]
        assert len(aadhaar) == 0


class TestPANDetection:
    def test_valid_pan(self, detector):
        text = "PAN is ABCPD1234E for tax filing."
        entities = detector.detect(text)
        pans = [e for e in entities if e.entity_type == PIIEntityType.PAN]
        assert len(pans) >= 1
        assert pans[0].text == "ABCPD1234E"

    def test_company_pan(self, detector):
        text = "Company PAN: AABCU1234F"
        entities = detector.detect(text)
        pans = [e for e in entities if e.entity_type == PIIEntityType.PAN]
        assert len(pans) >= 1

    def test_invalid_pan_wrong_4th_char(self, detector):
        """4th character must be a valid holder type letter."""
        text = "ABCXD1234E is not a valid PAN."
        entities = detector.detect(text)
        pans = [e for e in entities if e.entity_type == PIIEntityType.PAN]
        assert len(pans) == 0


class TestIndianPassportDetection:
    def test_indian_passport_with_keyword(self, detector):
        text = "Passport No: J8369854"
        entities = detector.detect(text)
        passports = [e for e in entities if e.entity_type == PIIEntityType.INDIAN_PASSPORT]
        assert len(passports) >= 1

    def test_no_match_without_keyword(self, detector):
        """Bare letter+7digits without passport keyword should not match."""
        text = "Reference K2345678 is ready."
        entities = detector.detect(text)
        ind_passports = [e for e in entities if e.entity_type == PIIEntityType.INDIAN_PASSPORT]
        assert len(ind_passports) == 0


class TestIFSCDetection:
    def test_valid_ifsc(self, detector):
        text = "IFSC code is SBIN0001234 for SBI."
        entities = detector.detect(text)
        ifsc = [e for e in entities if e.entity_type == PIIEntityType.IFSC]
        assert len(ifsc) >= 1
        assert ifsc[0].text == "SBIN0001234"

    def test_icici_ifsc(self, detector):
        text = "Transfer via ICIC0002345."
        entities = detector.detect(text)
        ifsc = [e for e in entities if e.entity_type == PIIEntityType.IFSC]
        assert len(ifsc) >= 1

    def test_invalid_ifsc_no_zero(self, detector):
        """5th character must be 0 in IFSC."""
        text = "Code SBIN1001234 is wrong."
        entities = detector.detect(text)
        ifsc = [e for e in entities if e.entity_type == PIIEntityType.IFSC]
        assert len(ifsc) == 0


class TestIndianPhoneDetection:
    def test_with_plus91(self, detector):
        text = "Call +91 98765 43210 for info."
        entities = detector.detect(text)
        phones = [e for e in entities if e.entity_type == PIIEntityType.INDIAN_PHONE]
        assert len(phones) >= 1

    def test_with_zero_prefix(self, detector):
        text = "Mobile: 09876543210"
        entities = detector.detect(text)
        phones = [e for e in entities if e.entity_type == PIIEntityType.INDIAN_PHONE]
        assert len(phones) >= 1

    def test_ten_digit_starting_with_6(self, detector):
        text = "Number 63456 78901 is active."
        entities = detector.detect(text)
        phones = [e for e in entities if e.entity_type == PIIEntityType.INDIAN_PHONE]
        assert len(phones) >= 1

    def test_no_match_starting_with_5(self, detector):
        """Indian mobiles start with 6-9 only."""
        text = "Call 51234 56789 now."
        entities = detector.detect(text)
        phones = [e for e in entities if e.entity_type == PIIEntityType.INDIAN_PHONE]
        assert len(phones) == 0


class TestVoterIDDetection:
    def test_voter_id_with_keyword(self, detector):
        text = "Voter ID: ABC1234567"
        entities = detector.detect(text)
        voters = [e for e in entities if e.entity_type == PIIEntityType.VOTER_ID]
        assert len(voters) >= 1

    def test_epic_keyword(self, detector):
        text = "EPIC: XYZ9876543"
        entities = detector.detect(text)
        voters = [e for e in entities if e.entity_type == PIIEntityType.VOTER_ID]
        assert len(voters) >= 1


class TestIndianDLDetection:
    def test_dl_with_keyword(self, detector):
        text = "DL: MH02 20190001234"
        entities = detector.detect(text)
        dls = [e for e in entities if e.entity_type == PIIEntityType.INDIAN_DL]
        assert len(dls) >= 1

    def test_license_keyword(self, detector):
        text = "Driving licence: KA01 20171234567"
        entities = detector.detect(text)
        dls = [e for e in entities if e.entity_type == PIIEntityType.INDIAN_DL]
        assert len(dls) >= 1


class TestGSTINDetection:
    def test_valid_gstin(self, detector):
        text = "GSTIN: 27AAPFU0939F1ZV"
        entities = detector.detect(text)
        gst = [e for e in entities if e.entity_type == PIIEntityType.GSTIN]
        assert len(gst) >= 1
        assert gst[0].text == "27AAPFU0939F1ZV"

    def test_another_gstin(self, detector):
        text = "GST is 07AAGCR4375J1ZU registered."
        entities = detector.detect(text)
        gst = [e for e in entities if e.entity_type == PIIEntityType.GSTIN]
        assert len(gst) >= 1

    def test_invalid_gstin_no_z(self, detector):
        """13th character must be Z in GSTIN."""
        text = "Code 27AAPFU0939F1AV is invalid."
        entities = detector.detect(text)
        gst = [e for e in entities if e.entity_type == PIIEntityType.GSTIN]
        assert len(gst) == 0


class TestUPIDetection:
    def test_paytm_upi(self, detector):
        text = "Pay to user123@paytm for order."
        entities = detector.detect(text)
        upi = [e for e in entities if e.entity_type == PIIEntityType.UPI_ID]
        assert len(upi) >= 1

    def test_okicici_upi(self, detector):
        text = "UPI: rajesh.kumar@okicici"
        entities = detector.detect(text)
        upi = [e for e in entities if e.entity_type == PIIEntityType.UPI_ID]
        assert len(upi) >= 1

    def test_ybl_upi(self, detector):
        text = "Send to 9876543210@ybl please."
        entities = detector.detect(text)
        upi = [e for e in entities if e.entity_type == PIIEntityType.UPI_ID]
        assert len(upi) >= 1

    def test_regular_email_not_upi(self, detector):
        """Normal emails should not match as UPI IDs."""
        text = "Contact user@gmail.com for help."
        entities = detector.detect(text)
        upi = [e for e in entities if e.entity_type == PIIEntityType.UPI_ID]
        assert len(upi) == 0
