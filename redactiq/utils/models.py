"""Common data models used across the redaction pipeline."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PIIEntityType(str, Enum):
    """Supported PII entity types."""
    PERSON = "PERSON"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    ADDRESS = "ADDRESS"
    IP_ADDRESS = "IP_ADDRESS"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    PASSPORT = "PASSPORT"
    DRIVERS_LICENSE = "DRIVERS_LICENSE"
    ORGANIZATION = "ORGANIZATION"
    MEDICAL_RECORD = "MEDICAL_RECORD"
    FINANCIAL_ACCOUNT = "FINANCIAL_ACCOUNT"
    # Indian PII types
    AADHAAR = "AADHAAR"
    PAN = "PAN"
    UPI_ID = "UPI_ID"
    VOTER_ID = "VOTER_ID"
    GSTIN = "GSTIN"
    IFSC = "IFSC"
    INDIAN_PASSPORT = "INDIAN_PASSPORT"
    INDIAN_PHONE = "INDIAN_PHONE"
    INDIAN_DL = "INDIAN_DL"
    UNKNOWN = "UNKNOWN"


class DetectionSource(str, Enum):
    """Where the detection came from."""
    RULE = "rule"
    LLM = "llm"
    MERGED = "merged"


class PIIEntity(BaseModel):
    """A single detected PII entity with span information."""
    entity_type: PIIEntityType
    text: str
    start: int
    end: int
    confidence: float = Field(ge=0.0, le=1.0)
    source: DetectionSource = DetectionSource.RULE


class AnomalyFlag(BaseModel):
    """An anomaly detected in a text segment."""
    segment_text: str
    start: int
    end: int
    anomaly_score: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class RedactionResult(BaseModel):
    """Complete result from the redaction pipeline."""
    original_text: str
    redacted_text: str
    entities: list[PIIEntity] = Field(default_factory=list)
    anomaly_flags: list[AnomalyFlag] = Field(default_factory=list)
    processing_time_ms: float = 0.0
    rule_detections: int = 0
    llm_detections: int = 0


class RedactionRequest(BaseModel):
    """API request for redaction."""
    text: str
    mode: str = "mask"  # mask | pseudonymize | hash
    detect_anomalies: bool = True
    entity_types: list[str] | None = None


class BatchRedactionRequest(BaseModel):
    """API request for batch redaction."""
    documents: list[str]
    mode: str = "mask"
    detect_anomalies: bool = True


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "0.1.0"
    model_loaded: bool = False
    anomaly_model_loaded: bool = False
