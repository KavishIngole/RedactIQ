"""Synthetic PII data generator for training and evaluation.

Uses Faker to generate realistic documents with embedded PII.
Produces labeled datasets in JSONL format suitable for training
the LLM detector and evaluating the full pipeline.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from faker import Faker

from redactiq.utils.models import PIIEntityType


fake = Faker()
Faker.seed(42)
random.seed(42)

# Document templates with {PII_TYPE} placeholders
_TEMPLATES = [
    # Bank statement
    (
        "Account holder: {PERSON}\n"
        "Account Number: {FINANCIAL_ACCOUNT}\n"
        "Address: {ADDRESS}\n"
        "Phone: {PHONE}\n"
        "Email: {EMAIL}\n"
        "Date of Birth: {DATE_OF_BIRTH}\n"
        "SSN: {SSN}\n"
        "Transaction on {DATE}: ${{amount}} at {{merchant}}"
    ),
    # Medical report
    (
        "Patient Name: {PERSON}\n"
        "MRN: {MEDICAL_RECORD}\n"
        "DOB: {DATE_OF_BIRTH}\n"
        "Address: {ADDRESS}\n"
        "Contact: {PHONE}\n"
        "Email: {EMAIL}\n"
        "Insurance ID: {FINANCIAL_ACCOUNT}\n"
        "Diagnosis: {{diagnosis}}\n"
        "Physician: {PERSON_2}"
    ),
    # Employee record
    (
        "Employee: {PERSON}\n"
        "Employee ID: {{emp_id}}\n"
        "SSN: {SSN}\n"
        "Company: {ORGANIZATION}\n"
        "Office: {ADDRESS}\n"
        "Work Email: {EMAIL}\n"
        "Phone: {PHONE}\n"
        "Hire Date: {DATE}\n"
        "Manager: {PERSON_2}"
    ),
    # Customer support ticket
    (
        "From: {PERSON} <{EMAIL}>\n"
        "Phone: {PHONE}\n"
        "Address: {ADDRESS}\n"
        "Subject: Issue with order #{{order_id}}\n\n"
        "Hi, my name is {PERSON} and my credit card ({CREDIT_CARD}) "
        "was charged incorrectly. My account number is {FINANCIAL_ACCOUNT}. "
        "Please reach me at {PHONE} or {EMAIL}. My SSN for verification: {SSN}."
    ),
    # Simple text with scattered PII
    (
        "Please contact {PERSON} at {EMAIL} or call {PHONE}. "
        "The meeting will be held at {ADDRESS}. "
        "Attendees from {ORGANIZATION} include {PERSON_2}. "
        "Connect via IP {IP_ADDRESS}."
    ),
]

# Anomalous patterns for anomaly detection training
_ANOMALY_TEMPLATES = [
    # Disguised phone number with letters
    "Call me at +1-8O0-555-1234 (that's oh not zero)",
    # Obfuscated email
    "Send to john DOT doe AT company DOT com",
    # Split SSN
    "My number is 123 - then 45 - then 6789",
    # Encoded credit card
    "Card: four-five-one-two 3456 7890 1234",
    # PII in URL parameters
    "Visit https://example.com/profile?ssn=123456789&name=JohnDoe&email=john@test.com",
    # PII in JSON-like strings
    '{"name": "Jane Smith", "ssn": "987-65-4321", "phone": "555-0123"}',
    # Reversed PII
    "moc.liamg@eod.nhoj - that's my email backwards",
    # Base64-like encoded name
    "User ID: Sm9obiBEb2U= (John Doe encoded)",
]


def _generate_pii_values() -> dict[str, tuple[str, str]]:
    """Generate a set of fake PII values with their types."""
    person1 = fake.name()
    person2 = fake.name()
    return {
        "{PERSON}": (person1, "PERSON"),
        "{PERSON_2}": (person2, "PERSON"),
        "{EMAIL}": (fake.email(), "EMAIL"),
        "{PHONE}": (fake.phone_number(), "PHONE"),
        "{SSN}": (fake.ssn(), "SSN"),
        "{CREDIT_CARD}": (fake.credit_card_number(), "CREDIT_CARD"),
        "{ADDRESS}": (fake.address().replace("\n", ", "), "ADDRESS"),
        "{IP_ADDRESS}": (fake.ipv4(), "IP_ADDRESS"),
        "{DATE_OF_BIRTH}": (f"DOB: {fake.date_of_birth()}", "DATE_OF_BIRTH"),
        "{DATE}": (str(fake.date_this_decade()), "DATE_OF_BIRTH"),
        "{ORGANIZATION}": (fake.company(), "ORGANIZATION"),
        "{FINANCIAL_ACCOUNT}": (fake.bban(), "FINANCIAL_ACCOUNT"),
        "{MEDICAL_RECORD}": (f"MRN-{fake.random_number(digits=8)}", "MEDICAL_RECORD"),
    }


def _fill_non_pii(text: str) -> str:
    """Fill non-PII template placeholders."""
    replacements = {
        "{{amount}}": f"{random.uniform(10, 5000):.2f}",
        "{{merchant}}": fake.company(),
        "{{diagnosis}}": random.choice([
            "Hypertension", "Type 2 Diabetes", "Migraine", "Asthma", "Fracture"
        ]),
        "{{emp_id}}": f"EMP-{fake.random_number(digits=6)}",
        "{{order_id}}": str(fake.random_number(digits=8)),
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    return text


def generate_document() -> dict[str, Any]:
    """Generate a synthetic document with PII labels.

    Returns:
        dict with keys:
        - text: The full document text
        - entities: List of {entity_type, text, start, end}
    """
    template = random.choice(_TEMPLATES)

    # Fill non-PII placeholders first, then replace PII
    text = _fill_non_pii(template)
    pii_map = _generate_pii_values()
    entities = []

    # Replace PII placeholders and record positions
    for placeholder, (value, etype) in pii_map.items():
        while placeholder in text:
            start = text.index(placeholder)
            text = text.replace(placeholder, value, 1)
            entities.append({
                "entity_type": etype,
                "text": value,
                "start": start,
                "end": start + len(value),
            })

    return {"text": text, "entities": entities}


def generate_anomaly_sample() -> dict[str, Any]:
    """Generate a text with anomalous PII patterns."""
    template = random.choice(_ANOMALY_TEMPLATES)
    # Add some normal text around the anomaly
    prefix = fake.sentence()
    suffix = fake.sentence()
    text = f"{prefix} {template} {suffix}"
    return {
        "text": text,
        "is_anomaly": True,
        "anomaly_segment": template,
    }


def generate_dataset(
    n_normal: int = 1000,
    n_anomalous: int = 50,
    output_dir: str = "data",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """Generate a full labeled dataset and save as JSONL files.

    Args:
        n_normal: Number of normal documents to generate.
        n_anomalous: Number of anomalous samples to generate.
        output_dir: Directory to write the dataset files.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Generate normal documents
    normal_docs = [generate_document() for _ in range(n_normal)]

    # Generate anomalous documents
    anomaly_docs = [generate_anomaly_sample() for _ in range(n_anomalous)]

    # Split normal docs
    n_train = int(n_normal * train_ratio)
    n_val = int(n_normal * val_ratio)

    train_data = normal_docs[:n_train]
    val_data = normal_docs[n_train:n_train + n_val]
    test_data = normal_docs[n_train + n_val:]

    # Add some anomalies to validation and test
    a_split = len(anomaly_docs) // 2
    val_data.extend(anomaly_docs[:a_split])
    test_data.extend(anomaly_docs[a_split:])

    # Write JSONL files
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = out / f"{name}.jsonl"
        with open(path, "w") as f:
            for doc in data:
                f.write(json.dumps(doc) + "\n")
        print(f"Wrote {len(data)} samples to {path}")

    # Create anomaly baseline (normal text only, for training anomaly detector)
    baseline_path = out / "anomaly_baseline.jsonl"
    with open(baseline_path, "w") as f:
        for doc in train_data:
            f.write(json.dumps({"text": doc["text"]}) + "\n")
    print(f"Wrote {len(train_data)} baseline samples to {baseline_path}")


if __name__ == "__main__":
    generate_dataset()
