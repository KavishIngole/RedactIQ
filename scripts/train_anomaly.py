"""Train the anomaly detection model on baseline (PII-free) text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from redactiq.anomaly.detector import AnomalyDetector
from redactiq.utils.config import load_config


def train_anomaly_model(
    baseline_path: str = "data/anomaly_baseline.jsonl",
    output_path: str = "models/anomaly_model.pkl",
    config_path: str | None = None,
):
    """Train anomaly detector on normal text."""
    config = load_config(config_path)
    detector = AnomalyDetector(config)
    detector.load_embedder()

    # Load baseline texts
    texts = []
    with open(baseline_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            texts.append(data["text"])

    logger.info(f"Training anomaly detector on {len(texts)} baseline samples")
    detector.train(texts)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    detector.save(output_path)
    logger.info(f"Anomaly model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="data/anomaly_baseline.jsonl")
    parser.add_argument("--output", default="models/anomaly_model.pkl")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    train_anomaly_model(args.baseline, args.output, args.config)
