"""Evaluation and benchmarking suite for RedactIQ.

Measures:
- Precision, Recall, F1 for PII detection
- Throughput (documents/sec, tokens/sec)
- Latency (average, p50, p95, p99)
- Anomaly detection accuracy
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from redactiq.redaction.pipeline import RedactionPipeline
from redactiq.utils.config import load_config


def load_test_data(path: str) -> list[dict]:
    """Load labeled test data from JSONL."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def evaluate_detection(
    pipeline: RedactionPipeline,
    test_data: list[dict],
) -> dict[str, Any]:
    """Evaluate PII detection precision, recall, and F1.

    Uses character-level overlap to determine true/false positives.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_type: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for sample in test_data:
        if "entities" not in sample:
            continue

        text = sample["text"]
        gold_entities = sample["entities"]

        # Get predictions
        predicted = pipeline.detector.detect(text)

        # Build character-level gold set
        gold_chars: set[int] = set()
        for g in gold_entities:
            for i in range(g["start"], g["end"]):
                gold_chars.add(i)

        # Build character-level prediction set
        pred_chars: set[int] = set()
        for p in predicted:
            for i in range(p.start, p.end):
                pred_chars.add(i)

        tp = len(gold_chars & pred_chars)
        fp = len(pred_chars - gold_chars)
        fn = len(gold_chars - pred_chars)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Per-entity-type metrics
        for g in gold_entities:
            etype = g["entity_type"]
            g_set = set(range(g["start"], g["end"]))
            matched = any(
                g_set & set(range(p.start, p.end))
                for p in predicted
            )
            if matched:
                per_type[etype]["tp"] += 1
            else:
                per_type[etype]["fn"] += 1

        for p in predicted:
            p_set = set(range(p.start, p.end))
            matched = any(
                p_set & set(range(g["start"], g["end"]))
                for g in gold_entities
            )
            if not matched:
                per_type[p.entity_type.value]["fp"] += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    type_metrics = {}
    for etype, counts in per_type.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        type_metrics[etype] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r) if (p + r) > 0 else 0, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    return {
        "overall": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
        },
        "per_type": type_metrics,
    }


def benchmark_throughput(
    pipeline: RedactionPipeline,
    test_data: list[dict],
    num_iterations: int = 3,
) -> dict[str, Any]:
    """Benchmark throughput and latency."""
    texts = [d["text"] for d in test_data if "text" in d]
    total_tokens = sum(len(t.split()) for t in texts)

    latencies = []

    for iteration in range(num_iterations):
        for text in texts:
            start = time.perf_counter()
            pipeline.process(text, detect_anomalies=False)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

    latencies_ms = np.array(latencies) * 1000

    docs_per_sec = len(texts) * num_iterations / sum(latencies)
    tokens_per_sec = total_tokens * num_iterations / sum(latencies)

    return {
        "documents_processed": len(texts) * num_iterations,
        "total_tokens": total_tokens * num_iterations,
        "docs_per_second": round(docs_per_sec, 2),
        "tokens_per_second": round(tokens_per_sec, 2),
        "latency_ms": {
            "mean": round(float(np.mean(latencies_ms)), 2),
            "p50": round(float(np.percentile(latencies_ms, 50)), 2),
            "p95": round(float(np.percentile(latencies_ms, 95)), 2),
            "p99": round(float(np.percentile(latencies_ms, 99)), 2),
            "min": round(float(np.min(latencies_ms)), 2),
            "max": round(float(np.max(latencies_ms)), 2),
        },
    }


def evaluate_anomaly_detection(
    pipeline: RedactionPipeline,
    test_data: list[dict],
) -> dict[str, Any]:
    """Evaluate anomaly detection on labeled test data."""
    if not pipeline.anomaly_detector._is_trained:
        return {"error": "Anomaly detector not trained"}

    tp = fp = tn = fn = 0

    for sample in test_data:
        is_anomaly = sample.get("is_anomaly", False)
        text = sample.get("text", "")

        segments = [text]
        flags = pipeline.anomaly_detector.detect(segments)

        predicted_anomaly = len(flags) > 0

        if is_anomaly and predicted_anomaly:
            tp += 1
        elif is_anomaly and not predicted_anomaly:
            fn += 1
        elif not is_anomaly and predicted_anomaly:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total_samples": tp + fp + tn + fn,
    }


def run_full_evaluation(
    config_path: str | None = None,
    test_data_path: str = "data/test.jsonl",
    output_path: str = "evaluation_results.json",
):
    """Run all evaluations and save results."""
    config = load_config(config_path)
    # Disable LLM for evaluation speed (unless explicitly enabled)
    config.setdefault("detection", {}).setdefault("llm", {})["enabled"] = False

    pipeline = RedactionPipeline(config)
    test_data = load_test_data(test_data_path)

    logger.info(f"Running evaluation on {len(test_data)} samples")

    results = {
        "detection": evaluate_detection(pipeline, test_data),
        "throughput": benchmark_throughput(pipeline, test_data),
        "anomaly_detection": evaluate_anomaly_detection(pipeline, test_data),
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    det = results["detection"]["overall"]
    thr = results["throughput"]
    print("\n=== RedactIQ Evaluation Results ===")
    print(f"\nDetection:")
    print(f"  Precision: {det['precision']:.4f}")
    print(f"  Recall:    {det['recall']:.4f}")
    print(f"  F1:        {det['f1']:.4f}")
    print(f"\nThroughput:")
    print(f"  Docs/sec:   {thr['docs_per_second']}")
    print(f"  Tokens/sec: {thr['tokens_per_second']}")
    print(f"  Latency p50: {thr['latency_ms']['p50']}ms")
    print(f"  Latency p95: {thr['latency_ms']['p95']}ms")
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_full_evaluation()
