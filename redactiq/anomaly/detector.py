"""Anomaly detection module for catching unforeseen PII patterns.

After the hybrid redaction pipeline removes known PII, this module
analyzes the remaining text for unusual patterns that might indicate
disguised or novel forms of PII.

Supported modes:
- llm: Uses Qwen3-8B to analyze text for hidden/obfuscated PII (no training needed)
- isolation_forest: Fast unsupervised ML on text embeddings (requires training)
- one_class_svm: Boundary-based on embeddings (requires training)
- autoencoder: Neural approach on embeddings (requires training)
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from loguru import logger

from redactiq.utils.models import AnomalyFlag


# ---------------------------------------------------------------------------
# LLM prompt for anomaly detection
# ---------------------------------------------------------------------------

_ANOMALY_SYSTEM_PROMPT = """\
You are a data privacy specialist. Your job is to analyze text that has \
ALREADY been through PII redaction and check if any personally identifiable \
information (PII) was MISSED or is DISGUISED/OBFUSCATED.

Look for:
1. Obfuscated PII: numbers/codes that look like IDs, phone numbers with altered \
formatting, encoded data (base64, hex), masked emails, etc.
2. Indirect identifiers: combinations of age + location + job that could re-identify someone
3. Hidden patterns: unusual character sequences, encoded strings, suspicious numeric patterns
4. Incomplete redaction: partial PII left behind (e.g., last 4 digits of SSN, partial names)

For each suspicious segment, return a JSON array of objects with:
- "text": the suspicious text fragment
- "score": anomaly score between 0.0 and 1.0 (1.0 = very likely hidden PII)
- "reason": brief explanation of why it looks suspicious

If nothing is suspicious, return []. Only return the JSON array, nothing else.\
"""

_ANOMALY_USER_TEMPLATE = "Analyze this already-redacted text for any missed or disguised PII:\n\n{text}"


class AnomalyDetector:
    """Detects anomalous text segments that may contain hidden PII."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = (config or {}).get("anomaly", {})
        self.enabled = cfg.get("enabled", True)
        self.model_type = cfg.get("model_type", "llm")
        self.contamination = cfg.get("contamination", 0.05)
        self.threshold = cfg.get("threshold", 0.7)
        self.embedding_model_name = cfg.get("embedding_model", "all-MiniLM-L6-v2")

        # LLM anomaly detection settings (reuse detection.llm config)
        det_cfg = (config or {}).get("detection", {}).get("llm", {})
        self.api_url = cfg.get(
            "api_url",
            det_cfg.get("api_url", "http://wiphackxlw49hx.cloudloka.com:8000"),
        )
        self.llm_model_name = (config or {}).get("model", {}).get("name", "Qwen/Qwen3-8B")
        self.max_tokens = cfg.get("max_tokens", 512)
        self.temperature = cfg.get("temperature", 0.1)

        self._model = None
        self._embedder = None
        self._tfidf_vectorizer = None  # Persisted across train/inference
        self._is_trained = False
        self._http_client = httpx.Client(timeout=120)

    @property
    def ready(self) -> bool:
        """Whether the detector is ready to run."""
        if self.model_type == "llm":
            return self.enabled  # LLM mode needs no training
        return self.enabled and self._is_trained

    def load_embedder(self):
        """Load the sentence embedding model (only needed for ML modes)."""
        if not self.enabled or self.model_type == "llm":
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Using fallback TF-IDF embeddings."
            )
            self._embedder = None

    def _embed(self, texts: list[str], fit: bool = False) -> np.ndarray:
        """Convert texts to embedding vectors.

        Args:
            texts: Input text segments.
            fit: If True, fit the TF-IDF vectorizer (training time only).
        """
        if self._embedder is not None:
            return self._embedder.encode(texts, show_progress_bar=False)

        # Fallback: TF-IDF vectorization with persistent vectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer

        if fit or self._tfidf_vectorizer is None:
            self._tfidf_vectorizer = TfidfVectorizer(max_features=512)
            return self._tfidf_vectorizer.fit_transform(texts).toarray()
        else:
            return self._tfidf_vectorizer.transform(texts).toarray()

    def train(self, normal_texts: list[str]):
        """Train the anomaly detection model on normal (PII-free) text.

        Not needed for LLM mode — call only for ML-based modes.
        """
        if not self.enabled:
            return

        if self.model_type == "llm":
            logger.info("LLM anomaly mode requires no training — ready to use")
            self._is_trained = True
            return

        logger.info(f"Training anomaly detector ({self.model_type}) on {len(normal_texts)} samples")

        embeddings = self._embed(normal_texts, fit=True)

        if self.model_type == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            self._model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=200,
            )
            self._model.fit(embeddings)

        elif self.model_type == "one_class_svm":
            from sklearn.svm import OneClassSVM
            self._model = OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=self.contamination,
            )
            self._model.fit(embeddings)

        elif self.model_type == "autoencoder":
            self._model = self._train_autoencoder(embeddings)

        else:
            raise ValueError(f"Unknown anomaly model type: {self.model_type}")

        self._is_trained = True
        logger.info("Anomaly detector trained successfully")

    def _train_autoencoder(self, embeddings: np.ndarray):
        """Train a simple autoencoder for anomaly detection."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        input_dim = embeddings.shape[1]
        encoding_dim = max(32, input_dim // 4)

        class Autoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, encoding_dim * 2),
                    nn.ReLU(),
                    nn.Linear(encoding_dim * 2, encoding_dim),
                    nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, encoding_dim * 2),
                    nn.ReLU(),
                    nn.Linear(encoding_dim * 2, input_dim),
                )

            def forward(self, x):
                return self.decoder(self.encoder(x))

        model = Autoencoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        tensor_data = torch.FloatTensor(embeddings)
        dataset = TensorDataset(tensor_data, tensor_data)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(50):
            total_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.debug(f"Autoencoder epoch {epoch+1}/50 loss: {total_loss:.4f}")
        model.eval()
        # Compute threshold from training data reconstruction error
        with torch.no_grad():
            recon = model(tensor_data)
            errors = torch.mean((tensor_data - recon) ** 2, dim=1).numpy()

        return {
            "model": model,
            "threshold": np.percentile(errors, 95),
            "type": "autoencoder",
        }

    def detect(self, text_segments: list[str], offsets: list[int] | None = None) -> list[AnomalyFlag]:
        """Score text segments and flag anomalies.

        Routes to LLM-based or ML-based detection depending on model_type.
        """
        if not self.enabled or not text_segments:
            return []

        if offsets is None:
            offsets = list(range(len(text_segments)))

        if self.model_type == "llm":
            return self._detect_with_llm(text_segments, offsets)

        # ML-based modes require training
        if not self._is_trained:
            return []

        embeddings = self._embed(text_segments, fit=False)
        scores = self._score(embeddings)

        flags: list[AnomalyFlag] = []
        for i, (segment, score) in enumerate(zip(text_segments, scores)):
            if score >= self.threshold:
                offset = offsets[i] if i < len(offsets) else 0
                flags.append(AnomalyFlag(
                    segment_text=segment,
                    start=offset,
                    end=offset + len(segment),
                    anomaly_score=float(score),
                    reason=self._explain_anomaly(score),
                ))

        logger.debug(f"Anomaly detection: {len(flags)}/{len(text_segments)} segments flagged")
        return flags

    def _detect_with_llm(
        self, text_segments: list[str], offsets: list[int]
    ) -> list[AnomalyFlag]:
        """Use Qwen3-8B to analyze text for hidden/disguised PII."""
        # Combine segments into a single block for efficiency
        combined_text = "\n".join(text_segments)

        # Cap input to avoid excessive token usage
        if len(combined_text) > 4000:
            combined_text = combined_text[:4000]

        messages = [
            {"role": "system", "content": _ANOMALY_SYSTEM_PROMPT},
            {"role": "user", "content": _ANOMALY_USER_TEMPLATE.format(text=combined_text)},
        ]

        try:
            response = self._http_client.post(
                f"{self.api_url}/v1/chat/completions",
                json={
                    "model": self.llm_model_name,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error(f"LLM anomaly API request failed: {e}")
            return []

        try:
            raw_output = response.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Unexpected LLM anomaly response format: {e}")
            return []

        return self._parse_llm_anomalies(raw_output, combined_text, offsets)

    def _parse_llm_anomalies(
        self, raw_output: str, source_text: str, offsets: list[int]
    ) -> list[AnomalyFlag]:
        """Parse JSON output from the LLM anomaly detector."""
        json_match = re.search(r"\[.*\]", raw_output, re.DOTALL)
        if not json_match:
            # Empty array or no JSON found — no anomalies
            if "[]" in raw_output:
                return []
            logger.debug(f"Could not parse LLM anomaly output: {raw_output[:200]}")
            return []

        try:
            items = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from LLM anomaly detector: {raw_output[:200]}")
            return []

        flags: list[AnomalyFlag] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            text_fragment = item.get("text", "")
            score = float(item.get("score", 0.5))
            reason = item.get("reason", "Flagged by LLM anomaly analysis")

            if score < self.threshold:
                continue

            # Find position in source text
            start = source_text.find(text_fragment)
            if start == -1:
                start = 0
            end = start + len(text_fragment) if text_fragment else start

            flags.append(AnomalyFlag(
                segment_text=text_fragment,
                start=start,
                end=end,
                anomaly_score=min(score, 1.0),
                reason=reason,
            ))

        logger.debug(f"LLM anomaly detection: {len(flags)} segments flagged")
        return flags

    def _score(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (0=normal, 1=highly anomalous). ML modes only."""
        if self.model_type in ("isolation_forest", "one_class_svm"):
            # decision_function: negative = anomalous, positive = normal
            raw_scores = self._model.decision_function(embeddings)
            # Normalize to [0, 1] where 1 = most anomalous
            min_s, max_s = raw_scores.min(), raw_scores.max()
            if max_s - min_s == 0:
                return np.zeros(len(embeddings))
            normalized = 1 - (raw_scores - min_s) / (max_s - min_s)
            return normalized

        elif self.model_type == "autoencoder":
            import torch
            model = self._model["model"]
            threshold = self._model["threshold"]
            tensor_data = torch.FloatTensor(embeddings)
            with torch.no_grad():
                recon = model(tensor_data)
                errors = torch.mean((tensor_data - recon) ** 2, dim=1).numpy()
            # Normalize: higher error = more anomalous
            return np.clip(errors / (threshold * 2), 0, 1)

        # Fallback: should be unreachable since train() validates model_type
        return np.zeros(len(embeddings))

    def _explain_anomaly(self, score: float) -> str:
        """Generate a human-readable reason for the anomaly flag (ML modes)."""
        if score >= 0.9:
            return "Highly unusual pattern detected - possible disguised PII"
        elif score >= 0.8:
            return "Significant deviation from normal text - review recommended"
        elif score >= 0.7:
            return "Moderate anomaly - may contain obfuscated identifiers"
        return "Low-level anomaly"

    def save(self, path: str | Path):
        """Persist the trained anomaly model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "model_type": self.model_type,
                "threshold": self.threshold,
                "is_trained": self._is_trained,
                "tfidf_vectorizer": self._tfidf_vectorizer,
            }, f)
        logger.info(f"Anomaly model saved to {path}")

    def load(self, path: str | Path):
        """Load a previously trained anomaly model."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Anomaly model file not found: {path}")
            return

        with open(path, "rb") as f:
            data = pickle.load(f)

        expected_keys = {"model", "model_type", "is_trained"}
        if not expected_keys.issubset(data.keys()):
            logger.error(f"Invalid anomaly model file (missing keys): {path}")
            return

        self._model = data["model"]
        self.model_type = data["model_type"]
        self.threshold = data.get("threshold", self.threshold)
        self._is_trained = data["is_trained"]
        self._tfidf_vectorizer = data.get("tfidf_vectorizer")
        logger.info(f"Anomaly model loaded from {path}")
