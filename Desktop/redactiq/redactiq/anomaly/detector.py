"""Anomaly detection module for catching unforeseen PII patterns.

After the hybrid redaction pipeline removes known PII, this module
analyzes the remaining text for unusual patterns that might indicate
disguised or novel forms of PII. It uses unsupervised learning on
text embeddings to flag statistical outliers.

Supported algorithms:
- Isolation Forest: Fast, scalable, good for high-dimensional data
- One-Class SVM: Strong boundary detection, works well with embeddings
- Autoencoder: Neural approach, learns complex normal patterns
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from redactiq.utils.models import AnomalyFlag


class AnomalyDetector:
    """Detects anomalous text segments that may contain hidden PII."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = (config or {}).get("anomaly", {})
        self.enabled = cfg.get("enabled", True)
        self.model_type = cfg.get("model_type", "isolation_forest")
        self.contamination = cfg.get("contamination", 0.05)
        self.threshold = cfg.get("threshold", 0.7)
        self.embedding_model_name = cfg.get("embedding_model", "all-MiniLM-L6-v2")

        self._model = None
        self._embedder = None
        self._tfidf_vectorizer = None  # Persisted across train/inference
        self._is_trained = False

    def load_embedder(self):
        """Load the sentence embedding model."""
        if not self.enabled:
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

        Args:
            normal_texts: List of text segments that represent the
                          'normal' baseline (already redacted).
        """
        if not self.enabled:
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

        Args:
            text_segments: List of text chunks to analyze.
            offsets: Character offsets of each segment in the original text.

        Returns:
            List of AnomalyFlag objects for flagged segments.
        """
        if not self.enabled or not self._is_trained or not text_segments:
            return []

        if offsets is None:
            offsets = list(range(len(text_segments)))

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

    def _score(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (0=normal, 1=highly anomalous)."""
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
        """Generate a human-readable reason for the anomaly flag."""
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
