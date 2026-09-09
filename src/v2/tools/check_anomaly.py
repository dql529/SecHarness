"""
check_anomaly.py — ML anomaly detection tool wrapping v1 BetaAgentML.

Returns top-3 predictions + anomaly score (Suggestion 1: compressed output).
Supports batch precomputation for efficiency.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np

from ...agents.beta_agent_ml import BetaAgentML, parse_kv_text

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "name": "check_anomaly",
    "description": (
        "Run ML anomaly detection (RandomForest) on the traffic record. "
        "Returns prediction, confidence, anomaly score, and top-3 class probabilities."
    ),
    "parameters": {
        "traffic_text": {
            "type": "string",
            "description": "Raw kv-format traffic record",
        }
    },
}

TOOL_SCHEMA_DEGRADED = {
    "name": "check_anomaly",
    "description": (
        "Run ML anomaly detection on the traffic record. "
        "Returns only a binary anomaly signal (anomaly_detected) and an anomaly_score. "
        "Does NOT return specific attack type or class probabilities. "
        "Use lookup_signature or load_knowledge for detailed classification."
    ),
    "parameters": {
        "traffic_text": {
            "type": "string",
            "description": "Raw kv-format traffic record",
        }
    },
}


class CheckAnomalyTool:
    """Wraps v1 BetaAgentML as a tool for the agent loop."""

    def __init__(self, model_path: str, degraded: bool = False):
        """Load a pre-trained RF model from pickle.

        Args:
            model_path: Path to the v1 BetaAgentML pickle file.
            degraded: If True, return only anomaly_score + binary judgment
                      (no prediction class, confidence, or top-3 distribution).
                      Forces the LLM agent to use other tools for classification.
        """
        self.agent = BetaAgentML.load(model_path)
        self.degraded = degraded
        self._cache: dict[str, str] = {}  # traffic_text hash -> JSON result
        logger.info("CheckAnomalyTool loaded model from %s (degraded=%s)", model_path, degraded)

    def __call__(self, traffic_text: str) -> str:
        """Run anomaly detection on a single traffic record.

        Returns:
            JSON string with prediction, confidence, anomaly_score, top3.
        """
        # Check cache first
        cache_key = hash(traffic_text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._predict_one(traffic_text)
        result_json = json.dumps(result, ensure_ascii=False)
        self._cache[cache_key] = result_json
        return result_json

    def _predict_one(self, traffic_text: str) -> dict:
        """Run RF prediction and format as compact result."""
        feat_df = self.agent._texts_to_dataframe([traffic_text])
        X = self.agent._encode_features(feat_df, fit=False)

        proba = self.agent.clf.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = self.agent.label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx])

        is_attack = pred_label.lower() not in ("normal", "benign")
        anomaly_score = 1.0 - float(proba[self._normal_idx]) if self._normal_idx is not None else (1.0 if is_attack else 0.0)

        if self.degraded:
            # Degraded mode: only binary anomaly signal + score
            anomaly_detected = anomaly_score > 0.5
            return {
                "anomaly_detected": anomaly_detected,
                "anomaly_score": round(anomaly_score, 3),
                "note": (
                    "Anomaly score above threshold. Use lookup_signature or load_knowledge for detailed classification."
                    if anomaly_detected else
                    "Traffic appears normal based on anomaly score."
                ),
            }

        # Full mode: top-3 (Suggestion 1: compress output)
        top3_idx = np.argsort(proba)[::-1][:3]
        top3_str = " | ".join(
            f"{self.agent.label_encoder.inverse_transform([i])[0]}={proba[i]:.2f}"
            for i in top3_idx
        )

        return {
            "prediction": pred_label,
            "confidence": round(confidence, 3),
            "anomaly_score": round(anomaly_score, 3),
            "top3": top3_str,
        }

    @property
    def _normal_idx(self) -> Optional[int]:
        """Index of 'Normal'/'Benign' class in label encoder."""
        classes = list(self.agent.label_encoder.classes_)
        for label in ("Normal", "normal", "Benign", "benign"):
            if label in classes:
                return classes.index(label)
        return None

    def precompute(self, traffic_texts: list[str]) -> None:
        """Batch precompute predictions for all samples and cache results.

        Call this before running the agent loop for efficiency.
        """
        if not traffic_texts:
            return

        logger.info("Precomputing check_anomaly for %d samples...", len(traffic_texts))
        feat_df = self.agent._texts_to_dataframe(traffic_texts)
        X = self.agent._encode_features(feat_df, fit=False)
        proba_all = self.agent.clf.predict_proba(X)

        normal_idx = self._normal_idx

        for i, (text, proba) in enumerate(zip(traffic_texts, proba_all)):
            pred_idx = int(np.argmax(proba))
            pred_label = self.agent.label_encoder.inverse_transform([pred_idx])[0]
            confidence = float(proba[pred_idx])
            is_attack = pred_label.lower() not in ("normal", "benign")
            anomaly_score = 1.0 - float(proba[normal_idx]) if normal_idx is not None else (1.0 if is_attack else 0.0)

            if self.degraded:
                anomaly_detected = anomaly_score > 0.5
                result = {
                    "anomaly_detected": anomaly_detected,
                    "anomaly_score": round(anomaly_score, 3),
                    "note": (
                        "Anomaly score above threshold. Use lookup_signature or load_knowledge for detailed classification."
                        if anomaly_detected else
                        "Traffic appears normal based on anomaly score."
                    ),
                }
            else:
                top3_idx = np.argsort(proba)[::-1][:3]
                top3_str = " | ".join(
                    f"{self.agent.label_encoder.inverse_transform([j])[0]}={proba[j]:.2f}"
                    for j in top3_idx
                )
                result = {
                    "prediction": pred_label,
                    "confidence": round(confidence, 3),
                    "anomaly_score": round(anomaly_score, 3),
                    "top3": top3_str,
                }
            self._cache[hash(text)] = json.dumps(result, ensure_ascii=False)

        logger.info("Precomputed %d samples", len(traffic_texts))

    def get_cached_prediction(self, traffic_text: str) -> Optional[dict]:
        """Get cached ML prediction as dict (for Permissions checks)."""
        cache_key = hash(traffic_text)
        if cache_key in self._cache:
            return json.loads(self._cache[cache_key])
        return None
