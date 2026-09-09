"""
beta_agent_ml.py — Agent-Beta ML variant (XGBoost / RandomForest classifier).

Wraps a traditional ML classifier in the BaseAgent interface so the
Consensus Module can treat it identically to LLM-based agents.
Supports training from preprocessed CSVs and inference from kv text.
"""

from __future__ import annotations

import json
import pickle
import re
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from .base_agent import AgentVerdict, BaseAgent

# ---------------------------------------------------------------------------
# Feature parsing from kv text
# ---------------------------------------------------------------------------
def parse_kv_text(text: str) -> Dict[str, str]:
    """Parse 'key=value ; key=value ; ...' into a dict."""
    pairs = {}
    for part in text.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            pairs[k.strip()] = v.strip()
    return pairs


# ---------------------------------------------------------------------------
# BetaAgentML
# ---------------------------------------------------------------------------
class BetaAgentML(BaseAgent):
    """Agent-Beta: ML classifier (XGBoost or RandomForest) for pattern matching.

    Training workflow:
        agent = BetaAgentML()
        agent.train(train_csv_path, text_col="text", label_col="label_name")
        agent.save("model.pkl")

    Inference workflow:
        agent = BetaAgentML.load("model.pkl")
        verdict = agent.analyze("proto=tcp ; state=FIN ; dur=low ; ...")
    """

    def __init__(
        self,
        name: str = "Beta-ML",
        classifier_type: Literal["xgboost", "random_forest"] = "random_forest",
        n_estimators: int = 200,
        max_depth: int = 12,
        random_state: int = 42,
    ):
        super().__init__(name)
        self.classifier_type = classifier_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.clf = None
        self.feature_names: List[str] = []
        self.cat_features: List[str] = []
        self.num_features: List[str] = []
        self.ordinal_encoder: Optional[OrdinalEncoder] = None
        self.label_encoder: Optional[LabelEncoder] = None
        # Mapping: category values seen during training for each cat feature
        self._cat_vocab: Dict[str, List[str]] = {}
        # Mapping: bin labels seen during training for each num feature
        self._num_vocab: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _detect_feature_types(self, df: pd.DataFrame, text_col: str) -> None:
        """Auto-detect categorical vs binned-numeric features from kv text."""
        sample_texts = df[text_col].head(100).tolist()
        all_keys = set()
        value_samples: Dict[str, List[str]] = {}
        for t in sample_texts:
            pairs = parse_kv_text(t)
            all_keys.update(pairs.keys())
            for k, v in pairs.items():
                value_samples.setdefault(k, []).append(v)

        bin_labels = {"zero", "missing", "low", "medium", "high", "very_high", "positive"}
        self.cat_features = []
        self.num_features = []

        for key in sorted(all_keys):
            vals = set(value_samples.get(key, []))
            if vals.issubset(bin_labels) or vals - bin_labels == set():
                self.num_features.append(key)
            else:
                self.cat_features.append(key)

        self.feature_names = self.cat_features + self.num_features

    def _texts_to_dataframe(self, texts: List[str]) -> pd.DataFrame:
        """Convert kv texts to a feature DataFrame."""
        rows = [parse_kv_text(t) for t in texts]
        df = pd.DataFrame(rows)
        # Ensure all training features present
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = "missing"
        return df[self.feature_names]

    def _encode_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Encode categorical + binned features to numeric array."""
        parts = []

        # Categorical features → ordinal encoding
        if self.cat_features:
            cat_df = df[self.cat_features].fillna("UNK").astype(str)
            if fit:
                self.ordinal_encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                cat_encoded = self.ordinal_encoder.fit_transform(cat_df)
                self._cat_vocab = {
                    col: list(cats)
                    for col, cats in zip(self.cat_features, self.ordinal_encoder.categories_)
                }
            else:
                cat_encoded = self.ordinal_encoder.transform(cat_df)
            parts.append(cat_encoded)

        # Binned numeric features → ordinal encoding (ordered bins)
        if self.num_features:
            bin_order = {"missing": 0, "zero": 1, "positive": 2, "low": 3, "medium": 4, "high": 5, "very_high": 6}
            num_df = df[self.num_features].fillna("missing").astype(str)
            num_encoded = num_df.apply(lambda col: col.map(lambda v: bin_order.get(v, 0))).values.astype(np.float32)
            parts.append(num_encoded)

        return np.hstack(parts) if parts else np.empty((len(df), 0))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        train_csv: str | Path,
        text_col: str = "text",
        label_col: str = "label_name",
        val_csv: Optional[str | Path] = None,
    ) -> Dict:
        """Train the ML classifier from a preprocessed CSV.

        Returns a dict with training stats (accuracy, class distribution).
        """
        train_df = pd.read_csv(train_csv)

        # Detect feature types from text
        self._detect_feature_types(train_df, text_col)

        # Build features
        feat_df = self._texts_to_dataframe(train_df[text_col].tolist())
        X_train = self._encode_features(feat_df, fit=True)

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(train_df[label_col].astype(str))

        # Build classifier
        if self.classifier_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                self.clf = XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    eval_metric="mlogloss",
                    use_label_encoder=False,
                )
            except ImportError:
                raise ImportError("xgboost not installed. Run: pip install xgboost")
        else:
            self.clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )

        t0 = time.perf_counter()
        self.clf.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        train_acc = self.clf.score(X_train, y_train)
        stats = {
            "train_samples": len(y_train),
            "num_features": X_train.shape[1],
            "cat_features": self.cat_features,
            "num_features_list": self.num_features,
            "classes": list(self.label_encoder.classes_),
            "train_accuracy": round(train_acc, 4),
            "train_time_sec": round(train_time, 2),
            "classifier": self.classifier_type,
        }

        # Optional validation
        if val_csv is not None:
            val_df = pd.read_csv(val_csv)
            val_feat = self._texts_to_dataframe(val_df[text_col].tolist())
            X_val = self._encode_features(val_feat, fit=False)
            y_val = self.label_encoder.transform(val_df[label_col].astype(str))
            stats["val_accuracy"] = round(self.clf.score(X_val, y_val), 4)
            stats["val_samples"] = len(y_val)

        return stats

    # ------------------------------------------------------------------
    # Inference (BaseAgent interface)
    # ------------------------------------------------------------------
    def analyze(self, traffic_text: str) -> AgentVerdict:
        if self.clf is None:
            return AgentVerdict(
                verdict="benign", confidence=0.0,
                reasoning="Model not trained", agent_name=self.name,
            )

        t0 = time.perf_counter()
        feat_df = self._texts_to_dataframe([traffic_text])
        X = self._encode_features(feat_df, fit=False)

        proba = self.clf.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx])
        latency = (time.perf_counter() - t0) * 1000

        is_attack = pred_label.lower() != "normal" and pred_label.lower() != "benign"

        # Build top-3 reasoning string
        top3_idx = np.argsort(proba)[::-1][:3]
        top3 = [(self.label_encoder.inverse_transform([i])[0], float(proba[i])) for i in top3_idx]
        reasoning = " | ".join(f"{label}={prob:.3f}" for label, prob in top3)

        return AgentVerdict(
            verdict="attack" if is_attack else "benign",
            attack_type=pred_label if is_attack else None,
            confidence=confidence,
            reasoning=f"ML({self.classifier_type}): {reasoning}",
            agent_name=self.name,
            latency_ms=latency,
        )

    def analyze_batch(self, traffic_texts: List[str]) -> List[AgentVerdict]:
        """Vectorized batch prediction — much faster than sequential."""
        if self.clf is None:
            return [AgentVerdict(verdict="benign", confidence=0.0,
                                 reasoning="Model not trained", agent_name=self.name)
                    for _ in traffic_texts]

        t0 = time.perf_counter()
        feat_df = self._texts_to_dataframe(traffic_texts)
        X = self._encode_features(feat_df, fit=False)
        proba_all = self.clf.predict_proba(X)
        batch_latency = (time.perf_counter() - t0) * 1000
        per_sample_ms = batch_latency / len(traffic_texts) if traffic_texts else 0

        results = []
        for i, proba in enumerate(proba_all):
            pred_idx = int(np.argmax(proba))
            pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
            confidence = float(proba[pred_idx])
            is_attack = pred_label.lower() not in ("normal", "benign")

            top3_idx = np.argsort(proba)[::-1][:3]
            top3 = [(self.label_encoder.inverse_transform([j])[0], float(proba[j])) for j in top3_idx]
            reasoning = " | ".join(f"{label}={prob:.3f}" for label, prob in top3)

            results.append(AgentVerdict(
                verdict="attack" if is_attack else "benign",
                attack_type=pred_label if is_attack else None,
                confidence=confidence,
                reasoning=f"ML({self.classifier_type}): {reasoning}",
                agent_name=self.name,
                latency_ms=per_sample_ms,
            ))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save trained model + encoders to a pickle file."""
        state = {
            "classifier_type": self.classifier_type,
            "clf": self.clf,
            "feature_names": self.feature_names,
            "cat_features": self.cat_features,
            "num_features": self.num_features,
            "ordinal_encoder": self.ordinal_encoder,
            "label_encoder": self.label_encoder,
            "cat_vocab": self._cat_vocab,
            "num_vocab": self._num_vocab,
            "name": self.name,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "BetaAgentML":
        """Load a trained model from pickle."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        agent = cls(
            name=state["name"],
            classifier_type=state["classifier_type"],
        )
        agent.clf = state["clf"]
        agent.feature_names = state["feature_names"]
        agent.cat_features = state["cat_features"]
        agent.num_features = state["num_features"]
        agent.ordinal_encoder = state["ordinal_encoder"]
        agent.label_encoder = state["label_encoder"]
        agent._cat_vocab = state["cat_vocab"]
        agent._num_vocab = state["num_vocab"]
        return agent
