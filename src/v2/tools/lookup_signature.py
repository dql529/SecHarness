"""
lookup_signature.py — Attack signature matching engine.

Matches traffic features against a YAML-based signature database.
Input: traffic_text (kv format); the tool parses it internally.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from ...agents.beta_agent_ml import parse_kv_text

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "name": "lookup_signature",
    "description": (
        "Match traffic against known attack signature database. "
        "Returns top matches with similarity scores."
    ),
    "parameters": {
        "traffic_text": {
            "type": "string",
            "description": "Raw kv-format traffic record",
        }
    },
}


class LookupSignatureTool:
    """Signature matching engine against YAML-based attack patterns."""

    def __init__(self, signatures_dir: str | Path):
        """Load all signature YAML files from directory.

        Args:
            signatures_dir: Path to directory containing *.yaml signature files.
        """
        self.signatures: list[dict[str, Any]] = []
        sig_dir = Path(signatures_dir)
        if sig_dir.exists():
            for f in sorted(sig_dir.glob("*.yaml")):
                with open(f, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                    if isinstance(data, list):
                        self.signatures.extend(data)
        logger.info("LookupSignatureTool loaded %d signatures", len(self.signatures))

    def __call__(self, traffic_text: str, top_k: int = 3) -> str:
        """Match traffic against signatures.

        Returns:
            JSON string with top-k matches.
        """
        features = parse_kv_text(traffic_text)
        matches = []

        for sig in self.signatures:
            pattern = sig.get("pattern", {})
            if not pattern:
                continue

            matched = 0
            total = len(pattern)
            matching_features = []

            for feat_name, expected_vals in pattern.items():
                actual = features.get(feat_name)
                if actual is None:
                    continue
                # expected_vals can be a single value or list
                if isinstance(expected_vals, list):
                    if actual in expected_vals:
                        matched += 1
                        matching_features.append(f"{feat_name}={actual}")
                else:
                    if actual == str(expected_vals):
                        matched += 1
                        matching_features.append(f"{feat_name}={actual}")

            similarity = matched / total if total > 0 else 0.0
            if similarity > 0:
                matches.append({
                    "signature_id": sig.get("id", "unknown"),
                    "attack_type": sig.get("type", "unknown"),
                    "name": sig.get("name", "unknown"),
                    "similarity": round(similarity, 3),
                    "matching_features": matching_features,
                })

        # Sort by similarity descending, take top-k
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        top_matches = matches[:top_k]

        result = {
            "matches": top_matches,
            "best_match_similarity": top_matches[0]["similarity"] if top_matches else 0.0,
            "no_match": len(top_matches) == 0,
        }
        return json.dumps(result, ensure_ascii=False)
