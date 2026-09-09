"""
SecHarness — Harness Engineering components for LLM-based IDS.

Components:
- Observability: audit trail recording + aggregation
- Metrics: detection performance computation (Accuracy, F1, FPR, etc.)
"""

from __future__ import annotations

# Re-export ConsensusResult from the canonical location
from ..agents.consensus import ConsensusResult

from .observability import AuditRecord, AuditLogger, LatencyBreakdown, TokenUsage, build_audit_record
from .metrics import (
    compute_binary_metrics,
    compute_perclass_metrics,
    compute_zeroday_rate,
    compute_harness_metrics,
    compute_efficiency_metrics,
    export_all_metrics,
)

__all__ = [
    "ConsensusResult",
    "AuditRecord",
    "AuditLogger",
    "LatencyBreakdown",
    "TokenUsage",
    "build_audit_record",
    "compute_binary_metrics",
    "compute_perclass_metrics",
    "compute_zeroday_rate",
    "compute_harness_metrics",
    "compute_efficiency_metrics",
    "export_all_metrics",
]
