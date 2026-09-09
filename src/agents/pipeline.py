"""
pipeline.py — Full SecHarness detection pipeline.

Traffic → Agent-Alpha + Agent-Beta (parallel) → Consensus Module → ConsensusResult

Supports single-sample and batch processing. Alpha and Beta agents run
concurrently via ThreadPoolExecutor for LLM-based agents.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Optional

from .base_agent import BaseAgent, AgentVerdict
from .consensus import ConsensusModule, ConsensusResult

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """End-to-end SecHarness detection pipeline.

    Parameters
    ----------
    alpha : BaseAgent
        Agent-Alpha (behavioral analysis).
    beta : BaseAgent
        Agent-Beta (pattern matching — LLM or ML variant).
    consensus : ConsensusModule
        Consensus module with configured weights and strategy.
    parallel : bool
        Whether to run Alpha and Beta concurrently (default True).
    """

    def __init__(
        self,
        alpha: BaseAgent,
        beta: BaseAgent,
        consensus: Optional[ConsensusModule] = None,
        parallel: bool = True,
    ):
        self.alpha = alpha
        self.beta = beta
        self.consensus = consensus or ConsensusModule()
        self.parallel = parallel

    def detect(self, traffic_text: str) -> ConsensusResult:
        """Run the full pipeline on a single traffic record."""
        t0 = time.perf_counter()

        if self.parallel:
            with ThreadPoolExecutor(max_workers=2) as pool:
                fa: Future = pool.submit(self.alpha.analyze, traffic_text)
                fb: Future = pool.submit(self.beta.analyze, traffic_text)
                alpha_verdict = fa.result()
                beta_verdict = fb.result()
        else:
            alpha_verdict = self.alpha.analyze(traffic_text)
            beta_verdict = self.beta.analyze(traffic_text)

        result = self.consensus.resolve(alpha_verdict, beta_verdict)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def detect_batch(
        self, traffic_texts: List[str], batch_size: int = 32
    ) -> List[ConsensusResult]:
        """Run the pipeline on a batch of traffic records.

        Alpha and Beta process their full batches concurrently,
        then results are paired for consensus.
        """
        all_results: List[ConsensusResult] = []

        for start in range(0, len(traffic_texts), batch_size):
            chunk = traffic_texts[start : start + batch_size]
            t0 = time.perf_counter()

            if self.parallel:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    fa = pool.submit(self.alpha.analyze_batch, chunk)
                    fb = pool.submit(self.beta.analyze_batch, chunk)
                    alpha_verdicts = fa.result()
                    beta_verdicts = fb.result()
            else:
                alpha_verdicts = self.alpha.analyze_batch(chunk)
                beta_verdicts = self.beta.analyze_batch(chunk)

            chunk_results = self.consensus.resolve_batch(alpha_verdicts, beta_verdicts)
            batch_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "Batch [%d:%d] processed in %.0f ms",
                start, start + len(chunk), batch_ms,
            )
            all_results.extend(chunk_results)

        return all_results

    def get_stats(self, results: List[ConsensusResult]) -> dict:
        """Compute summary statistics over a list of consensus results."""
        if not results:
            return {}

        n = len(results)
        verdicts = [r.final_verdict for r in results]
        ctypes = [r.consensus_type for r in results]

        return {
            "total": n,
            "benign": verdicts.count("benign"),
            "attack": verdicts.count("attack"),
            "escalate": verdicts.count("escalate"),
            "agreement_rate": ctypes.count("agreement") / n,
            "disagreement_rate": 1 - ctypes.count("agreement") / n,
            "consensus_type_dist": {
                t: ctypes.count(t) / n
                for t in ("agreement", "alpha_only", "beta_only", "type_conflict", "uncertain")
            },
            "avg_confidence": sum(r.combined_confidence for r in results) / n,
        }

    def __repr__(self) -> str:
        return (
            f"<DetectionPipeline alpha={self.alpha.name!r} "
            f"beta={self.beta.name!r} parallel={self.parallel}>"
        )
