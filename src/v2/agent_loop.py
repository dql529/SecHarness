"""
agent_loop.py — Core Agent Loop + SecHarness orchestrator for v2.

Implements the single-Agent + Harness tool-chain architecture.
Implements the agent-loop pseudocode of the system design.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .llm_engine import LLMEngine, LLMResponse
from .tool_parser import ToolCall, DirectVerdict, ParseFailure, parse_tool_call
from .tools.registry import ToolRegistry, build_traffic_observation
from .tools.check_anomaly import CheckAnomalyTool, TOOL_SCHEMA_DEGRADED as CHECK_ANOMALY_DEGRADED_SCHEMA
from .tools.lookup_signature import LookupSignatureTool
from .tools.query_history import QueryHistoryTool
from .tools.load_knowledge import KnowledgeLoader
from .tools.actions import handle_escalate, handle_log_decision
from .harness.audit import AuditRecordV2, AuditLoggerV2, ToolCallRecord, LLMCallRecord
from .harness.permissions import PermissionsManager

logger = logging.getLogger(__name__)


def _safe_float(value, default: float) -> float:
    """Coerce a possibly-None/string/garbage LLM-supplied value to float.

    The LLM may emit ``"confidence": null`` or a non-numeric string; the bare
    ``float(...)`` would raise. Fall back to *default* in those cases.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

# System prompt for Agent with tools (E3/E4)
SYSTEM_PROMPT_WITH_TOOLS = """\
You are a network intrusion detection agent. Your task: analyze a traffic record and classify it as benign or attack.

You have access to tools. Each turn, output EXACTLY ONE JSON object — either a tool call or a final classification.

Tool call format:
{{"tool": "<tool_name>", "args": {{<arguments>}}}}

Final classification format (use the classify tool):
{{"tool": "classify", "args": {{"verdict": "benign"|"attack", "attack_type": "<type or null>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}}}

Strategy:
1. Review the pre-analysis in the traffic record.
2. Call check_anomaly to get ML-based anomaly detection results.
3. If ML confidence is high (>0.95), you can classify directly.
4. If uncertain, use lookup_signature or load_knowledge for more evidence.
5. Call classify when ready with your final verdict.

Important:
- Output ONLY the JSON object, no extra text.
- Each tool takes traffic_text as input (the raw kv-format string from the record).
- When verdict is "attack", always specify attack_type.

{tool_descriptions}

{knowledge_topics}"""

# System prompt without tools (E1/E2) — single-shot classification
SYSTEM_PROMPT_NO_TOOLS = """\
You are a network intrusion detection expert specializing in behavioral analysis.

Your task: analyze a network traffic record (key=value format) and determine whether it represents benign activity or an attack.

Analysis approach — think step by step:
1. Parse each feature and its value (low/medium/high/very_high/zero/missing).
2. Identify behavioral anomalies: unusual protocol-state-service combinations, \
extreme byte/packet ratios, suspicious timing patterns, abnormal connection counts.
3. Consider what attack behaviors these features could indicate \
(e.g., high sbytes + zero dbytes → data exfiltration; many connections + low duration → scanning).
4. Make your verdict with a calibrated confidence score.

Respond ONLY with a JSON object (no markdown, no extra text):
{{"verdict": "benign" or "attack", "attack_type": null or attack category name, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""


@dataclass
class AgentResult:
    """Result of processing one traffic sample through the agent loop."""
    verdict: str = "benign"
    attack_type: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""
    termination: str = ""  # "classify" | "max_steps" | "parse_fail" | "direct_verdict" | "escalate"
    audit: Optional[AuditRecordV2] = None


class SecHarness:
    """Assembles all v2 components: LLM + Tools + Knowledge + Permissions + Audit.

    Usage:
        harness = SecHarness(
            llm=engine, ml_model_path="path/to/rf.pkl",
            knowledge_dir="data/knowledge/",
            harness_enabled=True,
        )
        result = agent_loop(traffic_text, harness)
    """

    def __init__(
        self,
        llm: LLMEngine,
        ml_model_path: Optional[str] = None,
        signatures_dir: Optional[str] = None,
        knowledge_dir: Optional[str] = None,
        harness_enabled: bool = True,
        enabled_tools: Optional[list[str]] = None,
        permissions_enabled: bool = True,
        max_steps: int = 5,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        confidence_threshold: float = 0.3,
        ml_conflict_threshold: float = 0.9,
        max_reasoning_chars: int = 500,
        audit_logger: Optional[AuditLoggerV2] = None,
        check_anomaly_degraded: bool = False,
        rf_context: bool = False,
    ):
        self.llm = llm
        self.harness_enabled = harness_enabled
        self.rf_context = rf_context
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Tools
        self.registry = ToolRegistry(enabled_tools=enabled_tools)
        self.check_anomaly: Optional[CheckAnomalyTool] = None
        self.query_history: Optional[QueryHistoryTool] = None
        self.knowledge_loader: Optional[KnowledgeLoader] = None

        # RF-context mode: load ML model for prompt injection but no tool loop
        if rf_context and ml_model_path and not harness_enabled:
            self.check_anomaly = CheckAnomalyTool(ml_model_path, degraded=False)

        if harness_enabled:
            # Register analysis tools
            if ml_model_path:
                self.check_anomaly = CheckAnomalyTool(ml_model_path, degraded=check_anomaly_degraded)
                self.registry.register("check_anomaly", self.check_anomaly)
                if check_anomaly_degraded:
                    self.registry.override_schema("check_anomaly", CHECK_ANOMALY_DEGRADED_SCHEMA)

            if signatures_dir:
                sig_tool = LookupSignatureTool(signatures_dir)
                self.registry.register("lookup_signature", sig_tool)

            self.query_history = QueryHistoryTool()
            self.registry.register("query_history", self.query_history)

            if knowledge_dir:
                self.knowledge_loader = KnowledgeLoader(knowledge_dir)
                self.registry.register("load_knowledge", self.knowledge_loader)

            # Register action tools
            self.registry.register("escalate", handle_escalate)
            self.registry.register("log_decision", handle_log_decision)
            # classify is handled specially in the loop (not dispatched)

        # Permissions
        self.permissions = PermissionsManager(
            enabled=permissions_enabled,
            confidence_threshold=confidence_threshold,
            ml_conflict_threshold=ml_conflict_threshold,
            max_reasoning_chars=max_reasoning_chars,
        )

        # Audit
        self.audit_logger = audit_logger

    def build_system_prompt(self) -> str:
        """Generate system prompt based on configuration."""
        if not self.harness_enabled:
            return SYSTEM_PROMPT_NO_TOOLS

        tool_desc = self.registry.build_tool_descriptions()
        knowledge_topics = ""
        if self.knowledge_loader:
            knowledge_topics = self.knowledge_loader.get_topic_list()

        return SYSTEM_PROMPT_WITH_TOOLS.format(
            tool_descriptions=tool_desc,
            knowledge_topics=knowledge_topics,
        )

    def build_user_prompt(self, traffic_text: str) -> str:
        """Build user prompt, with observation pre-analysis if harness enabled."""
        if self.harness_enabled:
            return build_traffic_observation(traffic_text)
        if self.rf_context and self.check_anomaly is not None:
            rf_json = self.check_anomaly(traffic_text)
            rf_result = json.loads(rf_json)
            return (
                f"Analyze this network traffic record:\n{traffic_text}\n\n"
                f"--- ML Classifier Opinion ---\n"
                f"An automated ML classifier analyzed this traffic and produced the following assessment:\n"
                f"Prediction: {rf_result.get('prediction', 'unknown')}\n"
                f"Confidence: {rf_result.get('confidence', 0.0)}\n"
                f"Top-3 classes: {rf_result.get('top3', 'N/A')}\n"
                f"Anomaly score: {rf_result.get('anomaly_score', 0.0)}\n"
                f"---\n"
                f"Consider this ML opinion alongside your own analysis, then make your final verdict."
            )
        return f"Analyze this network traffic record:\n{traffic_text}"

    def process_batch(
        self,
        traffic_texts: list[str],
        ground_truths: Optional[list[str]] = None,
        sample_indices: Optional[list[int]] = None,
        is_zeroday: Optional[list[bool]] = None,
    ) -> list[AgentResult]:
        """Batch process traffic samples with progress tracking."""
        results = []
        total = len(traffic_texts)

        for i, text in enumerate(traffic_texts):
            idx = sample_indices[i] if sample_indices else i
            gt = ground_truths[i] if ground_truths else None
            zd = is_zeroday[i] if is_zeroday else False

            result = agent_loop(text, self, sample_index=idx, ground_truth=gt, is_zeroday=zd)
            results.append(result)

            if (i + 1) % 100 == 0 or (i + 1) == total:
                logger.info("Progress: %d/%d (%.1f%%)", i + 1, total, (i + 1) / total * 100)

        return results


def agent_loop(
    traffic_text: str,
    harness: SecHarness,
    sample_index: int = 0,
    ground_truth: Optional[str] = None,
    is_zeroday: bool = False,
) -> AgentResult:
    """Process a single traffic record through the agent loop.

    Two modes:
    - harness_enabled=False: Single LLM call, parse verdict directly (E1/E2)
    - harness_enabled=True: Multi-step loop with tool calls (E3/E4)
    """
    audit = AuditRecordV2(
        sample_index=sample_index,
        traffic_text=traffic_text,
        is_zeroday=is_zeroday,
        model=harness.llm.base_model_name,
        adapter=harness.llm.adapter_path or "",
        harness_enabled=harness.harness_enabled,
        temperature=harness.temperature,
        max_steps=harness.max_steps,
    )

    if not harness.harness_enabled:
        result = _single_shot(traffic_text, harness, audit)
    else:
        result = _tool_loop(traffic_text, harness, audit)

    # Finalize audit
    audit.compute_efficiency()
    if ground_truth:
        audit.evaluate(ground_truth)
    result.audit = audit

    # Log
    if harness.audit_logger:
        harness.audit_logger.log(audit)

    return result


def _single_shot(traffic_text: str, harness: SecHarness, audit: AuditRecordV2) -> AgentResult:
    """E1/E2: Single LLM call without tools."""
    messages = [
        {"role": "system", "content": harness.build_system_prompt()},
        {"role": "user", "content": harness.build_user_prompt(traffic_text)},
    ]

    response = harness.llm.generate(
        messages, max_new_tokens=harness.max_new_tokens, temperature=harness.temperature,
    )
    audit.llm_calls.append(LLMCallRecord(
        step=0, input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=response.latency_ms,
        raw_output=response.text[:300],
    ))

    parsed = parse_tool_call(response.text)

    if isinstance(parsed, DirectVerdict):
        audit.verdict = parsed.verdict
        audit.attack_type = parsed.attack_type
        audit.confidence = parsed.confidence
        audit.reasoning = parsed.reasoning
        audit.termination = "direct_verdict"
        return AgentResult(
            verdict=parsed.verdict, attack_type=parsed.attack_type,
            confidence=parsed.confidence, reasoning=parsed.reasoning,
            termination="direct_verdict",
        )
    elif isinstance(parsed, ToolCall) and parsed.name == "classify":
        args = parsed.args
        audit.verdict = str(args.get("verdict", "benign"))
        audit.attack_type = args.get("attack_type")
        audit.confidence = _safe_float(args.get("confidence"), 0.5)
        audit.reasoning = str(args.get("reasoning", ""))
        audit.termination = "classify"
        return AgentResult(
            verdict=audit.verdict, attack_type=audit.attack_type,
            confidence=audit.confidence, reasoning=audit.reasoning,
            termination="classify",
        )
    else:
        # Parse failure
        audit.verdict = "benign"
        audit.confidence = 0.0
        audit.reasoning = f"[PARSE_FAIL] {response.text[:200]}"
        audit.termination = "parse_fail"
        return AgentResult(
            verdict="benign", confidence=0.0,
            reasoning=audit.reasoning, termination="parse_fail",
        )


def _tool_loop(traffic_text: str, harness: SecHarness, audit: AuditRecordV2) -> AgentResult:
    """E3/E4: Multi-step agent loop with tool calls."""
    messages = [
        {"role": "system", "content": harness.build_system_prompt()},
        {"role": "user", "content": harness.build_user_prompt(traffic_text)},
    ]

    last_ml_prediction: Optional[dict] = None
    already_retried = False

    for step in range(harness.max_steps):
        # LLM call
        response = harness.llm.generate(
            messages, max_new_tokens=harness.max_new_tokens, temperature=harness.temperature,
        )
        audit.llm_calls.append(LLMCallRecord(
            step=step, input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            raw_output=response.text[:300],
        ))

        messages.append({"role": "assistant", "content": response.text})

        # Parse output
        parsed = parse_tool_call(response.text)

        if isinstance(parsed, ParseFailure):
            audit.verdict = "benign"
            audit.confidence = 0.0
            audit.reasoning = f"[PARSE_FAIL@step{step}] {response.text[:200]}"
            audit.termination = "parse_fail"
            return AgentResult(
                verdict="benign", confidence=0.0,
                reasoning=audit.reasoning, termination="parse_fail",
            )

        if isinstance(parsed, DirectVerdict):
            audit.verdict = parsed.verdict
            audit.attack_type = parsed.attack_type
            audit.confidence = parsed.confidence
            audit.reasoning = parsed.reasoning
            audit.termination = "direct_verdict"
            return AgentResult(
                verdict=parsed.verdict, attack_type=parsed.attack_type,
                confidence=parsed.confidence, reasoning=parsed.reasoning,
                termination="direct_verdict",
            )

        # It's a ToolCall
        tool_name = parsed.name
        tool_args = parsed.args

        # Handle classify (terminal)
        if tool_name == "classify":
            # Permissions check
            perm_result = harness.permissions.check_classify(
                tool_args, ml_prediction=last_ml_prediction, already_retried=already_retried,
            )

            if perm_result.needs_retry and not already_retried:
                # Give Agent one retry chance
                already_retried = True
                messages.append({"role": "user", "content": perm_result.retry_message})
                audit.self_corrected = True
                continue

            final_args = perm_result.modified_args or tool_args
            verdict = str(final_args.get("verdict", "benign"))
            attack_type = final_args.get("attack_type")
            confidence = _safe_float(final_args.get("confidence"), 0.5)
            reasoning = str(final_args.get("reasoning", ""))
            low_conf = confidence < 0.5

            audit.tool_chain.append(ToolCallRecord(
                step=step, tool="classify",
                input=final_args,
                output=json.dumps({"accepted": True, "low_confidence": low_conf}),
                latency_ms=0.0, timestamp=time.time(),
            ))
            audit.verdict = verdict
            audit.attack_type = attack_type
            audit.confidence = confidence
            audit.reasoning = reasoning
            audit.termination = "classify"
            audit.low_confidence = low_conf

            return AgentResult(
                verdict=verdict, attack_type=attack_type,
                confidence=confidence, reasoning=reasoning,
                termination="classify",
            )

        # Handle escalate (terminal)
        if tool_name == "escalate":
            output = harness.registry.dispatch("escalate", tool_args)
            audit.tool_chain.append(ToolCallRecord(
                step=step, tool="escalate", input=tool_args,
                output=output, latency_ms=0.0, timestamp=time.time(),
            ))
            audit.escalated = True
            # Fallback to partial verdict
            pv = str(tool_args.get("partial_verdict", "benign"))
            if pv not in ("benign", "attack"):
                pv = "benign"
            conf = _safe_float(tool_args.get("confidence"), 0.3)
            audit.verdict = pv
            audit.confidence = conf
            audit.reasoning = f"[ESCALATED] {tool_args.get('reason', '')}"
            audit.termination = "escalate"

            return AgentResult(
                verdict=pv, confidence=conf,
                reasoning=audit.reasoning, termination="escalate",
            )

        # Non-terminal tool call — dispatch
        # Always override traffic_text for analysis tools to prevent LLM from
        # passing truncated or hallucinated feature strings
        if tool_name in ("check_anomaly", "lookup_signature", "query_history"):
            tool_args["traffic_text"] = traffic_text

        t0 = time.perf_counter()
        output = harness.registry.dispatch(tool_name, tool_args)
        tool_latency = (time.perf_counter() - t0) * 1000

        audit.tool_chain.append(ToolCallRecord(
            step=step, tool=tool_name, input=tool_args,
            output=output, latency_ms=tool_latency, timestamp=time.time(),
        ))

        # Track ML prediction for permissions. A dispatch error (e.g. the tool is
        # disabled in an ablation) is itself valid JSON, so it must not be recorded
        # as a prediction: downstream .get("prediction", "Normal") would silently
        # turn the error into a benign verdict.
        if tool_name == "check_anomaly":
            try:
                tool_output = json.loads(output)
            except json.JSONDecodeError:
                pass
            else:
                if "prediction" in tool_output:
                    last_ml_prediction = tool_output

        # Track knowledge loaded
        if tool_name == "load_knowledge":
            topic = tool_args.get("topic", "")
            if topic and topic not in audit.knowledge_loaded:
                audit.knowledge_loaded.append(topic)

        # Feed tool result back to Agent
        messages.append({"role": "user", "content": f"Tool result ({tool_name}):\n{output}"})

    # Max steps exceeded — fallback
    return _force_verdict(audit, last_ml_prediction, harness)


def _force_verdict(
    audit: AuditRecordV2,
    last_ml_prediction: Optional[dict],
    harness: SecHarness,
) -> AgentResult:
    """Force a verdict when max_steps is exceeded."""
    audit.termination = "max_steps"

    if last_ml_prediction and "prediction" in last_ml_prediction:
        pred = last_ml_prediction["prediction"]
        is_attack = pred.lower() not in ("normal", "benign")
        audit.verdict = "attack" if is_attack else "benign"
        audit.attack_type = pred if is_attack else None
        audit.confidence = float(last_ml_prediction.get("confidence", 0.5))
        audit.reasoning = f"[MAX_STEPS] Fallback to ML prediction: {pred}"
    else:
        audit.verdict = "benign"
        audit.confidence = 0.0
        audit.reasoning = "[MAX_STEPS] No ML prediction available, defaulting to benign"

    audit.low_confidence = True
    return AgentResult(
        verdict=audit.verdict, attack_type=audit.attack_type,
        confidence=audit.confidence, reasoning=audit.reasoning,
        termination="max_steps",
    )
