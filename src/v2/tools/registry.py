"""
registry.py — Tool dispatch map + Observation preprocessor.

Manages tool registration, schema generation, and traffic pre-analysis.
parse_traffic is not a tool — it is an Observation-layer preprocessor.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from ...agents.beta_agent_ml import parse_kv_text
from .check_anomaly import TOOL_SCHEMA as CHECK_ANOMALY_SCHEMA, TOOL_SCHEMA_DEGRADED as CHECK_ANOMALY_DEGRADED_SCHEMA
from .lookup_signature import TOOL_SCHEMA as LOOKUP_SIGNATURE_SCHEMA
from .query_history import TOOL_SCHEMA as QUERY_HISTORY_SCHEMA
from .load_knowledge import TOOL_SCHEMA as LOAD_KNOWLEDGE_SCHEMA
from .actions import CLASSIFY_SCHEMA, ESCALATE_SCHEMA, LOG_DECISION_SCHEMA

logger = logging.getLogger(__name__)

# All available tool schemas
ALL_TOOL_SCHEMAS = {
    "check_anomaly": CHECK_ANOMALY_SCHEMA,
    "lookup_signature": LOOKUP_SIGNATURE_SCHEMA,
    "query_history": QUERY_HISTORY_SCHEMA,
    "load_knowledge": LOAD_KNOWLEDGE_SCHEMA,
    "classify": CLASSIFY_SCHEMA,
    "escalate": ESCALATE_SCHEMA,
    "log_decision": LOG_DECISION_SCHEMA,
}

# Anomalous feature values that should be highlighted
_ANOMALOUS_VALUES = {"very_high", "zero"}
# Features where "zero" is suspicious
_ZERO_SUSPICIOUS = {"dbytes", "dpkts", "sbytes", "spkts", "dur"}


class ToolRegistry:
    """Tool dispatch map with ablation support.

    Usage:
        registry = ToolRegistry(enabled_tools=["check_anomaly", "classify"])
        registry.register("check_anomaly", check_anomaly_handler)
        output = registry.dispatch("check_anomaly", {"traffic_text": "..."})
    """

    def __init__(self, enabled_tools: list[str] | None = None):
        """Initialize registry.

        Args:
            enabled_tools: If provided, only these tools are registered.
                           If None, all tools are available.
        """
        self._handlers: dict[str, Callable] = {}
        self._schemas: dict[str, dict] = {}
        self._enabled = set(enabled_tools) if enabled_tools else None

    def register(self, name: str, handler: Callable) -> None:
        """Register a tool handler."""
        if self._enabled is not None and name not in self._enabled:
            logger.debug("Skipping disabled tool: %s", name)
            return
        self._handlers[name] = handler
        if name in ALL_TOOL_SCHEMAS:
            self._schemas[name] = ALL_TOOL_SCHEMAS[name]

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool by name.

        Returns:
            JSON string result, or error JSON if tool not found.
        """
        handler = self._handlers.get(name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            return handler(**args)
        except Exception as e:
            logger.error("Tool %s failed: %s", name, e)
            return json.dumps({"error": f"Tool {name} failed: {str(e)}"})

    def override_schema(self, name: str, schema: dict) -> None:
        """Override the schema for a registered tool (e.g., degraded mode)."""
        if name in self._schemas:
            self._schemas[name] = schema

    def has_tool(self, name: str) -> bool:
        return name in self._handlers

    @property
    def tool_names(self) -> list[str]:
        return list(self._handlers.keys())

    def build_tool_descriptions(self) -> str:
        """Generate tool list text for the system prompt."""
        lines = ["Available tools (call one at a time using JSON format):"]
        for name, schema in sorted(self._schemas.items()):
            desc = schema.get("description", "")
            params = schema.get("parameters", {})
            param_strs = []
            for pname, pinfo in params.items():
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")
                param_strs.append(f"    {pname} ({ptype}): {pdesc}")
            lines.append(f"\n- {name}: {desc}")
            if param_strs:
                lines.extend(param_strs)
        return "\n".join(lines)


def build_traffic_observation(traffic_text: str) -> str:
    """Pre-analyze traffic and generate structured observation for user prompt.

    This is not a tool: the observation is embedded in the user prompt.
    Highlights anomalous features so the Agent can skip parse_traffic.
    """
    features = parse_kv_text(traffic_text)
    if not features:
        return f"Traffic record: {traffic_text}\nPre-analysis: Unable to parse features."

    anomalous = []
    normal = []

    for key, value in features.items():
        if value in _ANOMALOUS_VALUES:
            if value == "zero" and key in _ZERO_SUSPICIOUS:
                anomalous.append(f"{key}={value} (suspicious zero)")
            elif value == "very_high":
                anomalous.append(f"{key}={value} (extreme value)")
            else:
                normal.append(f"{key}={value}")
        else:
            normal.append(f"{key}={value}")

    proto = features.get("proto", "unknown")
    state = features.get("state", "unknown")

    lines = [
        f"Traffic record: {traffic_text}",
        "Pre-analysis:",
        f"  Protocol: {proto}, State: {state}",
        f"  Features: {len(features)} total, {len(anomalous)} anomalous",
    ]
    if anomalous:
        lines.append(f"  Anomalous: {', '.join(anomalous)}")
    if normal:
        lines.append(f"  Normal: {', '.join(normal[:10])}")
        if len(normal) > 10:
            lines.append(f"  ... and {len(normal) - 10} more normal features")

    return "\n".join(lines)
