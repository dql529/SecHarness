#!/usr/bin/env python3
"""
extract_trajectories.py — Extract tool-use trajectory SFT data from E3 audit logs.

Reads successful E3 audit records and rebuilds multi-turn conversation trajectories
suitable for training the fine-tuned model to output tool calls (not direct JSON verdicts).

No torch/transformers imports — pure data processing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import types
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extract_trajectories")

# ---------------------------------------------------------------------------
# Dynamic import of SecHarness modules (no torch)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # project/
SRC_DIR = PROJECT_ROOT / "src"


def _setup_imports():
    """Load SecHarness modules without triggering torch imports.

    Creates minimal package stubs and loads individual .py files via importlib.
    Returns (SYSTEM_PROMPT_WITH_TOOLS, ToolRegistry, ALL_TOOL_SCHEMAS, build_traffic_observation).
    """
    # Create package stubs
    for pkg_name, pkg_path in [
        ("src", str(SRC_DIR)),
        ("src.agents", str(SRC_DIR / "agents")),
        ("src.v2", str(SRC_DIR / "v2")),
        ("src.v2.tools", str(SRC_DIR / "v2" / "tools")),
        ("src.v2.harness", str(SRC_DIR / "v2" / "harness")),
    ]:
        mod = types.ModuleType(pkg_name)
        mod.__path__ = [pkg_path]
        sys.modules[pkg_name] = mod

    def _load(name: str, path: str):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    # Load dependency chain (no torch in any of these)
    _load("src.agents.base_agent", str(SRC_DIR / "agents" / "base_agent.py"))
    _load("src.agents.beta_agent_ml", str(SRC_DIR / "agents" / "beta_agent_ml.py"))
    _load("src.v2.tools.check_anomaly", str(SRC_DIR / "v2" / "tools" / "check_anomaly.py"))
    _load("src.v2.tools.lookup_signature", str(SRC_DIR / "v2" / "tools" / "lookup_signature.py"))
    _load("src.v2.tools.query_history", str(SRC_DIR / "v2" / "tools" / "query_history.py"))
    _load("src.v2.tools.load_knowledge", str(SRC_DIR / "v2" / "tools" / "load_knowledge.py"))
    _load("src.v2.tools.actions", str(SRC_DIR / "v2" / "tools" / "actions.py"))
    registry_mod = _load("src.v2.tools.registry", str(SRC_DIR / "v2" / "tools" / "registry.py"))

    # Load agent_loop for SYSTEM_PROMPT_WITH_TOOLS
    # agent_loop imports LLMEngine (deferred torch) + tool_parser + harness modules
    # We need stubs for those
    _load("src.v2.llm_engine", str(SRC_DIR / "v2" / "llm_engine.py"))
    _load("src.v2.tool_parser", str(SRC_DIR / "v2" / "tool_parser.py"))
    _load("src.v2.harness.audit", str(SRC_DIR / "v2" / "harness" / "audit.py"))
    _load("src.v2.harness.permissions", str(SRC_DIR / "v2" / "harness" / "permissions.py"))
    agent_loop_mod = _load("src.v2.agent_loop", str(SRC_DIR / "v2" / "agent_loop.py"))

    assert "torch" not in sys.modules, "torch was imported — script should be CPU-only"

    return (
        agent_loop_mod.SYSTEM_PROMPT_WITH_TOOLS,
        registry_mod.ToolRegistry,
        registry_mod.ALL_TOOL_SCHEMAS,
        registry_mod.build_traffic_observation,
    )


def build_system_prompt(
    prompt_template: str,
    ToolRegistry,
    ALL_TOOL_SCHEMAS: dict,
) -> str:
    """Build system prompt with tool descriptions (no knowledge topics for SFT)."""
    # Register all tools with dummy handlers to generate descriptions
    reg = ToolRegistry()
    for name, schema in ALL_TOOL_SCHEMAS.items():
        reg._handlers[name] = lambda: None
        reg._schemas[name] = schema
    tool_desc = reg.build_tool_descriptions()
    return prompt_template.format(tool_descriptions=tool_desc, knowledge_topics="")


def extract_trajectory(
    audit: dict,
    system_prompt: str,
    build_observation,
) -> list[dict] | None:
    """Convert one audit record into a multi-turn messages list.

    Returns None if the record should be filtered out.
    """
    result = audit["result"]

    # Only keep successful classify terminations
    if result["termination"] != "classify":
        return None

    traffic_text = audit["input"]["traffic_text"]
    tool_chain = audit["tool_chain"]

    if not tool_chain:
        return None

    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_observation(traffic_text)},
    ]

    for tc in tool_chain:
        tool_name = tc["tool"]

        if tool_name == "classify":
            # Terminal tool call — build classify args from the tool_chain input
            classify_input = tc["input"]
            assistant_content = json.dumps({
                "tool": "classify",
                "args": {
                    "verdict": classify_input.get("verdict", result["verdict"]),
                    "attack_type": classify_input.get("attack_type", result.get("attack_type")),
                    "confidence": classify_input.get("confidence", result["confidence"]),
                    "reasoning": classify_input.get("reasoning", result.get("reasoning", "")),
                },
            })
            messages.append({"role": "assistant", "content": assistant_content})
        else:
            # Non-terminal tool call — args set to {} because agent_loop
            # auto-injects traffic_text (line 424)
            assistant_content = json.dumps({"tool": tool_name, "args": {}})
            messages.append({"role": "assistant", "content": assistant_content})

            # Tool result feedback (matches agent_loop.py line 450)
            messages.append({
                "role": "user",
                "content": f"Tool result ({tool_name}):\n{tc['output']}",
            })

    return messages


def verify_trajectory(messages: list[dict], idx: int) -> bool:
    """Validate a single trajectory. Returns True if valid."""
    ok = True

    if len(messages) < 5:
        log.warning("Sample %d: only %d messages (need >= 5)", idx, len(messages))
        ok = False

    # Check all assistant turns are valid JSON
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            try:
                obj = json.loads(msg["content"])
                if "tool" not in obj:
                    log.warning("Sample %d msg %d: missing 'tool' key", idx, i)
                    ok = False
                if "args" not in obj:
                    log.warning("Sample %d msg %d: missing 'args' key", idx, i)
                    ok = False
            except json.JSONDecodeError as e:
                log.warning("Sample %d msg %d: invalid JSON: %s", idx, i, e)
                ok = False

    return ok


def main():
    ap = argparse.ArgumentParser(description="Extract tool-use trajectories from E3 audit logs")
    ap.add_argument("--audit-log", type=str, required=True,
                     help="Path to E3 audit JSONL file")
    ap.add_argument("--output", type=str, required=True,
                     help="Output JSONL path for trajectories")
    ap.add_argument("--verify", type=int, default=0,
                     help="Number of samples to print for manual verification")
    args = ap.parse_args()

    # Dynamic imports
    log.info("Loading SecHarness modules (no torch)...")
    SYSTEM_PROMPT_WITH_TOOLS, ToolRegistry, ALL_TOOL_SCHEMAS, build_traffic_observation = _setup_imports()

    # Build system prompt
    system_prompt = build_system_prompt(SYSTEM_PROMPT_WITH_TOOLS, ToolRegistry, ALL_TOOL_SCHEMAS)
    log.info("System prompt built (%d chars)", len(system_prompt))

    # Read audit log
    audit_path = Path(args.audit_log)
    if not audit_path.exists():
        log.error("Audit log not found: %s", audit_path)
        sys.exit(1)

    records = []
    with open(audit_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info("Loaded %d audit records", len(records))

    # Extract trajectories
    trajectories = []
    filtered = 0
    chain_lengths = Counter()

    for rec in records:
        messages = extract_trajectory(rec, system_prompt, build_traffic_observation)
        if messages is None:
            filtered += 1
            continue

        # Count assistant turns (= tool chain length)
        n_assistant = sum(1 for m in messages if m["role"] == "assistant")
        chain_lengths[n_assistant] += 1

        trajectories.append({"messages": messages})

    log.info("Extracted %d trajectories, filtered %d", len(trajectories), filtered)

    # Validate all trajectories
    valid = 0
    invalid = 0
    for i, traj in enumerate(trajectories):
        if verify_trajectory(traj["messages"], i):
            valid += 1
        else:
            invalid += 1

    log.info("Validation: %d valid, %d invalid", valid, invalid)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for traj in trajectories:
            f.write(json.dumps(traj) + "\n")
    log.info("Written to %s", output_path)

    # Statistics
    log.info("=== Statistics ===")
    log.info("  Total audit records: %d", len(records))
    log.info("  Extracted trajectories: %d", len(trajectories))
    log.info("  Filtered (non-classify): %d", filtered)
    avg_msgs = sum(len(t["messages"]) for t in trajectories) / max(len(trajectories), 1)
    log.info("  Avg messages per trajectory: %.1f", avg_msgs)
    log.info("  Chain length distribution (assistant turns):")
    for length, count in sorted(chain_lengths.items()):
        log.info("    %d tools: %d samples (%.1f%%)", length, count, 100 * count / len(trajectories))

    # Verify samples
    if args.verify > 0:
        log.info("=== Verification samples ===")
        for i in range(min(args.verify, len(trajectories))):
            log.info("--- Sample %d ---", i)
            for msg in trajectories[i]["messages"]:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    log.info("  [system] (%d chars)", len(content))
                elif role == "user" and content.startswith("Tool result"):
                    log.info("  [user/tool_result] %s", content[:120])
                elif role == "user":
                    log.info("  [user/observation] %s", content[:120])
                else:
                    log.info("  [assistant] %s", content[:200])


if __name__ == "__main__":
    main()
