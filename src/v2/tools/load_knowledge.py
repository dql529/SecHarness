"""
load_knowledge.py — Two-layer knowledge loading.

Layer 1: Topic list for system prompt (~100 tokens/topic)
Layer 2: Full content loaded on demand (~200-500 tokens/topic)
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "name": "load_knowledge",
    "description": (
        "Load domain knowledge on a specific topic. "
        "Call get_topic_list() first to see available topics."
    ),
    "parameters": {
        "topic": {
            "type": "string",
            "description": "Knowledge topic to load (e.g., 'dos_patterns', 'tcp_baseline')",
        }
    },
}


class KnowledgeLoader:
    """Scans data/knowledge/ for .md files and provides two-layer loading.

    Directory structure expected:
        data/knowledge/
        ├── attack_patterns/
        │   ├── dos_patterns.md
        │   └── ...
        ├── protocol_baselines/
        │   └── tcp_baseline.md
        └── network_baselines/
            ├── unsw_baseline.md
            └── cic_baseline.md
    """

    def __init__(self, knowledge_dir: str | Path):
        self._dir = Path(knowledge_dir)
        self._topics: dict[str, Path] = {}  # topic_name -> file_path
        self._descriptions: dict[str, str] = {}  # topic_name -> one-line desc
        self._scan()

    def _scan(self) -> None:
        """Scan knowledge directory for .md files."""
        if not self._dir.exists():
            logger.warning("Knowledge directory not found: %s", self._dir)
            return

        for md_file in sorted(self._dir.rglob("*.md")):
            topic = md_file.stem  # e.g., "dos_patterns"
            self._topics[topic] = md_file

            # Extract first line as description
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    first_line = ""
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            first_line = line
                            break
                        elif line.startswith("# "):
                            first_line = line[2:]
                            break
                    self._descriptions[topic] = first_line or topic
            except OSError:
                self._descriptions[topic] = topic

        logger.info("KnowledgeLoader: found %d topics in %s", len(self._topics), self._dir)

    def get_topic_list(self) -> str:
        """Layer 1: Generate topic list text for system prompt."""
        if not self._topics:
            return "No knowledge topics available."

        lines = ["Available knowledge topics:"]
        for topic, desc in sorted(self._descriptions.items()):
            lines.append(f"  - {topic}: {desc}")
        return "\n".join(lines)

    def load(self, topic: str) -> str:
        """Layer 2: Load full content of a topic.

        Returns:
            Topic content string, or error message if not found.
        """
        path = self._topics.get(topic)
        if path is None:
            available = ", ".join(sorted(self._topics.keys()))
            return f'{{"error": "Unknown topic: {topic}. Available: {available}"}}'

        try:
            content = path.read_text(encoding="utf-8")
            return content
        except OSError as e:
            return f'{{"error": "Failed to load {topic}: {e}"}}'

    def __call__(self, topic: str) -> str:
        """Tool interface — same as load()."""
        return self.load(topic)

    @property
    def topics(self) -> list[str]:
        return sorted(self._topics.keys())
