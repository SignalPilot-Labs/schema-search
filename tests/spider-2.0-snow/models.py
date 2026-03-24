"""Data models for Spider 2.0-Snow evaluation."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Instance:
    """A single Spider 2.0-Snow evaluation instance."""

    instance_id: str
    instruction: str
    db_id: str
    external_knowledge: Optional[str]


@dataclass
class AgentResult:
    """Result from a single agent run."""

    instance_id: str
    sql: str
    error: Optional[str]
    tool_calls_count: int
    messages: list = field(default_factory=list)


@dataclass
class ModeStats:
    """Tracks per-instance results for a mode.

    Public API:
        record_success(instance_id, tool_calls_count, latency) — record a successful run.
        record_failure(instance_id) — record a failed run.
    """

    succeeded: int = 0
    failed: int = 0
    tool_calls: list[int] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    outcomes: dict[str, int] = field(default_factory=dict)

    def record_success(
        self, instance_id: str, tool_calls_count: int, latency: float
    ) -> None:
        """Record a successful instance run."""
        self.succeeded += 1
        self.outcomes[instance_id] = 1
        self.tool_calls.append(tool_calls_count)
        self.latencies.append(latency)

    def record_failure(self, instance_id: str) -> None:
        """Record a failed instance run."""
        self.failed += 1
        self.outcomes[instance_id] = 0
