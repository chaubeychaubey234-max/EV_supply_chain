"""
state.py — Lightweight agent state dataclass for the Fleet Electrification Agent.

Stores the complete lifecycle of a single agent invocation:
  user_query → selected_tools → tool_outputs → final_response
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentState:
    """Captures all data produced during one agent execution cycle.

    Attributes:
        user_query:           The raw natural-language query from the user.
        selected_tools:       Names of the tools chosen by the intent router.
        tool_outputs:         Map of tool_name → raw output dict returned by the tool.
        final_response:       Structured response dict returned to the caller.
        execution_status:     Overall run status: ``'success'`` | ``'partial'`` | ``'error'`` | ``'pending'``.
        conversation_history: Ordered list of ``{role, content}`` dicts for multi-turn support.
    """

    user_query: str = ""
    selected_tools: list[str] = field(default_factory=list)
    tool_outputs: dict[str, Any] = field(default_factory=dict)
    final_response: dict[str, Any] = field(default_factory=dict)
    execution_status: str = "pending"
    conversation_history: list[dict[str, str]] = field(default_factory=list)

    # Orchestration fields for LangGraph compatibility
    detected_intent: str = ""
    analysis_mode: str = ""
    confidence: float = 1.0
    analysis_plan: list[str] = field(default_factory=list)
    reasoning_output: dict[str, Any] = field(default_factory=dict)
    planner_response: Optional[Any] = None  # Stores the LLMResponse[FleetQueryPlan]
    reasoner_response: Optional[Any] = None # Stores the LLMResponse[FleetReasoningOutput]
    vehicle_id: Optional[str] = None  # Optional vehicle ID passed down to tools (None for fleet-level mode)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def add_tool_output(self, tool_name: str, output: Any) -> None:
        """Store the output produced by a completed tool call."""
        self.tool_outputs[tool_name] = output

    def add_history(self, role: str, content: str) -> None:
        """Append a turn to the conversation history.

        Args:
            role:    ``'user'`` or ``'assistant'``.
            content: Message text.
        """
        self.conversation_history.append({"role": role, "content": content})

    def mark_success(self) -> None:
        """Mark this execution as fully successful."""
        self.execution_status = "success"

    def mark_partial(self) -> None:
        """Mark this execution as partial (some tools failed)."""
        self.execution_status = "partial"

    def mark_error(self) -> None:
        """Mark this execution as failed."""
        self.execution_status = "error"
