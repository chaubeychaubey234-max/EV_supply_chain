from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional

@dataclass
class AgentState:
    """Dataclass storing the state of a single maintenance agent invocation."""
    user_query: str = ""
    selected_tools: List[str] = field(default_factory=list)
    tool_outputs: Dict[str, Any] = field(default_factory=dict)
    final_response: Dict[str, Any] = field(default_factory=dict)
    execution_status: str = "pending"
    conversation_history: List[Dict[str, str]] = field(default_factory=list)

    # Orchestration fields for LangGraph compatibility
    detected_intent: str = ""
    analysis_mode: str = ""
    confidence: float = 1.0
    analysis_plan: List[str] = field(default_factory=list)
    reasoning_output: Dict[str, Any] = field(default_factory=dict)
    planner_response: Optional[Any] = None  # Stores the LLMResponse[MaintenanceQueryPlan]
    reasoner_response: Optional[Any] = None # Stores the LLMResponse[MaintenanceReasoningOutput]
    vehicle_record: Optional[Dict[str, Any]] = None  # Preserve vehicle_record passed down to tools