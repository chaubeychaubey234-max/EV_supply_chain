from dataclasses import dataclass, field
from typing import Any, List, Dict

@dataclass
class AgentState:
    """Dataclass storing the state of a single maintenance agent invocation."""
    user_query: str = ""
    selected_tools: List[str] = field(default_factory=list)
    tool_outputs: Dict[str, Any] = field(default_factory=dict)
    final_response: Dict[str, Any] = field(default_factory=dict)
    execution_status: str = "pending"
    conversation_history: List[Dict[str, str]] = field(default_factory=list)