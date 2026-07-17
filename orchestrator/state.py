import operator
from typing import TypedDict, Dict, Any, Optional, List, Annotated

class OrchestratorState(TypedDict):
    """Global state for the LangGraph Orchestrator."""
    user_query: str
    
    # We use Annotated and an operator to merge dictionary responses from parallel branches
    agent_responses: Annotated[Dict[str, Any], operator.ior]
    
    # We track which agents we want to run in parallel
    next_agents: List[str]
    
    final_synthesis: Optional[str]
