from typing import TypedDict, Any, List

class SupplyChainState(TypedDict, total=False):
    user_query: str
    detected_intent: str
    analysis_mode: str
    analysis_plan: List[str]
    confidence: float
    supplier_id: str
    batch_id: str
    country: str
    tool_outputs: dict[str, Any]
    reasoning_output: dict[str, Any]
    unified_report: str
    status: str
    summary: str
    supplier_details: dict[str, Any]
    risk_rating: str
    recommendations: List[str]
    next_steps: List[str]
