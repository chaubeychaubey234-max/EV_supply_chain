from typing import TypedDict, Any

class SupplyChainState(TypedDict, total=False):
    user_query: str
    detected_intent: str
    analysis_mode: str
    analysis_plan: list[str]
    confidence: float
    supplier_id: str
    batch_id: str
    country: str
    tool_outputs: dict[str, Any]
    reasoning_output: dict[str, Any]
