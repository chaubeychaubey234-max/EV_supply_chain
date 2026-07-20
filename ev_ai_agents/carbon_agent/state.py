from typing import TypedDict, Any

class CarbonState(TypedDict, total=False):
    user_query: str
    detected_intent: str
    analysis_mode: str
    analysis_plan: list[str]
    confidence: float
    tool_outputs: dict[str, Any]
    reasoning_output: dict[str, Any]
