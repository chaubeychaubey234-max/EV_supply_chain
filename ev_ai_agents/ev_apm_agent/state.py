from typing import TypedDict, List, Optional

class APMState(TypedDict):
    ev_id: str
    telemetry_data: dict
    battery_analysis: dict
    safety_analysis: dict
    recommendations: List[str]
    maintenance_triggers: List[str]
    messages: List[str]
