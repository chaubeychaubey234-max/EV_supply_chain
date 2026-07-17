from typing import TypedDict, List, Optional

class APMState(TypedDict, total=False):
    ev_id: Optional[str]
    # Live prediction inputs
    avg_temperature_c: Optional[float]
    fast_charge_ratio_pct: Optional[float]
    deep_discharge_cycles: Optional[int]
    avg_charge_duration_hours: Optional[float]
    max_temperature_c: Optional[float]
    
    # Outputs
    telemetry_data: dict
    battery_analysis: dict
    safety_analysis: dict
    recommendations: List[str]
    maintenance_triggers: List[str]
    messages: List[str]
