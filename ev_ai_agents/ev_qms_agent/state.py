from typing import TypedDict, List, Optional

class QMSState(TypedDict, total=False):
    batch_id: Optional[str]
    # ML Features
    ambient_temp_c: Optional[float]
    anode_overhang_mm: Optional[float]
    electrolyte_volume_ml: Optional[float]
    internal_resistance_mohm: Optional[float]
    capacity_mah: Optional[float]
    retention_50cycle_pct: Optional[float]
    
    # Outputs
    material_data: dict
    process_data: dict
    inspection_data: dict
    quality_drift_analysis: str
    root_cause_analysis: str
    alerts: List[str]
    messages: List[str]

    # Optional internal orchestration fields
    user_query: Optional[str]
    detected_intent: Optional[str]
    analysis_mode: Optional[str]
    tool_outputs: Optional[dict]
    confidence: Optional[float]
    analysis_plan: Optional[List[str]]
    reasoning_output: Optional[dict]
