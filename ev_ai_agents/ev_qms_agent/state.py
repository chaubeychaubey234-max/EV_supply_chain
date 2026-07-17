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
