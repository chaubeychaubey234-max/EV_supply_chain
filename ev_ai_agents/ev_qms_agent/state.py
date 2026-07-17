from typing import TypedDict, List, Optional

class QMSState(TypedDict):
    batch_id: str
    material_data: dict
    process_data: dict
    inspection_data: dict
    quality_drift_analysis: str
    root_cause_analysis: str
    alerts: List[str]
    messages: List[str]
