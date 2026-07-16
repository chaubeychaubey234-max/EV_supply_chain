from typing import TypedDict, List, Optional, Any

class SupplyChainState(TypedDict):
    # Primary user input
    query: str
    
    # Entity parameters extracted from query
    supplier_id: Optional[str]
    batch_id: Optional[str]
    material: Optional[str]
    country: Optional[str]
    
    # Analysis outputs from specialist agents
    supplier_analysis: Optional[dict]
    traceability_analysis: Optional[dict]
    risk_analysis: Optional[dict]
    quality_analysis: Optional[dict]
    
    # Combined final results
    unified_report: Optional[str]
    
    # Logs and messaging history
    messages: List[dict]
