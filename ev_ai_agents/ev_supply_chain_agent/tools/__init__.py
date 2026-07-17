"""
EV Supply Chain Risk & Traceability Tools
==========================================
LangChain-compatible tools for simulating EV battery supply chain intelligence.
Covers supplier profiling, risk assessment, and material traceability.
"""


class ToolError(Exception):
    """Custom exception for tool-level failures in the EV supply chain toolset."""
    pass


from .supplier_tools import (
    get_supplier_profile,
    get_supplier_tier,
    get_supplier_geography,
)

from .risk_tools import (
    calculate_supplier_risk_score,
    detect_geopolitical_risk,
    detect_supplier_concentration,
    detect_quality_deviation,
)

from .traceability_tools import (
    trace_material_batch,
    map_cell_to_pack,
    map_pack_to_vehicle,
    verify_traceability_completeness,
)

__all__ = [
    "ToolError",
    "get_supplier_profile",
    "get_supplier_tier",
    "get_supplier_geography",
    "calculate_supplier_risk_score",
    "detect_geopolitical_risk",
    "detect_supplier_concentration",
    "detect_quality_deviation",
    "trace_material_batch",
    "map_cell_to_pack",
    "map_pack_to_vehicle",
    "verify_traceability_completeness",
]
