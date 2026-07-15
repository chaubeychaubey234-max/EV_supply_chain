class ToolError(Exception):
    """Custom exception class for net zero and carbon intelligence tools."""
    pass

from .emissions_tools import calculate_emissions_reduction, track_scope_emissions
from .route_tools import analyze_route_emissions, identify_high_impact_routes
from .progress_tools import track_net_zero_progress

__all__ = [
    "ToolError",
    "calculate_emissions_reduction",
    "track_scope_emissions",
    "analyze_route_emissions",
    "identify_high_impact_routes",
    "track_net_zero_progress",
]
