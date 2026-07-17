import random
from langchain.tools import tool
try:
    from carbon_tracker_tools import ToolError
except ImportError:
    from . import ToolError

@tool
def track_net_zero_progress(current_emissions: float, target_emissions: float) -> dict:
    """Track progress toward Net Zero goals.
    
    Args:
        current_emissions (float): The current annual carbon emissions in metric tons.
        target_emissions (float): The target annual carbon emissions in metric tons.
        
    Returns:
        dict: A dictionary containing progress_percentage, emissions_gap, and status.
    """
    try:
        if current_emissions < 0 or target_emissions < 0:
            raise ToolError("Emissions values cannot be negative.")
            
        emissions_gap = current_emissions - target_emissions
        
        # Progress percentage logic (assuming target is lower than initial, but generally tracking reduction)
        # If current is already at or below target, progress is 100%
        if current_emissions <= target_emissions:
            progress_percentage = 100.0
            status = "On Track"
        else:
            # Let's say baseline was higher, say target_emissions * 2 or dynamic baseline
            # Progress can be simulated/calculated
            # A simple metric: progress_percentage = target / current * 100
            # Or we can simulate progress based on gap
            progress_percentage = round(max(0.0, 100.0 - (emissions_gap / max(1.0, current_emissions) * 100.0)), 2)
            
            # Determine status based on gap and simulated run-rate
            # Status: On Track / At Risk / Behind
            gap_ratio = emissions_gap / max(1.0, target_emissions)
            
            # Add some randomness to simulate real intelligence assessment
            random_factor = random.uniform(-0.05, 0.05)
            effective_gap = gap_ratio + random_factor
            
            if effective_gap <= 0.1:
                status = "On Track"
            elif effective_gap <= 0.3:
                status = "At Risk"
            else:
                status = "Behind"
                
        return {
            "progress_percentage": round(progress_percentage, 2),
            "emissions_gap": round(emissions_gap, 2),
            "status": status
        }
    except Exception as e:
        if isinstance(e, ToolError):
            raise
        raise ToolError(f"Error tracking net zero progress: {str(e)}")
