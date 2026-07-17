import random
from langchain.tools import tool
try:
    from carbon_tracker_tools import ToolError
except ImportError:
    from . import ToolError

@tool
def calculate_emissions_reduction(diesel_usage: float, ev_usage: float) -> dict:
    """Calculate CO2 savings when replacing diesel with EV usage.
    
    Args:
        diesel_usage (float): Diesel fuel consumption in liters.
        ev_usage (float): EV electricity consumption in kWh.
        
    Returns:
        dict: A dictionary containing co2_saved and reduction_percentage.
    """
    try:
        if diesel_usage < 0 or ev_usage < 0:
            raise ToolError("Usage values cannot be negative.")
        
        # Simulating industrial-scale emissions logic
        # 1 liter of diesel produces ~2.68 kg CO2
        # 1 kWh of electricity produces ~0.4 kg CO2 (depends on grid, simulate some variability)
        diesel_emissions = diesel_usage * 2.68
        ev_emissions = ev_usage * (0.35 + random.uniform(-0.05, 0.05))
        
        co2_saved = diesel_emissions - ev_emissions
        
        if diesel_emissions > 0:
            reduction_percentage = (co2_saved / diesel_emissions) * 100.0
        else:
            reduction_percentage = 0.0
            
        return {
            "co2_saved": round(co2_saved, 2),
            "reduction_percentage": round(min(max(reduction_percentage, 0.0), 100.0), 2)
        }
    except Exception as e:
        if isinstance(e, ToolError):
            raise
        raise ToolError(f"Error calculating emissions reduction: {str(e)}")

@tool
def track_scope_emissions(fleet_data: list) -> dict:
    """Aggregate Scope 1 (fuel) and Scope 3 (indirect) emissions.
    
    Args:
        fleet_data (list): A list of dictionaries representing fleet vehicles.
                           Each dict should have keys like 'fuel_consumed_liters' and 'electricity_consumed_kwh'.
                           
    Returns:
        dict: A dictionary containing scope1_emissions and scope3_emissions.
    """
    try:
        if not isinstance(fleet_data, list):
            raise ToolError("fleet_data must be a list of dictionaries.")
            
        scope1_total = 0.0
        scope3_total = 0.0
        
        for vehicle in fleet_data:
            if not isinstance(vehicle, dict):
                raise ToolError("Each item in fleet_data must be a dictionary.")
            
            # Scope 1: Direct emissions from burning fuel
            fuel = vehicle.get("fuel_consumed_liters", 0.0)
            if fuel < 0:
                raise ToolError("Fuel consumed cannot be negative.")
            scope1_total += fuel * 2.68
            
            # Scope 3: Indirect emissions (e.g., electricity production for EV charging, supply chain, etc.)
            electricity = vehicle.get("electricity_consumed_kwh", 0.0)
            if electricity < 0:
                raise ToolError("Electricity consumed cannot be negative.")
            # Electricity indirect emissions + lifecycle emissions
            scope3_total += electricity * 0.40
            
            # Add some simulated variance for other scope 3 elements (e.g. supply chain transport)
            scope3_total += random.uniform(10.0, 50.0)
            
        return {
            "scope1_emissions": round(scope1_total, 2),
            "scope3_emissions": round(scope3_total, 2)
        }
    except Exception as e:
        if isinstance(e, ToolError):
            raise
        raise ToolError(f"Error tracking scope emissions: {str(e)}")
