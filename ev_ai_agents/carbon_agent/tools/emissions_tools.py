import pandas as pd
from langchain.tools import tool
from ev_ai_agents.carbon_agent.utils.data_loader import load_green_logistics, load_emission_factors

class ToolError(Exception):
    """Custom exception for emissions tools."""
    pass

@tool
def calculate_emissions_reduction(diesel_usage: float, ev_usage: float) -> dict:
    """Calculate CO2 savings and reduction percentage when replacing diesel with EV usage.
    
    Args:
        diesel_usage (float): Diesel fuel consumption in liters (or diesel emissions in kg CO2).
        ev_usage (float): EV electricity consumption in kWh (or EV emissions in kg CO2).
        
    Returns:
        dict: A dictionary containing diesel_emissions, ev_emissions, co2_saved, and reduction_percentage.
    """
    try:
        if diesel_usage < 0 or ev_usage < 0:
            raise ToolError("Usage values cannot be negative.")
        
        # 1 liter of diesel = 2.68 kg CO2
        # 1 kWh of EV electricity = 0.38 kg CO2 (grid average)
        diesel_emissions = round(diesel_usage * 2.68, 2)
        ev_emissions = round(ev_usage * 0.38, 2)
        
        co2_saved = round(diesel_emissions - ev_emissions, 2)
        
        if diesel_emissions > 0:
            reduction_percentage = round((co2_saved / diesel_emissions) * 100.0, 2)
        else:
            reduction_percentage = 0.0
            
        return {
            "diesel_emissions": diesel_emissions,
            "ev_emissions": ev_emissions,
            "co2_saved": co2_saved,
            "reduction_percentage": min(max(reduction_percentage, 0.0), 100.0)
        }
    except Exception as e:
        if isinstance(e, ToolError):
            raise
        raise ToolError(f"Error calculating emissions reduction: {str(e)}")

@tool
def track_scope_emissions() -> dict:
    """Calculate aggregated Scope 1 (direct fuel) and Scope 3 (indirect + supply chain) emissions.
    
    Uses:
        - Green Logistics Dataset (for Scope 1 and fleet Scope 3)
        - Supply Chain GHG Dataset (for materials Scope 3)
        
    Returns:
        dict: A dictionary containing scope1_emissions, scope3_emissions, and total_emissions in metric tons.
    """
    try:
        df_logistics = load_green_logistics()
        df_factors = load_emission_factors()
        
        # Scope 1: Direct fuel combustion from diesel vehicles (liters * 2.68) * annual_trips.
        # Convert kg to Metric Tons (MT) by dividing by 1000
        scope1_kg = (df_logistics['fuel_consumed_liters'] * 2.68 * df_logistics['annual_trips']).sum()
        scope1_mt = round(scope1_kg / 1000.0, 2)
        
        # Scope 3 (Indirect fleet + Supply chain):
        # 1. Indirect emissions from EV charging (kwh * 0.38) * annual_trips.
        ev_charging_kg = (df_logistics['electricity_consumed_kwh'] * 0.38 * df_logistics['annual_trips']).sum()
        ev_charging_mt = ev_charging_kg / 1000.0
        
        # 2. Material emissions: Let's assume a baseline production weight of materials from the factor table
        # We can simulate/calculate Scope 3 from material supply chain (e.g. producing 1000 NMC battery cells, 20000 kg steel etc.)
        # Let's map materials in supply_chain_emission_factors.csv to typical fleet supply chain weights:
        material_weights = {
            "Lithium Carbonate": 5000, # kg
            "Cobalt Sulfate": 2000, # kg
            "Nickel Sulfate": 4000, # kg
            "Synthetic Graphite": 6000, # kg
            "LFP Battery Cell": 150, # units
            "NMC Battery Cell": 350, # units
            "Steel Sheets": 80000, # kg
            "Aluminum Ingots": 30000, # kg
            "Copper Wiring": 12000, # kg
        }
        
        material_emissions_kg = 0.0
        for _, row in df_factors.iterrows():
            mat_name = row['material']
            factor = row['scope3_emission_factor_kg_co2_per_kg']
            if mat_name in material_weights:
                material_emissions_kg += factor * material_weights[mat_name]
                
        material_emissions_mt = material_emissions_kg / 1000.0
        
        # Total Scope 3
        scope3_mt = round(ev_charging_mt + material_emissions_mt, 2)
        total_mt = round(scope1_mt + scope3_mt, 2)
        
        return {
            "scope1_emissions_mt": scope1_mt,
            "scope3_emissions_mt": scope3_mt,
            "total_emissions_mt": total_mt,
            "top_routes": df_logistics.head(10).to_dict(orient='records'),
            "emission_factors": df_factors.to_dict(orient='records')
        }
    except Exception as e:
        raise ToolError(f"Error tracking scope emissions: {str(e)}")
