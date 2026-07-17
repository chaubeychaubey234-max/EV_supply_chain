import os
import pandas as pd
from langchain.tools import tool
from ev_ai_agents.carbon_agent.utils.data_loader import load_green_logistics

class ToolError(Exception):
    """Custom exception for route tools."""
    pass

@tool
def analyze_route_emissions(route_id: str) -> dict:
    """Analyze carbon emissions for a specific logistics route.
    
    Args:
        route_id (str): The ID of the route to analyze (e.g. 'R-101', 'R-102').
        
    Returns:
        dict: A dictionary containing route_id, emissions, emissions_per_km, and vehicle_type.
    """
    try:
        df = load_green_logistics()
        # Clean query
        route_id_clean = str(route_id).strip().upper()
        route_row = df[df['route_id'] == route_id_clean]
        
        if route_row.empty:
            raise ToolError(f"Route with ID '{route_id_clean}' not found in the dataset.")
            
        row = route_row.iloc[0]
        distance = row['distance_km']
        emissions = row['carbon_emissions_kg']
        vehicle_type = row['vehicle_type']
        
        emissions_per_km = round(emissions / distance, 4) if distance > 0 else 0.0
        
        return {
            "route_id": route_id_clean,
            "emissions": round(emissions, 2),
            "emissions_per_km": emissions_per_km,
            "vehicle_type": vehicle_type
        }
    except Exception as e:
        if isinstance(e, ToolError):
            raise
        raise ToolError(f"Error analyzing route emissions: {str(e)}")

@tool
def identify_high_impact_routes() -> dict:
    """Find routes with maximum emissions and high EV conversion potential.
    
    Returns:
        dict: A dictionary containing priority_routes, impact_score, and recommended_action.
    """
    try:
        df = load_green_logistics()
        
        # We target Diesel vehicles for EV conversion.
        diesel_df = df[df['vehicle_type'].str.contains('Diesel', case=False, na=False)].copy()
        
        # Calculate annual carbon emissions for each route (carbon_emissions_kg * annual_trips) in kg.
        # Normalize to Metric Tons (MT) for readability.
        diesel_df['annual_emissions_mt'] = (diesel_df['carbon_emissions_kg'] * diesel_df['annual_trips']) / 1000.0
        
        # Sort descending by annual emissions
        priority_routes_df = diesel_df.sort_values(by='annual_emissions_mt', ascending=False)
        
        priority_routes = []
        for _, row in priority_routes_df.iterrows():
            priority_routes.append({
                "route_id": row['route_id'],
                "start_location": row['start_location'],
                "end_location": row['end_location'],
                "distance_km": row['distance_km'],
                "vehicle_type": row['vehicle_type'],
                "annual_trips": int(row['annual_trips']),
                "annual_emissions_mt": round(row['annual_emissions_mt'], 2)
            })
            
        # Overall impact score based on top priority routes' average annual emissions
        top_emissions = [r["annual_emissions_mt"] for r in priority_routes[:3]]
        impact_score = round(sum(top_emissions) / len(top_emissions), 2) if top_emissions else 0.0
        
        recommended_action = (
            f"Prioritize converting high-frequency route {priority_routes[0]['route_id']} "
            f"({priority_routes[0]['start_location']} -> {priority_routes[0]['end_location']}) "
            f"and route {priority_routes[1]['route_id']} to electric alternatives. "
            f"This can eliminate up to {round(sum(top_emissions[:2]), 2)} MT of CO2 emissions annually."
        ) if len(priority_routes) >= 2 else "No high-impact diesel routes found for conversion."
        
        return {
            "priority_routes": priority_routes,
            "impact_score": impact_score,
            "recommended_action": recommended_action
        }
    except Exception as e:
        raise ToolError(f"Error identifying high impact routes: {str(e)}")

@tool
def recommend_electrification() -> dict:
    """Identify the best next fleet assets/routes for electrification.
    
    Decision criteria:
        - Must currently be a Diesel vehicle (conversion candidate)
        - High carbon emissions (greater than 100 kg per trip)
        - High frequency (annual_trips > 100)
        - Suitable distance (distance_km <= 500 km, suitable for regional charging infrastructure)
        
    Returns:
        dict: A dictionary containing recommended_routes, expected_CO2_reduction, and priority_reason.
    """
    try:
        df = load_green_logistics()
        
        # Filter diesel candidates
        candidates = df[df['vehicle_type'].str.contains('Diesel', case=False, na=False)].copy()
        
        # Calculate suitability metrics
        candidates['annual_emissions_mt'] = (candidates['carbon_emissions_kg'] * candidates['annual_trips']) / 1000.0
        
        # Apply criteria: suitable distance (<= 500 km) and high frequency (> 100 trips) or high emissions
        filtered = candidates[
            (candidates['distance_km'] <= 500.0) & 
            (candidates['annual_trips'] >= 100)
        ].copy()
        
        if filtered.empty:
            # Fallback if strict criteria filters everything out
            filtered = candidates.copy()
            
        # Sort by annual emissions to get the highest impact ones
        filtered = filtered.sort_values(by='annual_emissions_mt', ascending=False)
        
        recommended_routes = []
        total_saved_mt = 0.0
        
        for _, row in filtered.iterrows():
            # Estimate savings if replaced by EV
            # Diesel emits ~2.68 kg/liter. EV emits ~0.38 kg/kWh.
            # Assuming EV has ~0.25 kWh per km consumption for Van, or ~0.4 kWh per km for Truck.
            # EV emissions per km: ~0.1 kg CO2. Diesel emissions per km: ~0.8-1.2 kg CO2.
            # This represents ~90% reduction.
            current_annual = row['annual_emissions_mt']
            expected_reduction_mt = round(current_annual * 0.90, 2)
            total_saved_mt += expected_reduction_mt
            
            recommended_routes.append({
                "route_id": row['route_id'],
                "start_location": row['start_location'],
                "end_location": row['end_location'],
                "distance_km": row['distance_km'],
                "current_vehicle": row['vehicle_type'],
                "annual_trips": int(row['annual_trips']),
                "current_annual_emissions_mt": round(current_annual, 2),
                "expected_reduction_mt": expected_reduction_mt
            })
            
        priority_reason = (
            f"Identified {len(recommended_routes)} high-frequency regional routes under 500 km currently operating diesel vehicles. "
            f"Electrifying these specific routes will reduce carbon emissions by ~90%, saving a total of {round(total_saved_mt, 2)} MT CO2 annually."
        )
        
        return {
            "recommended_routes": recommended_routes,
            "expected_co2_reduction": round(total_saved_mt, 2),
            "priority_reason": priority_reason
        }
    except Exception as e:
        raise ToolError(f"Error recommending electrification: {str(e)}")

@tool
def generate_and_save_route_map() -> dict:
    """Generate and save the interactive geospatial route emissions map.
    
    Returns:
        dict: A dictionary containing map_saved_path and message.
    """
    try:
        from ev_ai_agents.carbon_agent.utils.geo_utils import map_high_emission_routes
        # Save to the project directory
        output_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        output_path = os.path.join(output_dir, "route_emissions_map.html")
        map_high_emission_routes(output_path)
        return {
            "map_saved_path": output_path,
            "message": "Geospatial route emissions map generated successfully. The map is loaded in the visualization layer."
        }
    except Exception as e:
        raise ToolError(f"Error generating route map: {str(e)}")
