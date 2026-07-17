import random
from langchain.tools import tool
try:
    from carbon_tracker_tools import ToolError
except ImportError:
    from . import ToolError

@tool
def analyze_route_emissions(route_data: dict) -> dict:
    """Calculate emissions for a given route based on distance and fuel type.
    
    Args:
        route_data (dict): Dictionary containing route details:
                           - 'distance_km' (float)
                           - 'fuel_type' (str): e.g., 'diesel', 'electric', 'petrol'
                           - 'efficiency' (float, optional): fuel or energy consumption rate.
                           
    Returns:
        dict: A dictionary containing emissions_per_route and emissions_per_km.
    """
    try:
        if not isinstance(route_data, dict):
            raise ToolError("route_data must be a dictionary.")
            
        distance = route_data.get("distance_km")
        if distance is None:
            raise ToolError("route_data must contain 'distance_km'.")
        if distance < 0:
            raise ToolError("distance_km cannot be negative.")
            
        fuel_type = str(route_data.get("fuel_type", "diesel")).lower()
        
        # Establish base emission factors (kg CO2 per km) based on fuel type
        if fuel_type == "diesel":
            # standard diesel truck/van emits ~0.8 - 1.2 kg CO2 per km
            factor = 0.95 + random.uniform(-0.1, 0.1)
        elif fuel_type in ["electric", "ev"]:
            # indirect emissions
            factor = 0.15 + random.uniform(-0.05, 0.05)
        elif fuel_type in ["petrol", "gasoline"]:
            factor = 0.85 + random.uniform(-0.1, 0.1)
        else:
            # default factor
            factor = 0.5 + random.uniform(-0.1, 0.1)
            
        emissions_per_km = factor
        emissions_per_route = emissions_per_km * distance
        
        return {
            "emissions_per_route": round(emissions_per_route, 2),
            "emissions_per_km": round(emissions_per_km, 4)
        }
    except Exception as e:
        if isinstance(e, ToolError):
            raise
        raise ToolError(f"Error analyzing route emissions: {str(e)}")

@tool
def identify_high_impact_routes(routes: list) -> dict:
    """Identify routes with highest emissions and best EV conversion potential.
    
    Args:
        routes (list): A list of dictionaries, where each dictionary represents a route:
                      - 'route_id' (str)
                      - 'distance_km' (float)
                      - 'annual_trips' (int)
                      - 'fuel_type' (str)
                      
    Returns:
        dict: A dictionary containing priority_routes and impact_score.
    """
    try:
        if not isinstance(routes, list):
            raise ToolError("routes must be a list of dictionaries.")
            
        priority_routes = []
        
        for route in routes:
            if not isinstance(route, dict):
                raise ToolError("Each route in the routes list must be a dictionary.")
            
            route_id = route.get("route_id")
            distance = route.get("distance_km")
            trips = route.get("annual_trips", 1)
            fuel_type = str(route.get("fuel_type", "diesel")).lower()
            
            if route_id is None or distance is None:
                raise ToolError("Each route must contain 'route_id' and 'distance_km'.")
            if distance < 0 or trips < 0:
                raise ToolError("Distance and trips cannot be negative.")
                
            # Only prioritize diesel or petrol routes for EV conversion
            is_ev_candidate = fuel_type in ["diesel", "petrol", "gasoline"]
            
            # Impact score calculation based on distance, frequency, and fuel type
            if is_ev_candidate:
                # scale of 0 to 100
                base_score = min(100.0, (distance * trips) / 500.0)
                impact_score = round(base_score * (0.8 + random.uniform(0.0, 0.4)), 2)
            else:
                impact_score = 0.0
                
            priority_routes.append({
                "route_id": route_id,
                "impact_score": impact_score,
                "conversion_candidate": is_ev_candidate
            })
            
        # Sort priority routes descending by impact score
        priority_routes.sort(key=lambda x: x["impact_score"], reverse=True)
        
        # Calculate overall impact score as the average of top candidates
        top_scores = [r["impact_score"] for r in priority_routes[:3] if r["impact_score"] > 0]
        overall_impact_score = round(sum(top_scores) / len(top_scores), 2) if top_scores else 0.0
        
        return {
            "priority_routes": priority_routes,
            "impact_score": overall_impact_score
        }
    except Exception as e:
        if isinstance(e, ToolError):
            raise
        raise ToolError(f"Error identifying high impact routes: {str(e)}")
