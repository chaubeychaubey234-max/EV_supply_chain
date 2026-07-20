import pandas as pd
import os
from langchain_core.tools import tool

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FLEET_OPS_PATH = os.path.join(_BASE_DIR, "..", "datasets", "fleet_operations_clean.csv")
REGISTRY_PATH = os.path.join(_BASE_DIR, "..", "..", "datasets", "fleet_registry.csv")

@tool
def fetch_vehicle_data(vehicle_id: str) -> dict:
    """Retrieve complete asset and telemetry data for a fleet vehicle from the registry."""
    # 1. Try loading from fleet_operations_clean.csv
    if os.path.exists(FLEET_OPS_PATH):
        try:
            df = pd.read_csv(FLEET_OPS_PATH)
            row = df[df['vehicle_id'] == vehicle_id]
            if not row.empty:
                return row.iloc[0].to_dict()
        except Exception:
            pass
            
    # 2. Try loading from fleet_registry.csv
    if os.path.exists(REGISTRY_PATH):
        try:
            df = pd.read_csv(REGISTRY_PATH)
            row = df[df['vehicle_id'] == vehicle_id]
            if not row.empty:
                return row.iloc[0].to_dict()
        except Exception:
            pass
            
    return {"error": f"No vehicle found with ID {vehicle_id}."}

@tool
def analyze_fleet_csv(csv_path: str) -> dict:
    """Analyze a user-provided CSV containing fleet vehicle data and return readiness/matching scores.
    Expected columns: vehicle_id, daily_distance_km, available_charging_window_hours, 
                      avg_idle_minutes, stops_per_day, route_type, route_consistency_score, 
                      vehicle_age_years, fuel_efficiency_kmpl, operating_hours_per_day, 
                      utilization_rate, payload_kg.
                      
    Args:
        csv_path: Absolute path to the CSV file to analyze.
    """
    from .readiness_score_tool import _calculate_readiness_score
    from .ev_matching_tool import _recommend_ev_replacement
    
    if not os.path.exists(csv_path):
        return {"error": f"File not found: {csv_path}"}
        
    df = pd.read_csv(csv_path)
    results = []
    
    for _, row in df.iterrows():
        vid = row.get("vehicle_id", "Unknown")
        
        # Calculate readiness
        readiness = _calculate_readiness_score(
            daily_distance_km=float(row.get("daily_distance_km", 0)),
            available_charging_window_hours=float(row.get("available_charging_window_hours", 8)),
            avg_idle_minutes=float(row.get("avg_idle_minutes", 60)),
            stops_per_day=int(row.get("stops_per_day", 5)),
            route_type=str(row.get("route_type", "mixed")),
            route_consistency_score=float(row.get("route_consistency_score", 0.5)),
            vehicle_age_years=float(row.get("vehicle_age_years", 3)),
            fuel_efficiency_kmpl=float(row.get("fuel_efficiency_kmpl", 8.0)),
            operating_hours_per_day=float(row.get("operating_hours_per_day", 8)),
            utilization_rate=float(row.get("utilization_rate", 0.7)),
            payload_kg=float(row.get("payload_kg", 1000))
        )
        
        # Calculate match
        match = _recommend_ev_replacement(
            daily_distance_km=float(row.get("daily_distance_km", 0)),
            available_charging_window_hours=float(row.get("available_charging_window_hours", 8)),
            payload_kg=float(row.get("payload_kg", 1000))
        )
        
        results.append({
            "vehicle_id": vid,
            "readiness_score": readiness.get("readiness_score"),
            "readiness_classification": readiness.get("classification"),
            "recommended_ev": match.get("recommended_ev"),
            "compatibility_score": match.get("compatibility_score")
        })
        
    return {"fleet_analysis": results}
