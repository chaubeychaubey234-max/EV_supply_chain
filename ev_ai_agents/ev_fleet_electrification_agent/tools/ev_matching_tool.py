import pandas as pd
import os
from langchain_core.tools import tool

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOGUE_PATH = os.path.join(_BASE_DIR, "..", "..", "datasets", "ev_catalogue.csv")

def _recommend_ev_replacement(daily_distance_km: float, available_charging_window_hours: float, 
                              payload_kg: float) -> dict:
    """Recommend the most suitable EV replacement from the catalogue."""
    if not os.path.exists(CATALOGUE_PATH):
        return {"error": "EV catalogue dataset not found."}
        
    df = pd.read_csv(CATALOGUE_PATH)
    
    best_match = None
    best_score = -1.0
    
    for _, row in df.iterrows():
        # Score calculation
        score = 0.0
        
        # 1. Range check
        range_km = float(row['Range_km'])
        if range_km >= daily_distance_km * 1.2:
            score += 0.4
        elif range_km >= daily_distance_km:
            score += 0.2
            
        # 2. Payload check
        ev_payload = float(row['Payload_kg'])
        if ev_payload >= payload_kg:
            score += 0.4
            
        # 3. Charging check
        charge_time = float(row['Charge_Time_hours'])
        if available_charging_window_hours >= charge_time:
            score += 0.2
            
        if score > best_score:
            best_score = score
            best_match = row
            
    if best_match is None or best_score == 0.0:
        return {"error": "No compatible EV found in the catalogue."}
        
    reason = (f"Selected {best_match['EV_Model']} as it provides {best_match['Range_km']}km range "
              f"(covering {daily_distance_km}km requirement) and supports {best_match['Payload_kg']}kg payload "
              f"which meets the {payload_kg}kg requirement. It charges in {best_match['Charge_Time_hours']}h.")
              
    return {
        "recommended_ev": str(best_match['EV_Model']),
        "battery_capacity_kwh": float(best_match['Battery_Capacity_kWh']),
        "estimated_range_km": float(best_match['Range_km']),
        "charging_time_hours": float(best_match['Charge_Time_hours']),
        "payload_capacity_kg": float(best_match['Payload_kg']),
        "purchase_price_usd": float(best_match['Price_USD']),
        "compatibility_score": round(best_score, 2),
        "reason": reason
    }

@tool
def recommend_ev_replacement(daily_distance_km: float, available_charging_window_hours: float, payload_kg: float) -> dict:
    """Recommend the most suitable EV replacement for a fleet vehicle based on operational constraints.
    
    Args:
        daily_distance_km: The maximum daily distance the vehicle drives in km.
        available_charging_window_hours: Hours available for continuous charging (e.g., overnight).
        payload_kg: Required cargo payload capacity in kg.
    """
    return _recommend_ev_replacement(daily_distance_km, available_charging_window_hours, payload_kg)
