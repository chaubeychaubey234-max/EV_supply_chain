from langchain_core.tools import tool

def _calculate_readiness_score(daily_distance_km: float, available_charging_window_hours: float, avg_idle_minutes: float, 
                              stops_per_day: int, route_type: str, route_consistency_score: float, vehicle_age_years: float, 
                              fuel_efficiency_kmpl: float, operating_hours_per_day: float, utilization_rate: float, payload_kg: float) -> dict:
    """Calculates a deterministic readiness score based on operational profile."""
    
    score = 0.0
    reasons = []
    
    # 1. Range Viability (Weight: 35%)
    # Assuming average heavy EV range is ~300km
    max_range = 300.0
    range_ratio = daily_distance_km / max_range
    if range_ratio <= 0.6:
        score += 35
        reasons.append(f"Daily distance ({daily_distance_km}km) is well within average EV range, providing great buffer.")
    elif range_ratio <= 0.85:
        score += 25
        reasons.append(f"Daily distance ({daily_distance_km}km) is viable but requires consistent overnight charging.")
    elif range_ratio <= 1.0:
        score += 15
        reasons.append(f"Daily distance ({daily_distance_km}km) is at the boundary of EV range; opportunity charging needed.")
    else:
        score += 0
        reasons.append(f"Daily distance ({daily_distance_km}km) exceeds standard EV range; mid-route fast charging mandatory.")
        
    # 2. Charging Window (Weight: 25%)
    if available_charging_window_hours >= 10:
        score += 25
        reasons.append(f"Excellent charging window ({available_charging_window_hours}h).")
    elif available_charging_window_hours >= 6:
        score += 15
        reasons.append(f"Adequate charging window ({available_charging_window_hours}h) for Level 2 or DC Fast charging.")
    else:
        score += 5
        reasons.append(f"Tight charging window ({available_charging_window_hours}h) requires high-power DC infrastructure.")
        
    # 3. Predictability & Route (Weight: 20%)
    if route_consistency_score > 0.8:
        score += 20
        reasons.append(f"High route consistency ({route_consistency_score}) makes energy planning highly predictable.")
    elif route_consistency_score > 0.5:
        score += 10
        reasons.append(f"Moderate route consistency ({route_consistency_score}).")
    else:
        score += 0
        reasons.append(f"Low route consistency ({route_consistency_score}) increases range anxiety risk.")
        
    # 4. Financial & Urgency (Weight: 20%)
    # Older vehicles with worse fuel economy give better ROI
    if vehicle_age_years >= 5:
        score += 10
        reasons.append(f"Vehicle age ({vehicle_age_years} yrs) makes it a prime candidate for replacement.")
    else:
        score += 5
        
    if fuel_efficiency_kmpl < 5.0:
        score += 10
        reasons.append(f"Low fuel efficiency ({fuel_efficiency_kmpl} kmpl) amplifies electrification ROI.")
    else:
        score += 5

    total_score = min(int(score), 100)
    
    if total_score >= 85:
        classification = "Highly Ready"
        confidence = "high"
    elif total_score >= 70:
        classification = "Ready"
        confidence = "high"
    elif total_score >= 50:
        classification = "Conditionally Ready"
        confidence = "medium"
    else:
        classification = "Not Ready"
        confidence = "low"
        
    return {
        "readiness_score": total_score,
        "confidence": confidence,
        "classification": classification,
        "reason": " ".join(reasons)
    }

@tool
def calculate_readiness_score(daily_distance_km: float, available_charging_window_hours: float, avg_idle_minutes: float, 
                              stops_per_day: int, route_type: str, route_consistency_score: float, vehicle_age_years: float, 
                              fuel_efficiency_kmpl: float, operating_hours_per_day: float, utilization_rate: float, payload_kg: float) -> dict:
    """Estimate the EV electrification readiness of a fleet vehicle based on operational metrics.
    
    Args:
        daily_distance_km: Daily distance in km.
        available_charging_window_hours: Hours available for charging.
        avg_idle_minutes: Average idle time.
        stops_per_day: Number of stops.
        route_type: Urban, Suburban, Highway.
        route_consistency_score: Predictability from 0 to 1.
        vehicle_age_years: Age of ICE vehicle.
        fuel_efficiency_kmpl: Current efficiency.
        operating_hours_per_day: Hours in operation.
        utilization_rate: Fleet util rate (0 to 1).
        payload_kg: Cargo payload in kg.
    """
    return _calculate_readiness_score(
        daily_distance_km, available_charging_window_hours, avg_idle_minutes, stops_per_day,
        route_type, route_consistency_score, vehicle_age_years, fuel_efficiency_kmpl,
        operating_hours_per_day, utilization_rate, payload_kg
    )
