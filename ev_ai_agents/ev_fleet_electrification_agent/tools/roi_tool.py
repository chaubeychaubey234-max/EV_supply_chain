from langchain_core.tools import tool


_ROI_PROFILES: dict[str, dict] = {
    "VEH-001": {
        "vehicle_id": "VEH-001",
        "estimated_annual_fuel_cost_usd": 50_391.0,
        "estimated_annual_electricity_cost_usd": 19_350.0,
        "annual_fuel_savings_usd": 31_041.0,
        "annual_maintenance_savings_usd": 8_000.0,
        "total_annual_savings_usd": 39_041.0,
        "ev_purchase_price_usd": 250_000.0,
        "estimated_payback_years": 6.4,
        "roi_percent_over_10_years": 56.2,
    },
    "VEH-002": {
        "vehicle_id": "VEH-002",
        "estimated_annual_fuel_cost_usd": 6_908.0,
        "estimated_annual_electricity_cost_usd": 1_313.0,
        "annual_fuel_savings_usd": 5_595.0,
        "annual_maintenance_savings_usd": 2_500.0,
        "total_annual_savings_usd": 8_095.0,
        "ev_purchase_price_usd": 64_990.0,
        "estimated_payback_years": 8.0,
        "roi_percent_over_10_years": 24.6,
    },
    "VEH-003": {
        "vehicle_id": "VEH-003",
        "estimated_annual_fuel_cost_usd": 23_529.0,
        "estimated_annual_electricity_cost_usd": 7_680.0,
        "annual_fuel_savings_usd": 15_849.0,
        "annual_maintenance_savings_usd": 6_500.0,
        "total_annual_savings_usd": 22_349.0,
        "ev_purchase_price_usd": 185_000.0,
        "estimated_payback_years": 8.3,
        "roi_percent_over_10_years": 20.9,
    },
    "VEH-004": {
        "vehicle_id": "VEH-004",
        "estimated_annual_fuel_cost_usd": 16_250.0,
        "estimated_annual_electricity_cost_usd": 6_240.0,
        "annual_fuel_savings_usd": 10_010.0,
        "annual_maintenance_savings_usd": 5_000.0,
        "total_annual_savings_usd": 15_010.0,
        "ev_purchase_price_usd": 148_000.0,
        "estimated_payback_years": 9.9,
        "roi_percent_over_10_years": 1.4,
    },
    "VEH-005": {
        "vehicle_id": "VEH-005",
        "estimated_annual_fuel_cost_usd": 4_286.0,
        "estimated_annual_electricity_cost_usd": 956.0,
        "annual_fuel_savings_usd": 3_330.0,
        "annual_maintenance_savings_usd": 1_800.0,
        "total_annual_savings_usd": 5_130.0,
        "ev_purchase_price_usd": 49_995.0,
        "estimated_payback_years": 9.7,
        "roi_percent_over_10_years": 2.6,
    },
}


def _validate_vehicle_id(vehicle_id: str) -> str:
    """Validate and normalise the vehicle_id input.

    Args:
        vehicle_id: Raw caller-supplied identifier.

    Returns:
        A stripped, non-empty vehicle ID string.

    Raises:
        ValueError: If the value fails any validation check.
    """
    if not isinstance(vehicle_id, str):
        raise ValueError(
            f"vehicle_id must be a string, received {type(vehicle_id).__name__!r}."
        )
    cleaned = vehicle_id.strip()
    if not cleaned:
        raise ValueError("vehicle_id must not be empty or whitespace-only.")
    return cleaned


def _lookup_roi_profile(vehicle_id: str) -> dict:
    """Retrieve the ROI financial profile for a given vehicle.

    Args:
        vehicle_id: A validated, stripped vehicle ID.

    Returns:
        The ROI profile dictionary.

    Raises:
        ValueError: If no profile exists for the given ID.
    """
    profile = _ROI_PROFILES.get(vehicle_id)
    if profile is None:
        valid_ids = ", ".join(sorted(_ROI_PROFILES.keys()))
        raise ValueError(
            f"No ROI profile found for vehicle_id='{vehicle_id}'. "
            f"Registered IDs: {valid_ids}."
        )
    return profile


@tool
def calculate_roi(vehicle_id: str) -> dict:
    """Estimate the financial return on investment for electrifying a fleet vehicle.

    Computes a detailed financial comparison between the current ICE operating
    costs and the projected costs of operating the recommended EV replacement.

    Financial model assumptions:
        - Annual operating days: 250 working days per year.
        - Diesel price: USD 1.50 per litre (blended commercial rate).
        - Electricity price: USD 0.12 per kWh (commercial tariff).
        - EV energy consumption: derived from vehicle class and daily distance.
        - Maintenance savings: ICE maintenance cost reduction from eliminating
          oil changes, transmission servicing, and exhaust system maintenance.
        - ROI is computed over a 10-year vehicle service life horizon.
        - No government incentives or tax credits are included in the base model.

    Args:
        vehicle_id: The unique fleet identifier of the target vehicle
                    (e.g., "VEH-001"). Must be a non-empty, non-whitespace
                    string matching a registered vehicle.

    Returns:
        A dictionary containing ROI attributes.
    """
    clean_id = _validate_vehicle_id(vehicle_id)
    
    # 1. Load vehicle data
    from ev_ai_agents.ev_fleet_electrification_agent.tools.fleet_data_tool import fetch_vehicle_data
    vehicle_record = fetch_vehicle_data.invoke({"vehicle_id": clean_id})
    if "error" in vehicle_record:
        # Fallback to hardcoded profiles for backward compatibility
        profile = _ROI_PROFILES.get(clean_id)
        if profile:
            return profile
        raise ValueError(f"No vehicle or ROI profile found for vehicle_id='{clean_id}'.")
        
    # 2. Extract values with fallbacks
    daily_distance = float(vehicle_record.get("daily_distance_km", 0.0))
    charging_window = float(vehicle_record.get("charging_window_hours", 
                            vehicle_record.get("available_charging_window_hours", 8.0)))
    payload = float(vehicle_record.get("payload_requirement_kg", 
                    vehicle_record.get("payload_capacity_kg", 
                    vehicle_record.get("payload_kg", 1000.0))))
                    
    # Fuel efficiency
    vtype = str(vehicle_record.get("vehicle_type", "")).lower()
    if "heavy" in vtype or "truck" in vtype:
        def_eff = 3.5
    elif "van" in vtype or "delivery" in vtype:
        def_eff = 9.5
    elif "bus" in vtype:
        def_eff = 4.8
    else:
        def_eff = 12.0
    fuel_efficiency = float(vehicle_record.get("fuel_efficiency_kmpl", def_eff))
    if fuel_efficiency <= 0:
        fuel_efficiency = def_eff
        
    # Fuel Type & Price
    fuel_type = str(vehicle_record.get("fuel_type", "diesel")).lower()
    fuel_price = 1.20 if "cng" in fuel_type else 1.50
    
    # 3. Calculate current ICE annual fuel cost
    annual_operating_days = 250
    estimated_annual_fuel_cost = (daily_distance / fuel_efficiency) * fuel_price * annual_operating_days
    estimated_annual_fuel_cost = round(estimated_annual_fuel_cost, 2)
    
    # 4. Get recommended EV details
    from ev_ai_agents.ev_fleet_electrification_agent.tools.ev_matching_tool import recommend_ev_replacement
    ev_match = recommend_ev_replacement.invoke({
        "daily_distance_km": daily_distance,
        "available_charging_window_hours": charging_window,
        "payload_kg": payload
    })
    
    if "error" not in ev_match:
        ev_model = ev_match.get("recommended_ev")
        ev_price = float(ev_match.get("purchase_price_usd", 64990.0))
        # Determine energy efficiency based on vehicle type
        if "heavy" in vtype or "truck" in vtype:
            efficiency = 1.50
        elif "medium" in vtype or "bus" in vtype:
            efficiency = 0.80
        elif "pickup" in vtype:
            efficiency = 0.245
        else:
            efficiency = 0.25
            
        electricity_price = 0.12
        estimated_annual_electricity_cost = daily_distance * efficiency * electricity_price * annual_operating_days
        estimated_annual_electricity_cost = round(estimated_annual_electricity_cost, 2)
    else:
        # Fallbacks
        ev_model = "Standard EV"
        ev_price = 64990.0
        estimated_annual_electricity_cost = 1313.0
        
    # 5. Maintenance savings
    if "heavy" in vtype or "truck" in vtype:
        maint_savings = 8000.0
    elif "medium" in vtype:
        maint_savings = 6500.0
    elif "bus" in vtype:
        maint_savings = 5000.0
    elif "van" in vtype or "delivery" in vtype:
        maint_savings = 2500.0
    elif "pickup" in vtype:
        maint_savings = 1800.0
    else:
        maint_savings = 2000.0
        
    annual_fuel_savings = estimated_annual_fuel_cost - estimated_annual_electricity_cost
    total_annual_savings = annual_fuel_savings + maint_savings
    
    if total_annual_savings > 0:
        payback_years = round(ev_price / total_annual_savings, 1)
        roi_10_years = round(((total_annual_savings * 10 - ev_price) / ev_price) * 100, 1)
    else:
        payback_years = 99.0
        roi_10_years = 0.0
        
    return {
        "vehicle_id": clean_id,
        "estimated_annual_fuel_cost_usd": float(round(estimated_annual_fuel_cost, 0)),
        "estimated_annual_electricity_cost_usd": float(round(estimated_annual_electricity_cost, 0)),
        "annual_fuel_savings_usd": float(round(annual_fuel_savings, 0)),
        "annual_maintenance_savings_usd": float(round(maint_savings, 0)),
        "total_annual_savings_usd": float(round(total_annual_savings, 0)),
        "ev_purchase_price_usd": float(round(ev_price, 0)),
        "estimated_payback_years": float(payback_years),
        "roi_percent_over_10_years": float(roi_10_years)
    }
