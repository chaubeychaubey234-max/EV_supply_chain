from langchain_core.tools import tool


_PROCUREMENT_RECOMMENDATIONS: dict[str, dict] = {
    "VEH-001": {
        "vehicle_id": "VEH-001",
        "recommendation": "Proceed with planned EV conversion, subject to route optimisation.",
        "priority": "medium",
        "confidence": "medium",
        "recommended_purchase_window": "Q3 2025",
        "reason": (
            "VEH-001 has a moderate readiness score of 64, driven primarily by "
            "the daily distance of 430 km approaching the boundary of current "
            "heavy EV range (400 km). Electrification is financially viable with "
            "total annual savings of USD 39,041 and a payback period of 6.4 years, "
            "yielding a 56.2% ROI over 10 years — the strongest financial case in "
            "the fleet. Procurement is recommended in Q3 2025 following a route "
            "restructuring assessment to determine whether daily distance can be "
            "reduced by 8–10% through depot relocation or route splitting. "
            "Alternatively, the purchase may be deferred to Q4 2025 pending next "
            "generation eCascadia variants with extended range (500+ km) scheduled "
            "for production entry in late 2025."
        ),
    },
    "VEH-002": {
        "vehicle_id": "VEH-002",
        "recommendation": "Immediate procurement approved. Highest priority conversion.",
        "priority": "high",
        "confidence": "high",
        "recommended_purchase_window": "Q1 2025",
        "reason": (
            "VEH-002 achieves a readiness score of 91 — the second highest in "
            "the fleet — with daily distance of 175 km well within the 317 km "
            "E-Transit range. The 9-hour overnight charging window fully supports "
            "charge requirements. Annual total savings of USD 8,095 against an "
            "acquisition cost of USD 64,990 yield an 8-year payback and 24.6% "
            "ROI. Urban last-mile operations with high stop frequency maximise "
            "EV efficiency advantages. Immediate Q1 2025 procurement is recommended "
            "to begin realising fuel and maintenance savings at the start of the "
            "fiscal year."
        ),
    },
    "VEH-003": {
        "vehicle_id": "VEH-003",
        "recommendation": "Proceed with EV conversion. Route stop utilisation for opportunity charging required.",
        "priority": "medium",
        "confidence": "medium",
        "recommended_purchase_window": "Q2 2025",
        "reason": (
            "VEH-003 has a readiness score of 74 with average daily distance "
            "marginally exceeding the Volvo FE Electric's 300 km range. The 14 "
            "daily stops provide sufficient opportunity charging windows to bridge "
            "this gap operationally. Annual savings of USD 22,349 against an "
            "acquisition cost of USD 185,000 produce an 8.3-year payback. The "
            "vehicle's 5-year age and mixed route profile support a Q2 2025 "
            "procurement decision, with depot charging infrastructure to be "
            "confirmed by Q1 2025."
        ),
    },
    "VEH-004": {
        "vehicle_id": "VEH-004",
        "recommendation": "Proceed with EV conversion. Depot charging infrastructure must be confirmed first.",
        "priority": "medium",
        "confidence": "high",
        "recommended_purchase_window": "Q2 2025",
        "reason": (
            "VEH-004 achieves a readiness score of 83 with fixed suburban routes "
            "and high route consistency (0.80), making it well-suited for "
            "electrification. The 5-hour charging window is the tightest in the "
            "fleet and requires a confirmed 150 kW DC fast charger at the Thane "
            "Bus Terminus before procurement. Annual savings of USD 15,010 against "
            "USD 148,000 acquisition cost yield a 9.9-year payback — the longest "
            "in the fleet. The vehicle's 6-year age and 88% utilisation rate "
            "underscore the urgency of replacement. Q2 2025 procurement is "
            "recommended subject to infrastructure confirmation."
        ),
    },
    "VEH-005": {
        "vehicle_id": "VEH-005",
        "recommendation": "Immediate procurement approved. Strongest overall conversion candidate.",
        "priority": "high",
        "confidence": "high",
        "recommended_purchase_window": "Q1 2025",
        "reason": (
            "VEH-005 holds the highest readiness score in the fleet at 95, "
            "with a daily distance of 130 km providing a 172% buffer against the "
            "F-150 Lightning's 354 km range. The 10-hour charging window "
            "comfortably covers the 8-hour charge cycle. At USD 49,995, the "
            "F-150 Lightning Pro is the most affordable EV in the recommended "
            "catalogue. Annual savings of USD 5,130 yield a 9.7-year payback. "
            "While the financial ROI is modest at 2.6% over 10 years, the "
            "operational simplicity, low transition risk, and lowest acquisition "
            "cost make this the ideal pilot vehicle for establishing fleet EV "
            "operations. Q1 2025 procurement is recommended."
        ),
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


def _lookup_procurement_recommendation(vehicle_id: str) -> dict:
    """Retrieve the procurement recommendation record for a given vehicle.

    Args:
        vehicle_id: A validated, stripped vehicle ID.

    Returns:
        The procurement recommendation dictionary.

    Raises:
        ValueError: If no recommendation exists for the given ID.
    """
    record = _PROCUREMENT_RECOMMENDATIONS.get(vehicle_id)
    if record is None:
        valid_ids = ", ".join(sorted(_PROCUREMENT_RECOMMENDATIONS.keys()))
        raise ValueError(
            f"No procurement recommendation found for vehicle_id='{vehicle_id}'. "
            f"Registered IDs: {valid_ids}."
        )
    return record


@tool
def recommend_procurement(vehicle_id: str) -> dict:
    """Generate a final EV procurement recommendation for a fleet vehicle.

    Synthesises the outputs of readiness scoring, EV model matching, and ROI
    analysis to produce a consolidated, actionable procurement decision. The
    recommendation specifies whether to proceed with EV acquisition, the
    priority level relative to other fleet vehicles, the confidence of the
    assessment, and the optimal purchase window.

    Decision framework:
        - Vehicles with readiness score ≥ 85 and payback ≤ 9 years receive
          "high" priority and Q1 purchase windows.
        - Vehicles with readiness score 70–84 or payback 9–10 years receive
          "medium" priority and Q2–Q3 purchase windows.
        - Vehicles with readiness score < 70 or payback > 10 years are flagged
          for further assessment before procurement is approved.
        - Infrastructure requirements (depot charging) gate procurement timing
          where applicable.

    Args:
        vehicle_id: The unique fleet identifier of the target vehicle
                    (e.g., "VEH-005"). Must be a non-empty, non-whitespace
                    string matching a registered vehicle.

    Returns:
        A dictionary containing consolidated procurement decision attributes.
    """
    clean_id = _validate_vehicle_id(vehicle_id)
    
    # 1. Load vehicle data
    from ev_ai_agents.ev_fleet_electrification_agent.tools.fleet_data_tool import fetch_vehicle_data
    vehicle_record = fetch_vehicle_data.invoke({"vehicle_id": clean_id})
    if "error" in vehicle_record:
        # Fallback to hardcoded profiles for backward compatibility
        record = _PROCUREMENT_RECOMMENDATIONS.get(clean_id)
        if record:
            return record
        raise ValueError(f"No vehicle or procurement recommendation found for vehicle_id='{clean_id}'.")
        
    # 2. Extract values for readiness score tool
    daily_distance = float(vehicle_record.get("daily_distance_km", 0.0))
    charging_window = float(vehicle_record.get("charging_window_hours", 
                            vehicle_record.get("available_charging_window_hours", 8.0)))
    idle_minutes = float(vehicle_record.get("idle_time_minutes",
                         vehicle_record.get("avg_idle_minutes", 45.0)))
    stops = int(vehicle_record.get("stops_per_day", 10))
    route_type = str(vehicle_record.get("usage_pattern", 
                     vehicle_record.get("route_type", "mixed"))).lower()
    consistency = float(vehicle_record.get("route_consistency_score", 0.85))
    vehicle_age = float(vehicle_record.get("vehicle_age_years", 3.0))
    
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
    operating_hours = float(vehicle_record.get("operating_hours_per_day", 24.0 - charging_window))
    utilization = float(vehicle_record.get("utilization_rate", 0.75))
    payload = float(vehicle_record.get("payload_requirement_kg", 
                    vehicle_record.get("payload_capacity_kg", 
                    vehicle_record.get("payload_kg", 1000.0))))
                    
    # 3. Calculate readiness score
    from ev_ai_agents.ev_fleet_electrification_agent.tools.readiness_score_tool import calculate_readiness_score
    readiness = calculate_readiness_score.invoke({
        "daily_distance_km": daily_distance,
        "available_charging_window_hours": charging_window,
        "avg_idle_minutes": idle_minutes,
        "stops_per_day": stops,
        "route_type": route_type,
        "route_consistency_score": consistency,
        "vehicle_age_years": vehicle_age,
        "fuel_efficiency_kmpl": fuel_efficiency,
        "operating_hours_per_day": operating_hours,
        "utilization_rate": utilization,
        "payload_kg": payload
    })
    
    readiness_score = int(readiness.get("readiness_score", 50))
    classification = readiness.get("classification", "Conditionally Ready")
    
    # 4. Calculate ROI details
    from ev_ai_agents.ev_fleet_electrification_agent.tools.roi_tool import calculate_roi
    roi = calculate_roi.invoke({"vehicle_id": clean_id})
    payback_years = float(roi.get("estimated_payback_years", 8.0))
    total_savings = float(roi.get("total_annual_savings_usd", 8000.0))
    roi_pct = float(roi.get("roi_percent_over_10_years", 25.0))
    
    # Get recommended EV
    from ev_ai_agents.ev_fleet_electrification_agent.tools.ev_matching_tool import recommend_ev_replacement
    ev_match = recommend_ev_replacement.invoke({
        "daily_distance_km": daily_distance,
        "available_charging_window_hours": charging_window,
        "payload_kg": payload
    })
    recommended_ev = ev_match.get("recommended_ev", "Standard EV")
    
    # 5. Apply procurement decision rules
    if readiness_score >= 85 and payback_years <= 9.0:
        priority = "high"
        confidence = "high"
        recommended_purchase_window = "Q1 2025"
        recommendation = "Immediate procurement approved. Highest priority conversion."
    elif readiness_score >= 70 or (9.0 < payback_years <= 10.0):
        priority = "medium"
        confidence = "high"
        recommended_purchase_window = "Q2 2025"
        recommendation = "Proceed with EV conversion. Depot charging infrastructure must be confirmed first."
    else:
        priority = "low"
        confidence = "medium"
        recommended_purchase_window = "Q4 2025"
        recommendation = "Further operational and route assessment required before procurement approval."
        
    reason = (
        f"{clean_id} achieves a readiness score of {readiness_score} ({classification}) with daily distance of {daily_distance:.1f} km. "
        f"The available charging window of {charging_window:.1f}h is estimated to be sufficient for the recommended EV replacement ({recommended_ev}). "
        f"Financially, the transition is expected to save USD {total_savings:,.0f} annually with an estimated payback period of {payback_years:.1f} years, "
        f"yielding a 10-year ROI of {roi_pct:.1f}%. Therefore, immediate transition priority is rated as {priority}."
    )
    
    return {
        "vehicle_id": clean_id,
        "recommendation": recommendation,
        "priority": priority,
        "confidence": confidence,
        "recommended_purchase_window": recommended_purchase_window,
        "reason": reason
    }
