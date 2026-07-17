from langchain_core.tools import tool


_EV_RECOMMENDATIONS: dict[str, dict] = {
    "VEH-001": {
        "vehicle_id": "VEH-001",
        "recommended_ev": "Freightliner eCascadia",
        "manufacturer": "Freightliner",
        "battery_capacity_kwh": 475.0,
        "estimated_range_km": 400.0,
        "charging_time_hours": 1.5,
        "payload_capacity_kg": 16_000.0,
        "purchase_price_usd": 250_000.0,
        "compatibility_score": 0.76,
        "reason": (
            "The eCascadia is the closest EV equivalent to the Cascadia 126 "
            "for long-haul highway operations. Its 475 kWh battery provides up "
            "to 400 km range, covering the vehicle's 430 km daily average with "
            "opportunity charging at mid-route stops. Payload capacity exceeds "
            "requirement at 16,000 kg. The 7.5-hour overnight charging window "
            "is sufficient for a full recharge via a 350 kW DC fast charger. "
            "Compatibility is moderate due to the proximity of daily distance "
            "to the maximum range limit."
        ),
    },
    "VEH-002": {
        "vehicle_id": "VEH-002",
        "recommended_ev": "Ford E-Transit 350",
        "manufacturer": "Ford",
        "battery_capacity_kwh": 68.0,
        "estimated_range_km": 317.0,
        "charging_time_hours": 7.8,
        "payload_capacity_kg": 1_600.0,
        "purchase_price_usd": 64_990.0,
        "compatibility_score": 0.96,
        "reason": (
            "The E-Transit 350 is a direct EV replacement for the Transit 350, "
            "sharing identical payload capacity of 1,600 kg. The 317 km range "
            "provides an 81% buffer above the vehicle's 175 km daily average, "
            "making it highly compatible for urban last-mile delivery. The "
            "9-hour overnight charging window fully accommodates the 7.8-hour "
            "AC charge cycle. High route consistency (0.72) supports predictable "
            "energy consumption planning."
        ),
    },
    "VEH-003": {
        "vehicle_id": "VEH-003",
        "recommended_ev": "Volvo FE Electric",
        "manufacturer": "Volvo Trucks",
        "battery_capacity_kwh": 200.0,
        "estimated_range_km": 300.0,
        "charging_time_hours": 2.0,
        "payload_capacity_kg": 8_600.0,
        "purchase_price_usd": 185_000.0,
        "compatibility_score": 0.82,
        "reason": (
            "The Volvo FE Electric matches the medium-duty class of the Actros "
            "1845 and supports mixed urban-highway routes. Its 300 km range "
            "covers the vehicle's 320 km average with minimal range risk given "
            "the 14 daily stops that allow opportunity charging. Payload capacity "
            "of 8,600 kg exceeds the 8,000 kg requirement. The 6.5-hour charging "
            "window supports overnight DC fast charging."
        ),
    },
    "VEH-004": {
        "vehicle_id": "VEH-004",
        "recommended_ev": "Tata Ultra EV",
        "manufacturer": "Tata Motors",
        "battery_capacity_kwh": 213.0,
        "estimated_range_km": 300.0,
        "charging_time_hours": 3.0,
        "payload_capacity_kg": 6_500.0,
        "purchase_price_usd": 148_000.0,
        "compatibility_score": 0.89,
        "reason": (
            "The Tata Ultra EV is engineered for suburban passenger operations "
            "and provides a 300 km range that comfortably covers the vehicle's "
            "260 km daily route. Payload capacity matches exactly at 6,500 kg. "
            "High route consistency (0.80) supports efficient energy planning. "
            "The 5-hour charging window is sufficient for a full charge at a "
            "150 kW DC charger. Strong manufacturer continuity with Tata Motors "
            "reduces fleet transition complexity."
        ),
    },
    "VEH-005": {
        "vehicle_id": "VEH-005",
        "recommended_ev": "Ford F-150 Lightning Pro",
        "manufacturer": "Ford",
        "battery_capacity_kwh": 98.0,
        "estimated_range_km": 354.0,
        "charging_time_hours": 8.0,
        "payload_capacity_kg": 900.0,
        "purchase_price_usd": 49_995.0,
        "compatibility_score": 0.93,
        "reason": (
            "The F-150 Lightning Pro delivers 354 km range against a 130 km "
            "daily average, providing a 172% range buffer ideal for mixed-route "
            "field operations. The 10-hour overnight charging window fully "
            "accommodates the 8-hour charge cycle. Payload of 900 kg is "
            "adequate for the vehicle's operational profile. Lowest purchase "
            "price among recommended EVs makes this the most financially "
            "accessible conversion in the fleet."
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


def _lookup_ev_recommendation(vehicle_id: str) -> dict:
    """Retrieve the EV replacement recommendation for a given vehicle.

    Args:
        vehicle_id: A validated, stripped vehicle ID.

    Returns:
        The EV recommendation dictionary.

    Raises:
        ValueError: If no recommendation exists for the given ID.
    """
    record = _EV_RECOMMENDATIONS.get(vehicle_id)
    if record is None:
        valid_ids = ", ".join(sorted(_EV_RECOMMENDATIONS.keys()))
        raise ValueError(
            f"No EV recommendation found for vehicle_id='{vehicle_id}'. "
            f"Registered IDs: {valid_ids}."
        )
    return record


@tool
def recommend_ev_replacement(vehicle_id: str) -> dict:
    """Recommend the most suitable EV replacement for a fleet vehicle.

    Evaluates the vehicle's operational profile — including payload requirements,
    daily distance, available charging windows, route characteristics, and
    operating hours — and selects the best-matched EV model from the supported
    catalogue.

    The recommendation considers the following factors:
        - Payload compatibility between the current ICE vehicle and the EV.
        - Whether the EV range sufficiently covers the vehicle's maximum daily
          distance with an acceptable safety buffer.
        - Whether the available overnight charging window can support a full
          charge cycle for the recommended EV.
        - Route type suitability (urban, highway, mixed, suburban).
        - Operating hours alignment with the EV's duty cycle limits.

    Args:
        vehicle_id: The unique fleet identifier of the target vehicle
                    (e.g., "VEH-003"). Must be a non-empty, non-whitespace
                    string matching a registered vehicle.

    Returns:
        A dictionary containing:
            - vehicle_id (str): Echo of the requested identifier.
            - recommended_ev (str): Full name of the recommended EV model.
            - manufacturer (str): EV manufacturer name.
            - battery_capacity_kwh (float): Usable battery capacity in kWh.
            - estimated_range_km (float): Real-world estimated range in km.
            - charging_time_hours (float): Time to full charge at standard
              DC fast charger in hours.
            - payload_capacity_kg (float): EV maximum payload in kg.
            - purchase_price_usd (float): Estimated purchase price in USD.
            - compatibility_score (float): 0.0–1.0 normalised score indicating
              how well the EV fits the vehicle's operational requirements.
            - reason (str): Human-readable explanation of the recommendation,
              covering range coverage, payload match, charging feasibility,
              and route suitability.

    Raises:
        ValueError: If vehicle_id is not a string, is empty, is whitespace-only,
                    or does not match any registered vehicle.

    Example:
        >>> result = recommend_ev_replacement.invoke({"vehicle_id": "VEH-002"})
        >>> result["recommended_ev"]
        'Ford E-Transit 350'
    """
    clean_id = _validate_vehicle_id(vehicle_id)
    return _lookup_ev_recommendation(clean_id)
