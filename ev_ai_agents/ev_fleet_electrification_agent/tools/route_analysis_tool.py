from langchain_core.tools import tool


_ROUTE_PROFILES: dict[str, dict] = {
    "VEH-001": {
        "vehicle_id": "VEH-001",
        "average_daily_distance_km": 430.0,
        "maximum_daily_distance_km": 610.0,
        "average_speed_kmph": 74.5,
        "number_of_stops": 8,
        "average_idle_time_minutes": 110.0,
        "vehicle_utilization_percent": 85.0,
        "overnight_parking": True,
        "available_charging_window_hours": 7.5,
        "route_consistency_score": 0.87,
    },
    "VEH-002": {
        "vehicle_id": "VEH-002",
        "average_daily_distance_km": 175.0,
        "maximum_daily_distance_km": 240.0,
        "average_speed_kmph": 42.0,
        "number_of_stops": 22,
        "average_idle_time_minutes": 55.0,
        "vehicle_utilization_percent": 72.0,
        "overnight_parking": True,
        "available_charging_window_hours": 9.0,
        "route_consistency_score": 0.72,
    },
    "VEH-003": {
        "vehicle_id": "VEH-003",
        "average_daily_distance_km": 320.0,
        "maximum_daily_distance_km": 420.0,
        "average_speed_kmph": 62.0,
        "number_of_stops": 14,
        "average_idle_time_minutes": 80.0,
        "vehicle_utilization_percent": 78.0,
        "overnight_parking": True,
        "available_charging_window_hours": 6.5,
        "route_consistency_score": 0.65,
    },
    "VEH-004": {
        "vehicle_id": "VEH-004",
        "average_daily_distance_km": 260.0,
        "maximum_daily_distance_km": 320.0,
        "average_speed_kmph": 48.0,
        "number_of_stops": 18,
        "average_idle_time_minutes": 95.0,
        "vehicle_utilization_percent": 88.0,
        "overnight_parking": True,
        "available_charging_window_hours": 5.0,
        "route_consistency_score": 0.80,
    },
    "VEH-005": {
        "vehicle_id": "VEH-005",
        "average_daily_distance_km": 130.0,
        "maximum_daily_distance_km": 190.0,
        "average_speed_kmph": 58.0,
        "number_of_stops": 10,
        "average_idle_time_minutes": 30.0,
        "vehicle_utilization_percent": 60.0,
        "overnight_parking": True,
        "available_charging_window_hours": 10.0,
        "route_consistency_score": 0.75,
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


def _lookup_route_profile(vehicle_id: str) -> dict:
    """Retrieve historical route data for a given vehicle.

    Args:
        vehicle_id: A validated, stripped vehicle ID.

    Returns:
        The route analytics dictionary for the vehicle.

    Raises:
        ValueError: If no route profile exists for the given ID.
    """
    profile = _ROUTE_PROFILES.get(vehicle_id)
    if profile is None:
        valid_ids = ", ".join(sorted(_ROUTE_PROFILES.keys()))
        raise ValueError(
            f"No route profile found for vehicle_id='{vehicle_id}'. "
            f"Registered IDs: {valid_ids}."
        )
    return profile


@tool
def analyze_vehicle_route(vehicle_id: str) -> dict:
    """Analyse historical route utilisation data for a fleet vehicle.

    Processes historical route records for the specified vehicle and returns
    aggregated operational metrics covering distance patterns, stop behaviour,
    idle time, speed profiles, utilisation rate, overnight parking availability,
    and the estimated daily window available for EV charging.

    These metrics are consumed by downstream tools — including EV matching,
    readiness scoring, and ROI estimation — to evaluate electrification suitability.

    Args:
        vehicle_id: The unique fleet identifier of the target vehicle
                    (e.g., "VEH-002"). Must be a non-empty, non-whitespace
                    string matching a registered vehicle.

    Returns:
        A dictionary containing:
            - vehicle_id (str): Echo of the requested identifier.
            - average_daily_distance_km (float): Mean distance driven per day.
            - maximum_daily_distance_km (float): Peak single-day distance recorded.
            - average_speed_kmph (float): Mean operational speed across all routes.
            - number_of_stops (int): Average number of stops per operational day.
            - average_idle_time_minutes (float): Mean idle engine time per day.
            - vehicle_utilization_percent (float): Percentage of available
              operating time the vehicle is actively in use (0–100).
            - overnight_parking (bool): Whether the vehicle parks overnight at a
              fixed depot, indicating overnight charging is feasible.
            - available_charging_window_hours (float): Estimated daily hours
              during which EV charging is practically available.
            - route_consistency_score (float): 0.0–1.0 normalised score; higher
              values indicate more predictable, repeatable daily routes.

    Raises:
        ValueError: If vehicle_id is not a string, is empty, is whitespace-only,
                    or does not match any registered vehicle.

    Example:
        >>> result = analyze_vehicle_route.invoke({"vehicle_id": "VEH-002"})
        >>> result["route_consistency_score"]
        0.72
    """
    clean_id = _validate_vehicle_id(vehicle_id)
    return _lookup_route_profile(clean_id)
