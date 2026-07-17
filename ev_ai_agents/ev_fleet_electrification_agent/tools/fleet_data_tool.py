from langchain_core.tools import tool


_VEHICLE_REGISTRY: dict[str, dict] = {
    "VEH-001": {
        "vehicle_id": "VEH-001",
        "vehicle_type": "heavy_duty_truck",
        "manufacturer": "Freightliner",
        "model": "Cascadia 126",
        "fuel_type": "diesel",
        "year": 2021,
        "payload_capacity_kg": 14_500.0,
        "fuel_efficiency_kmpl": 3.2,
        "current_odometer_km": 148_320.0,
        "daily_distance_km": 430.0,
        "average_speed_kmph": 74.5,
        "idle_time_minutes": 110.0,
        "operating_hours_per_day": 8.5,
        "route_type": "highway",
        "depot_location": "Mumbai North Logistics Hub",
    },
    "VEH-002": {
        "vehicle_id": "VEH-002",
        "vehicle_type": "delivery_van",
        "manufacturer": "Ford",
        "model": "Transit 350",
        "fuel_type": "diesel",
        "year": 2022,
        "payload_capacity_kg": 1_600.0,
        "fuel_efficiency_kmpl": 9.5,
        "current_odometer_km": 36_450.0,
        "daily_distance_km": 175.0,
        "average_speed_kmph": 42.0,
        "idle_time_minutes": 55.0,
        "operating_hours_per_day": 7.0,
        "route_type": "urban",
        "depot_location": "Pune City Distribution Centre",
    },
    "VEH-003": {
        "vehicle_id": "VEH-003",
        "vehicle_type": "medium_duty_truck",
        "manufacturer": "Mercedes-Benz",
        "model": "Actros 1845",
        "fuel_type": "diesel",
        "year": 2020,
        "payload_capacity_kg": 8_000.0,
        "fuel_efficiency_kmpl": 5.1,
        "current_odometer_km": 215_780.0,
        "daily_distance_km": 320.0,
        "average_speed_kmph": 62.0,
        "idle_time_minutes": 80.0,
        "operating_hours_per_day": 9.0,
        "route_type": "mixed",
        "depot_location": "Nashik Regional Depot",
    },
    "VEH-004": {
        "vehicle_id": "VEH-004",
        "vehicle_type": "passenger_bus",
        "manufacturer": "Tata Motors",
        "model": "Starbus Ultra",
        "fuel_type": "cng",
        "year": 2019,
        "payload_capacity_kg": 6_500.0,
        "fuel_efficiency_kmpl": 4.8,
        "current_odometer_km": 310_200.0,
        "daily_distance_km": 260.0,
        "average_speed_kmph": 48.0,
        "idle_time_minutes": 95.0,
        "operating_hours_per_day": 12.0,
        "route_type": "suburban",
        "depot_location": "Thane Bus Terminus",
    },
    "VEH-005": {
        "vehicle_id": "VEH-005",
        "vehicle_type": "pickup_truck",
        "manufacturer": "Toyota",
        "model": "Hilux GD-6",
        "fuel_type": "diesel",
        "year": 2023,
        "payload_capacity_kg": 1_000.0,
        "fuel_efficiency_kmpl": 11.2,
        "current_odometer_km": 12_400.0,
        "daily_distance_km": 130.0,
        "average_speed_kmph": 58.0,
        "idle_time_minutes": 30.0,
        "operating_hours_per_day": 6.0,
        "route_type": "mixed",
        "depot_location": "Aurangabad Field Operations Base",
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


def _lookup_vehicle(vehicle_id: str) -> dict:
    """Retrieve a vehicle record from the in-process registry.

    Args:
        vehicle_id: A validated, stripped vehicle ID.

    Returns:
        The complete vehicle data dictionary.

    Raises:
        ValueError: If no record exists for the given ID.
    """
    record = _VEHICLE_REGISTRY.get(vehicle_id)
    if record is None:
        valid_ids = ", ".join(sorted(_VEHICLE_REGISTRY.keys()))
        raise ValueError(
            f"No vehicle found for vehicle_id='{vehicle_id}'. "
            f"Registered IDs: {valid_ids}."
        )
    return record


@tool
def fetch_vehicle_data(vehicle_id: str) -> dict:
    """Retrieve complete asset and telemetry data for a fleet vehicle.

    This tool is the primary data source for fleet vehicle information within
    the EV Fleet Electrification Readiness Platform. It returns a snapshot of
    a vehicle's static asset attributes and real-time operational telemetry,
    which downstream tools use to assess EV conversion feasibility.

    Args:
        vehicle_id: The unique fleet identifier of the target vehicle
                    (e.g., "VEH-001"). Must be a non-empty, non-whitespace
                    string matching a registered vehicle.

    Returns:
        A dictionary containing:

        Asset attributes:
            - vehicle_id (str): Unique fleet identifier.
            - vehicle_type (str): Category (e.g., "heavy_duty_truck").
            - manufacturer (str): OEM name.
            - model (str): Model designation.
            - fuel_type (str): Current propulsion fuel.
            - year (int): Manufacturing or registration year.
            - payload_capacity_kg (float): Maximum cargo payload in kg.

        Operational telemetry:
            - fuel_efficiency_kmpl (float): Kilometres per litre.
            - current_odometer_km (float): Total distance driven in km.
            - daily_distance_km (float): Average daily distance in km.
            - average_speed_kmph (float): Mean operational speed in km/h.
            - idle_time_minutes (float): Average daily idle time in minutes.
            - operating_hours_per_day (float): Daily engine operating hours.

        Logistical metadata:
            - route_type (str): Classification (urban/suburban/highway/mixed).
            - depot_location (str): Home depot name or identifier.

    Raises:
        ValueError: If vehicle_id is not a string, is empty, is whitespace-only,
                    or does not match any registered vehicle.

    Example:
        >>> result = fetch_vehicle_data.invoke({"vehicle_id": "VEH-001"})
        >>> result["manufacturer"]
        'Freightliner'
    """
    clean_id = _validate_vehicle_id(vehicle_id)
    return _lookup_vehicle(clean_id)
