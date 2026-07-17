from langchain_core.tools import tool


_READINESS_PROFILES: dict[str, dict] = {
    "VEH-001": {
        "vehicle_id": "VEH-001",
        "readiness_score": 64,
        "confidence": "medium",
        "classification": "Conditionally Ready",
        "reason": (
            "VEH-001 operates 430 km daily, which is close to the upper boundary "
            "of current heavy EV range (400 km). While overnight charging (7.5 h) "
            "is available and payload is covered, the daily distance exceeds the "
            "estimated EV range, requiring route optimisation or mid-route "
            "opportunity charging to be viable. Idle time of 110 min/day and "
            "highway route type support charging feasibility. Vehicle age of 4 "
            "years and moderate fuel efficiency (3.2 kmpl) indicate urgency for "
            "transition planning. Score reflects operational risk from range "
            "boundary proximity."
        ),
    },
    "VEH-002": {
        "vehicle_id": "VEH-002",
        "readiness_score": 91,
        "confidence": "high",
        "classification": "Highly Ready",
        "reason": (
            "VEH-002 is an ideal EV conversion candidate. Daily distance of "
            "175 km is well within the 317 km E-Transit range, providing an "
            "81% buffer. The 9-hour overnight charging window fully supports a "
            "complete charge cycle. Urban route type with 22 stops per day "
            "enables opportunity charging and minimises idle energy waste. "
            "Vehicle age of 2 years and good fuel efficiency (9.5 kmpl) reduce "
            "immediate financial pressure, but early conversion locks in savings "
            "across the vehicle's full EV service life."
        ),
    },
    "VEH-003": {
        "vehicle_id": "VEH-003",
        "readiness_score": 74,
        "confidence": "medium",
        "classification": "Ready with Conditions",
        "reason": (
            "VEH-003 averages 320 km daily on mixed routes, marginally exceeding "
            "the Volvo FE Electric's 300 km range. The 14 daily stops provide "
            "opportunity charging windows to bridge the gap. The vehicle's age "
            "of 5 years and moderate efficiency (5.1 kmpl) make electrification "
            "financially compelling. A 6.5-hour overnight charging window is "
            "adequate for DC fast charging. Route consistency score of 0.65 "
            "introduces mild prediction uncertainty in daily energy consumption."
        ),
    },
    "VEH-004": {
        "vehicle_id": "VEH-004",
        "readiness_score": 83,
        "confidence": "high",
        "classification": "Ready",
        "reason": (
            "VEH-004 operates 260 km per day on suburban fixed routes with a "
            "route consistency score of 0.80, making energy consumption highly "
            "predictable. The Tata Ultra EV's 300 km range provides a 15% "
            "buffer. High utilisation (88%) and 12 operating hours per day "
            "indicate high fuel expenditure, amplifying ROI from conversion. "
            "The 5-hour charging window is tight but sufficient with 150 kW DC "
            "charging. Vehicle age of 6 years increases the urgency of replacement."
        ),
    },
    "VEH-005": {
        "vehicle_id": "VEH-005",
        "readiness_score": 95,
        "confidence": "high",
        "classification": "Highly Ready",
        "reason": (
            "VEH-005 is the strongest conversion candidate in the fleet. Daily "
            "distance of 130 km is 63% below the F-150 Lightning's 354 km range, "
            "eliminating range anxiety entirely. A 10-hour overnight charging "
            "window far exceeds the 8-hour charge requirement. Low idle time "
            "(30 min/day), low operating hours (6 h/day), and a recent "
            "registration year (2023) confirm operational efficiency. Mixed "
            "route type supports flexible charging scheduling. The lowest "
            "purchase price in the recommended EV catalogue further strengthens "
            "the investment case."
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


def _lookup_readiness_profile(vehicle_id: str) -> dict:
    """Retrieve the readiness scoring profile for a given vehicle.

    Args:
        vehicle_id: A validated, stripped vehicle ID.

    Returns:
        The readiness profile dictionary.

    Raises:
        ValueError: If no profile exists for the given ID.
    """
    profile = _READINESS_PROFILES.get(vehicle_id)
    if profile is None:
        valid_ids = ", ".join(sorted(_READINESS_PROFILES.keys()))
        raise ValueError(
            f"No readiness profile found for vehicle_id='{vehicle_id}'. "
            f"Registered IDs: {valid_ids}."
        )
    return profile


@tool
def calculate_readiness_score(vehicle_id: str) -> dict:
    """Estimate the EV electrification readiness of a fleet vehicle.

    Computes a deterministic readiness score by evaluating the vehicle's
    operational profile against key electrification viability factors:

        - Daily distance vs available EV range (range coverage headroom)
        - Available charging window vs required charge time
        - Average idle time and stop frequency (opportunity charging potential)
        - Route type and consistency score (predictability of energy consumption)
        - Vehicle age (replacement urgency and remaining ICE service life)
        - Fuel efficiency (magnitude of fuel cost savings from conversion)
        - Operating hours and utilisation rate (financial impact multiplier)
        - Payload compatibility with available EV models

    The output is used by procurement and ROI tools to prioritise fleet
    electrification investments.

    Args:
        vehicle_id: The unique fleet identifier of the target vehicle
                    (e.g., "VEH-004"). Must be a non-empty, non-whitespace
                    string matching a registered vehicle.

    Returns:
        A dictionary containing:
            - vehicle_id (str): Echo of the requested identifier.
            - readiness_score (int): Integer score from 0 to 100 where higher
              values indicate greater electrification readiness.
            - confidence (str): Confidence level of the assessment.
              One of: "high", "medium", "low".
            - classification (str): Human-readable readiness label.
              One of: "Highly Ready", "Ready", "Ready with Conditions",
              "Conditionally Ready", "Not Ready".
            - reason (str): Detailed explanation of the score, citing
              specific operational factors that drove the assessment.

    Raises:
        ValueError: If vehicle_id is not a string, is empty, is whitespace-only,
                    or does not match any registered vehicle.

    Example:
        >>> result = calculate_readiness_score.invoke({"vehicle_id": "VEH-005"})
        >>> result["readiness_score"]
        95
    """
    clean_id = _validate_vehicle_id(vehicle_id)
    return _lookup_readiness_profile(clean_id)
