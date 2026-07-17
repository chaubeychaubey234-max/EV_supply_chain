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
        A dictionary containing:
            - vehicle_id (str): Echo of the requested identifier.
            - recommendation (str): The procurement decision statement.
            - priority (str): Relative acquisition priority.
              One of: "high", "medium", "low".
            - confidence (str): Confidence in the recommendation.
              One of: "high", "medium", "low".
            - recommended_purchase_window (str): Suggested calendar quarter
              and year for EV acquisition (e.g., "Q1 2025").
            - reason (str): Comprehensive justification citing readiness score,
              ROI figures, operational fit, and any preconditions for procurement.

    Raises:
        ValueError: If vehicle_id is not a string, is empty, is whitespace-only,
                    or does not match any registered vehicle.

    Example:
        >>> result = recommend_procurement.invoke({"vehicle_id": "VEH-005"})
        >>> result["priority"]
        'high'
    """
    clean_id = _validate_vehicle_id(vehicle_id)
    return _lookup_procurement_recommendation(clean_id)
