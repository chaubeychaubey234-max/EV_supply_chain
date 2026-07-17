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
        A dictionary containing:
            - vehicle_id (str): Echo of the requested identifier.
            - estimated_annual_fuel_cost_usd (float): Current annual diesel or
              CNG fuel expenditure in USD.
            - estimated_annual_electricity_cost_usd (float): Projected annual
              electricity cost for the EV equivalent in USD.
            - annual_fuel_savings_usd (float): Difference between fuel and
              electricity cost per year in USD.
            - annual_maintenance_savings_usd (float): Estimated annual reduction
              in maintenance expenditure from ICE to EV in USD.
            - total_annual_savings_usd (float): Combined fuel and maintenance
              savings per year in USD.
            - ev_purchase_price_usd (float): Purchase price of the recommended
              EV model in USD.
            - estimated_payback_years (float): Years until cumulative savings
              recover the EV purchase investment.
            - roi_percent_over_10_years (float): Percentage return on investment
              over a 10-year operating horizon.

    Raises:
        ValueError: If vehicle_id is not a string, is empty, is whitespace-only,
                    or does not match any registered vehicle.

    Example:
        >>> result = calculate_roi.invoke({"vehicle_id": "VEH-001"})
        >>> result["estimated_payback_years"]
        6.4
    """
    clean_id = _validate_vehicle_id(vehicle_id)
    return _lookup_roi_profile(clean_id)
