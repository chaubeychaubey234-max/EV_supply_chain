"""
charging_availability_planner.py
==================================
Tool 3 of the Maintenance Operations Optimiser feature.

Purpose:
    Ensure that charging infrastructure availability is aligned with fleet
    vehicle operational schedules and maintenance windows. For a given vehicle,
    this tool recommends the optimal charging station at its depot location,
    determines the appropriate charger type, and calculates the best charging
    time slot considering shift end time and vehicle charging requirements.

Algorithm:
    1. Validate all inputs.
    2. Look up vehicle operational data from fleet_operations_clean.csv to get:
         - charging_window_hours   : available window for charging
         - required_range_km       : range requirement (determines charger class)
         - recommended_ev_type     : vehicle category (influences charger preference)
    3. Determine required charger class based on range and vehicle type:
         - Heavy/long-range (>300 km) → Ultra_Fast (DC 150+ kW)
         - Medium range (150–300 km)  → Fast_DC    (DC 22–150 kW)
         - Short range (<150 km)      → AC         (AC <22 kW)
    4. Filter charging_station_clean.csv:
         a. City must match depot_location (case-insensitive).
         b. Stations with power_kw matching required class preferred.
         c. is_fast_dc must be True if Fast_DC or Ultra_Fast needed.
    5. Rank filtered stations:
         - fast_charger_flag = True  → ranked higher
         - charger_density_score     → higher is better (more ports, lower utilization)
         - power_kw                  → more power preferred
    6. Compute recommended charging start time from shift_end_time.
    7. Return charging plan dict.

Integrates with:
    - clean_data/fleet_operations_clean.csv  (vehicle shift and depot data)
    - clean_data/charging_station_clean.csv  (station availability data)

FastAPI ready:
    The @tool-decorated public function accepts individual primitive arguments
    compatible with FastAPI Query/Body parameters and returns a serializable dict.
"""

from __future__ import annotations

import sys
import os
import re
from datetime import datetime, timedelta
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.tools import tool

from utils import (
    load_fleet_operations,
    load_charging_stations,
    validate_string,
    validate_float,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Charging speed categories and their power thresholds (kW)
_CHARGER_CLASS_THRESHOLDS = {
    "Ultra_Fast": 150.0,   # power_kw >= 150
    "Fast_DC":    22.0,    # 22 <= power_kw < 150
    "AC":         0.0,     # power_kw < 22
}

# Required charger class per range bucket
_RANGE_TO_CHARGER_CLASS = {
    "high":   "Ultra_Fast",   # > 300 km
    "medium": "Fast_DC",      # 150 – 300 km
    "low":    "AC",           # < 150 km
}

# Vehicle types that always require DC fast charging regardless of range
_ALWAYS_FAST_DC_TYPES = {
    "Truck", "Heavy Commercial", "Light Commercial", "Delivery Van", "Minivan"
}


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_time(time_str: str) -> datetime:
    """Parse a time string into a datetime object (today's date + given time).

    Supported formats: 'HH:MM', 'HH:MM:SS', 'H:MM AM/PM'.

    Args:
        time_str: Time string to parse.

    Returns:
        A datetime object for today at the given time.

    Raises:
        ValueError: If the time string cannot be parsed.
    """
    clean = time_str.strip()
    for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p"):
        try:
            t = datetime.strptime(clean, fmt)
            today = datetime.today().replace(
                hour=t.hour, minute=t.minute, second=0, microsecond=0
            )
            return today
        except ValueError:
            continue
    raise ValueError(
        f"'shift_end_time' could not be parsed: {time_str!r}. "
        "Expected formats: 'HH:MM', 'HH:MM:SS', or 'HH:MM AM/PM'."
    )


def _determine_charger_class(
    required_range_km: float,
    vehicle_type: str,
) -> str:
    """Determine the minimum required charger class for a vehicle.

    Args:
        required_range_km: Vehicle's required range in km.
        vehicle_type:      Vehicle type string from fleet operations.

    Returns:
        One of: 'Ultra_Fast', 'Fast_DC', 'AC'.
    """
    # Heavy commercial vehicles always need DC fast
    vtype_clean = vehicle_type.strip().title()
    if vtype_clean in _ALWAYS_FAST_DC_TYPES:
        if required_range_km > 300:
            return "Ultra_Fast"
        return "Fast_DC"

    # Range-based classification
    if required_range_km > 300:
        return "Ultra_Fast"
    if required_range_km >= 150:
        return "Fast_DC"
    return "AC"


def _estimate_charging_duration(
    required_range_km: float,
    charger_power_kw: float,
    efficiency_assumption_wh_per_km: float = 180.0,
) -> float:
    """Estimate charging duration in hours for a given range and charger power.

    Uses: energy_needed = range_km × efficiency / 1000 (kWh)
          charge_time   = energy_needed / charger_power_kw

    Args:
        required_range_km:             Range needed in km.
        charger_power_kw:              Charger power in kW.
        efficiency_assumption_wh_per_km: Energy consumption in Wh/km (default 180).

    Returns:
        Estimated charge time in hours, rounded to 2 decimal places.
    """
    if charger_power_kw <= 0:
        return 0.0
    energy_kwh = (required_range_km * efficiency_assumption_wh_per_km) / 1000.0
    return round(energy_kwh / charger_power_kw, 2)


def _recommend_charging_time(shift_end_time: str, buffer_minutes: int = 30) -> str:
    """Compute the recommended charging start time after shift end.

    Adds a configurable buffer (default 30 min) to the shift end time to allow
    the driver to complete handover before plugging in.

    Args:
        shift_end_time:  Shift end time string (e.g. '18:30').
        buffer_minutes:  Minutes to add before charging starts. Default: 30.

    Returns:
        Formatted charging start time string 'HH:MM'.
    """
    try:
        shift_end_dt = _parse_time(shift_end_time)
        start_dt     = shift_end_dt + timedelta(minutes=buffer_minutes)
        return start_dt.strftime("%H:%M")
    except ValueError:
        return shift_end_time  # Return as-is if parsing fails


def _lookup_vehicle_ops(vehicle_id: str) -> dict[str, Any]:
    """Look up fleet operational data for a vehicle from fleet_operations_clean.csv.

    Args:
        vehicle_id: Validated vehicle ID string.

    Returns:
        Dict with keys: vehicle_id, charging_window_hours, required_range_km,
        vehicle_type, recommended_ev_type, depot_location.

    Raises:
        ValueError: If the vehicle_id is not found.
    """
    df = load_fleet_operations()
    match = df[df["vehicle_id"].astype(str).str.strip() == vehicle_id]
    if match.empty:
        sample = ", ".join(df["vehicle_id"].astype(str).head(5).tolist()) + "…"
        raise ValueError(
            f"vehicle_id='{vehicle_id}' not found in fleet_operations_clean.csv. "
            f"Sample IDs: {sample}"
        )
    row = match.iloc[0]
    return {
        "vehicle_id":          vehicle_id,
        "charging_window_hours": float(row.get("charging_window_hours", 8.0)),
        "required_range_km":   float(row.get("required_range_km", 150.0)),
        "vehicle_type":        str(row.get("vehicle_type", "Unknown")),
        "recommended_ev_type": str(row.get("recommended_ev_type", "Unknown")),
        "depot_location":      str(row.get("depot_location", "Unknown")),
        "ev_suitable":         bool(row.get("ev_suitable", False)),
    }


def _filter_stations_by_city(city: str, charger_class: str) -> Any:
    """Filter and rank charging stations for a given depot city and charger class.

    Filtering logic:
        1. City match (case-insensitive, Title Case normalized).
        2. Charger class match: is_fast_dc required for Fast_DC / Ultra_Fast.
        3. Power threshold: station power_kw must meet minimum for charger class.

    Ranking:
        - fast_charger_flag = True first
        - charger_density_score descending
        - power_kw descending

    Args:
        city:          Depot city name to match.
        charger_class: Required charger class ('Ultra_Fast', 'Fast_DC', 'AC').

    Returns:
        Filtered and ranked DataFrame slice (may be empty).
    """
    df = load_charging_stations()

    # Normalize city for matching
    city_norm = city.strip().title()
    city_match = df[df["city"].astype(str).str.strip().str.title() == city_norm].copy()

    if city_match.empty:
        return city_match  # empty DataFrame

    # Apply charger class filter
    if charger_class in ("Ultra_Fast", "Fast_DC"):
        # Require DC fast charging capability
        city_match = city_match[city_match["is_fast_dc"] == True]
        min_power = _CHARGER_CLASS_THRESHOLDS.get(charger_class, 22.0)
        city_match = city_match[city_match["power_kw"] >= min_power]
    # AC: no additional filtering needed

    if city_match.empty:
        return city_match

    # Rank: fast_charger_flag desc → charger_density_score desc → power_kw desc
    city_match = city_match.sort_values(
        by=["fast_charger_flag", "charger_density_score", "power_kw"],
        ascending=[False, False, False],
    )
    return city_match


def _build_charging_plan(
    vehicle_ops: dict[str, Any],
    station_row: dict[str, Any],
    charger_class: str,
    shift_end_time: str,
    charging_start_time: str,
) -> dict[str, Any]:
    """Construct the full charging plan output dict.

    Args:
        vehicle_ops:         Fleet ops data for the vehicle.
        station_row:         Best matching station as a dict.
        charger_class:       Determined charger class.
        shift_end_time:      Original shift end time string.
        charging_start_time: Computed charging start time string.

    Returns:
        Complete charging plan dict.
    """
    power_kw = float(station_row.get("power_kw", 22.0))
    charge_duration = _estimate_charging_duration(
        vehicle_ops["required_range_km"], power_kw
    )

    return {
        "vehicle_id":                  vehicle_ops["vehicle_id"],
        "depot_location":              vehicle_ops["depot_location"],
        "vehicle_type":                vehicle_ops["vehicle_type"],
        "required_range_km":           vehicle_ops["required_range_km"],
        "charging_window_hours":       vehicle_ops["charging_window_hours"],
        "recommended_station":         f"Station_{station_row.get('station_id', 'N/A')}",
        "station_id":                  int(station_row.get("station_id", 0)),
        "station_city":                str(station_row.get("city", "Unknown")),
        "charger_type":                str(station_row.get("charger_type", charger_class)),
        "recommended_charger_class":   charger_class,
        "station_power_kw":            power_kw,
        "port_count":                  int(station_row.get("port_count", 1)),
        "is_fast_dc":                  bool(station_row.get("is_fast_dc", False)),
        "charger_density_score":       float(station_row.get("charger_density_score", 0.0)),
        "shift_end_time":              shift_end_time,
        "charging_time":               charging_start_time,
        "estimated_charge_duration_hours": charge_duration,
        "charging_feasible_in_window": charge_duration <= vehicle_ops["charging_window_hours"],
        "recommendation_confidence":   "high" if charger_class != "AC" else "medium",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_planner_inputs(
    vehicle_id:      str,
    depot_location:  str,
    shift_end_time:  str,
) -> tuple[str, str, str]:
    """Validate all inputs for plan_charging_availability.

    Args:
        vehicle_id:     Fleet vehicle identifier.
        depot_location: Depot city name.
        shift_end_time: End-of-shift time string.

    Returns:
        Tuple of (validated vehicle_id, depot_location, shift_end_time).

    Raises:
        ValueError: If any input fails validation.
    """
    vid    = validate_string(vehicle_id,    "vehicle_id")
    depot  = validate_string(depot_location, "depot_location")
    time_s = validate_string(shift_end_time, "shift_end_time")

    # Validate time format
    _parse_time(time_s)  # raises ValueError if invalid

    return vid, depot, time_s


# ─────────────────────────────────────────────────────────────────────────────
# Public LangChain tool
# ─────────────────────────────────────────────────────────────────────────────

@tool
def plan_charging_availability(
    vehicle_id:     str,
    depot_location: str,
    shift_end_time: str,
) -> dict:
    """Recommend the optimal charging station and time slot for a fleet vehicle.

    Aligns charging infrastructure availability with the vehicle's operational
    schedule and maintenance window. Selects the best-fit charging station at
    the vehicle's depot city, determines the required charger type from the
    vehicle's range requirements, and calculates the earliest practical charging
    start time after the end of the vehicle's operating shift.

    Station selection logic:
        1. Filters to stations in the vehicle's depot city.
        2. Determines required charger class from required_range_km and vehicle_type:
             - Heavy/long-range (>300 km or heavy vehicle) → Ultra_Fast (≥150 kW DC)
             - Medium range (150–300 km)                   → Fast_DC (22–150 kW)
             - Short range (<150 km)                       → AC (<22 kW)
        3. Excludes stations that do not meet the minimum power threshold.
        4. Ranks by: fast_charger_flag > charger_density_score > station power.
        5. Falls back to any available station in the city if no exact class match found.

    Charging time logic:
        - Adds 30-minute post-shift buffer to shift_end_time to allow vehicle
          handover before plug-in.
        - Verifies the estimated charge duration fits within the available
          charging window from fleet_operations_clean.csv.

    Args:
        vehicle_id (str):
            Fleet vehicle identifier (e.g. 'VH_15592').
            Must exist in fleet_operations_clean.csv.

        depot_location (str):
            Depot city name for station filtering (e.g. 'Mumbai', 'Delhi').
            Case-insensitive. Must be a non-empty string.

        shift_end_time (str):
            End time of the vehicle's operating shift in 'HH:MM' format
            (e.g. '18:30', '20:00'). Used to compute recommended charging start.

    Returns:
        dict: Charging plan containing:
            - vehicle_id (str):                     Echo of the input identifier.
            - depot_location (str):                 Depot city from input.
            - vehicle_type (str):                   Vehicle category from fleet ops.
            - required_range_km (float):            Vehicle range requirement.
            - charging_window_hours (float):        Available daily charging window.
            - recommended_station (str):            Human-readable station label.
            - station_id (int):                     Numeric station identifier.
            - station_city (str):                   City where station is located.
            - charger_type (str):                   Raw charger type from dataset.
            - recommended_charger_class (str):      Required class: AC/Fast_DC/Ultra_Fast.
            - station_power_kw (float):             Station power rating in kW.
            - port_count (int):                     Number of charging ports available.
            - is_fast_dc (bool):                    Whether station supports DC fast charging.
            - charger_density_score (float):        Station availability score 0–1.
            - shift_end_time (str):                 Input shift end time.
            - charging_time (str):                  Recommended start time 'HH:MM'.
            - estimated_charge_duration_hours (float): Estimated time to charge.
            - charging_feasible_in_window (bool):   True if charge fits in window.
            - recommendation_confidence (str):      'high' or 'medium'.
            - fallback_used (bool):                 True if no preferred station found.
            - warning (str):                        Warning message if applicable.

    Raises:
        ValueError: If any input fails validation, the vehicle is not found in
                    fleet_operations_clean.csv, or the time format is invalid.
        RuntimeError: If dataset files cannot be loaded.

    Example:
        >>> plan = plan_charging_availability.invoke({
        ...     "vehicle_id": "VH_15592",
        ...     "depot_location": "Mumbai",
        ...     "shift_end_time": "18:30"
        ... })
        >>> plan["recommended_station"]
        'Station_307660'
        >>> plan["charger_type"]
        'DC_ULTRA_(>=150kW)'
        >>> plan["charging_time"]
        '19:00'
    """
    vid, depot, shift_end = _validate_planner_inputs(
        vehicle_id, depot_location, shift_end_time
    )

    # ── Step 1: Fetch vehicle operational data ───────────────────────────────
    vehicle_ops = _lookup_vehicle_ops(vid)

    # Allow depot_location override (caller may know better than the CSV default)
    effective_depot = depot if depot.strip().title() != "Unknown" else vehicle_ops["depot_location"]

    # ── Step 2: Determine required charger class ─────────────────────────────
    charger_class = _determine_charger_class(
        vehicle_ops["required_range_km"],
        vehicle_ops["vehicle_type"],
    )

    # ── Step 3: Filter and rank stations ────────────────────────────────────
    stations = _filter_stations_by_city(effective_depot, charger_class)
    fallback_used = False
    warning_msg   = ""

    if stations.empty:
        # Fallback 1: Relax to any Fast_DC if Ultra_Fast not available
        if charger_class == "Ultra_Fast":
            stations = _filter_stations_by_city(effective_depot, "Fast_DC")
            if not stations.empty:
                charger_class = "Fast_DC"
                fallback_used = True
                warning_msg = (
                    "No Ultra_Fast station found in depot city. "
                    "Downgraded to Fast_DC — charging time will be longer."
                )

        # Fallback 2: Any station in the city
        if stations.empty:
            all_city = load_charging_stations()
            city_norm = effective_depot.strip().title()
            stations = all_city[
                all_city["city"].astype(str).str.strip().str.title() == city_norm
            ].sort_values(by=["power_kw"], ascending=False)
            if not stations.empty:
                fallback_used = True
                warning_msg = (
                    f"No {charger_class} station found in {effective_depot}. "
                    "Returning highest-power available station as fallback."
                )

    if stations.empty:
        # No stations at all for this city
        return {
            "vehicle_id":                      vid,
            "depot_location":                  effective_depot,
            "vehicle_type":                    vehicle_ops["vehicle_type"],
            "required_range_km":               vehicle_ops["required_range_km"],
            "charging_window_hours":           vehicle_ops["charging_window_hours"],
            "recommended_station":             "None",
            "station_id":                      None,
            "station_city":                    effective_depot,
            "charger_type":                    "N/A",
            "recommended_charger_class":       charger_class,
            "station_power_kw":                0.0,
            "port_count":                      0,
            "is_fast_dc":                      False,
            "charger_density_score":           0.0,
            "shift_end_time":                  shift_end,
            "charging_time":                   "N/A",
            "estimated_charge_duration_hours": 0.0,
            "charging_feasible_in_window":     False,
            "recommendation_confidence":       "none",
            "fallback_used":                   True,
            "warning":                         (
                f"No charging stations found in {effective_depot}. "
                "Consider depot infrastructure investment."
            ),
        }

    # ── Step 4: Select best station ──────────────────────────────────────────
    best_station = stations.iloc[0].to_dict()

    # ── Step 5: Compute charging time ────────────────────────────────────────
    charging_start = _recommend_charging_time(shift_end, buffer_minutes=30)

    # ── Step 6: Build output ──────────────────────────────────────────────────
    plan = _build_charging_plan(
        vehicle_ops, best_station, charger_class, shift_end, charging_start
    )
    plan["fallback_used"] = fallback_used
    plan["warning"]       = warning_msg

    return plan


# ─────────────────────────────────────────────────────────────────────────────
# Sample test calls
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import pandas as pd

    print("=" * 65)
    print("TOOL 3: Charging Availability Planner — Sample Test Calls")
    print("=" * 65)

    # ── Load a few real vehicle IDs from the fleet CSV for testing ───────────
    try:
        fleet_df = load_fleet_operations()
        sample_vehicles = fleet_df.head(4)[["vehicle_id", "depot_location"]].to_dict("records")
    except Exception as e:
        print(f"Could not load fleet data: {e}")
        sample_vehicles = []

    # ── Test 1: First real vehicle from fleet CSV ─────────────────────────
    if sample_vehicles:
        v = sample_vehicles[0]
        print(f"\n[Test 1] Real vehicle: {v['vehicle_id']} @ {v['depot_location']}")
        try:
            plan = plan_charging_availability.invoke({
                "vehicle_id":     v["vehicle_id"],
                "depot_location": v["depot_location"],
                "shift_end_time": "18:30",
            })
            print(json.dumps(plan, indent=2, default=str))
        except Exception as e:
            print(f"  Error: {e}")

    # ── Test 2: Second vehicle, different shift end ────────────────────────
    if len(sample_vehicles) >= 2:
        v = sample_vehicles[1]
        print(f"\n[Test 2] Real vehicle: {v['vehicle_id']} @ {v['depot_location']}")
        try:
            plan = plan_charging_availability.invoke({
                "vehicle_id":     v["vehicle_id"],
                "depot_location": v["depot_location"],
                "shift_end_time": "21:00",
            })
            print(f"  Recommended station : {plan['recommended_station']}")
            print(f"  Charger class       : {plan['recommended_charger_class']}")
            print(f"  Power (kW)          : {plan['station_power_kw']}")
            print(f"  Charging time       : {plan['charging_time']}")
            print(f"  Est. duration (h)   : {plan['estimated_charge_duration_hours']}")
            print(f"  Feasible in window  : {plan['charging_feasible_in_window']}")
            print(f"  Fallback used       : {plan['fallback_used']}")
            if plan.get("warning"):
                print(f"  Warning             : {plan['warning']}")
        except Exception as e:
            print(f"  Error: {e}")

    # ── Test 3: City with no matching station (should trigger fallback) ────
    print("\n[Test 3] Depot city likely with no Ultra_Fast — expect fallback")
    if sample_vehicles:
        v = sample_vehicles[2] if len(sample_vehicles) > 2 else sample_vehicles[0]
        try:
            plan = plan_charging_availability.invoke({
                "vehicle_id":     v["vehicle_id"],
                "depot_location": "Nagpur",
                "shift_end_time": "17:00",
            })
            print(f"  Recommended station     : {plan['recommended_station']}")
            print(f"  Recommended class       : {plan['recommended_charger_class']}")
            print(f"  Charging time           : {plan['charging_time']}")
            print(f"  Fallback used           : {plan['fallback_used']}")
            print(f"  Confidence              : {plan['recommendation_confidence']}")
        except Exception as e:
            print(f"  Error: {e}")

    # ── Test 4: Unknown city — expect 'no station' response ──────────────
    print("\n[Test 4] Unknown depot city — expect no-station response")
    if sample_vehicles:
        v = sample_vehicles[0]
        try:
            plan = plan_charging_availability.invoke({
                "vehicle_id":     v["vehicle_id"],
                "depot_location": "Atlantis_City",
                "shift_end_time": "20:00",
            })
            print(f"  Recommended station : {plan['recommended_station']}")
            print(f"  Warning             : {plan['warning']}")
        except Exception as e:
            print(f"  Error: {e}")

    # ── Test 5: Validation error — bad time format ────────────────────────
    print("\n[Test 5] Validation error — invalid shift_end_time format")
    if sample_vehicles:
        v = sample_vehicles[0]
        try:
            plan_charging_availability.invoke({
                "vehicle_id":     v["vehicle_id"],
                "depot_location": v["depot_location"],
                "shift_end_time": "6pm",
            })
        except ValueError as e:
            print(f"  Caught expected ValueError: {e}")

    # ── Test 6: Validation error — empty vehicle_id ────────────────────────
    print("\n[Test 6] Validation error — empty vehicle_id")
    try:
        plan_charging_availability.invoke({
            "vehicle_id":     "   ",
            "depot_location": "Mumbai",
            "shift_end_time": "18:00",
        })
    except ValueError as e:
        print(f"  Caught expected ValueError: {e}")

    print("\n" + "=" * 65)
    print("All test calls completed.")