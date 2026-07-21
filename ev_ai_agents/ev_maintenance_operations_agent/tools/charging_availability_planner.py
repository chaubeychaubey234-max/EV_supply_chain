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
    """Look up fleet operational data for a vehicle from fleet_operations_clean.csv or maintenance history.

    Args:
        vehicle_id: Validated vehicle ID string.

    Returns:
        Dict with keys: vehicle_id, charging_window_hours, required_range_km,
        vehicle_type, recommended_ev_type, depot_location.
    """
    try:
        df = load_fleet_operations()
        match = df[df["vehicle_id"].astype(str).str.strip() == vehicle_id]
        if not match.empty:
            row = match.iloc[0]
            return {
                "vehicle_id":          vehicle_id,
                "charging_window_hours": float(row.get("charging_window_hours", 8.0)),
                "required_range_km":   float(row.get("required_range_km", 150.0)),
                "vehicle_type":        str(row.get("vehicle_type", "Delivery Van")),
                "recommended_ev_type": str(row.get("recommended_ev_type", "Light Commercial")),
                "depot_location":      str(row.get("depot_location", "Delhi")),
                "ev_suitable":         bool(row.get("ev_suitable", True)),
            }
    except Exception:
        pass

    # Check vehicle_maintenance_history.csv as secondary fallback
    try:
        from utils import load_maintenance_history
        df_h = load_maintenance_history()
        match_h = df_h[df_h["vehicle_id"].astype(str).str.strip() == vehicle_id]
        if not match_h.empty:
            row = match_h.iloc[0]
            return {
                "vehicle_id":          vehicle_id,
                "charging_window_hours": 8.0,
                "required_range_km":   220.0,
                "vehicle_type":        str(row.get("vehicle_model", "Commercial EV")),
                "recommended_ev_type": "Light Commercial EV",
                "depot_location":      "Delhi",
                "ev_suitable":         True,
            }
    except Exception:
        pass

    return {
        "vehicle_id":          vehicle_id,
        "charging_window_hours": 8.0,
        "required_range_km":   200.0,
        "vehicle_type":        "Commercial EV",
        "recommended_ev_type": "Light Commercial EV",
        "depot_location":      "Delhi",
        "ev_suitable":         True,
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
        dc_match = city_match[city_match["is_fast_dc"] == True]
        min_power = _CHARGER_CLASS_THRESHOLDS.get(charger_class, 22.0)
        p_match = dc_match[dc_match["power_kw"] >= min_power]
        if not p_match.empty:
            city_match = p_match
        elif not dc_match.empty:
            city_match = dc_match

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
    """Construct the full charging plan output dict."""
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
        "recommended_station":         f"Station_{station_row.get('station_id', '312046')}",
        "station_id":                  int(station_row.get("station_id", 312046)),
        "station_city":                str(station_row.get("city", "Delhi")),
        "charger_type":                str(station_row.get("charger_type", charger_class)),
        "recommended_charger_class":   charger_class,
        "station_power_kw":            power_kw,
        "port_count":                  int(station_row.get("port_count", 4)),
        "is_fast_dc":                  bool(station_row.get("is_fast_dc", True)),
        "charger_density_score":       float(station_row.get("charger_density_score", 0.95)),
        "shift_end_time":              shift_end_time,
        "charging_time":               charging_start_time,
        "estimated_charge_duration_hours": charge_duration,
        "charging_feasible_in_window": charge_duration <= vehicle_ops["charging_window_hours"],
        "recommendation_confidence":   "high" if charger_class != "AC" else "medium",
    }


def _validate_planner_inputs(
    vehicle_id:      str,
    depot_location:  str,
    shift_end_time:  str,
) -> tuple[str, str, str]:
    """Validate all inputs for plan_charging_availability."""
    vid    = validate_string(vehicle_id,    "vehicle_id")
    depot  = validate_string(depot_location, "depot_location")
    time_s = validate_string(shift_end_time, "shift_end_time")

    _parse_time(time_s)  # raises ValueError if invalid
    return vid, depot, time_s


@tool
def plan_charging_availability(
    vehicle_id:     str,
    depot_location: str,
    shift_end_time: str,
) -> dict:
    """Recommend the optimal charging station and time slot for a fleet vehicle."""
    vid, depot, shift_end = _validate_planner_inputs(
        vehicle_id, depot_location, shift_end_time
    )

    vehicle_ops = _lookup_vehicle_ops(vid)

    effective_depot = depot if depot.strip().title() != "Unknown" else vehicle_ops["depot_location"]

    charger_class = _determine_charger_class(
        vehicle_ops["required_range_km"],
        vehicle_ops["vehicle_type"],
    )

    stations = _filter_stations_by_city(effective_depot, charger_class)
    fallback_used = False
    warning_msg   = ""

    if stations.empty:
        if charger_class == "Ultra_Fast":
            stations = _filter_stations_by_city(effective_depot, "Fast_DC")
            if not stations.empty:
                charger_class = "Fast_DC"
                fallback_used = True
                warning_msg = (
                    "No Ultra_Fast station found in depot city. "
                    "Downgraded to Fast_DC — charging time will be longer."
                )

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
        # Fallback to top DC fast charging station in regional hub
        all_stations = load_charging_stations()
        stations = all_stations[all_stations["is_fast_dc"] == True].sort_values(
            by=["charger_density_score", "power_kw"], ascending=[False, False]
        )
        if not stations.empty:
            fallback_used = True
            warning_msg = f"No station matching {effective_depot}. Assigned regional hub fast charging station as fallback."

    best_station = stations.iloc[0].to_dict()
    charging_start = _recommend_charging_time(shift_end, buffer_minutes=30)

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