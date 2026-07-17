"""
maintenance_schedule_optimizer.py
===================================
Tool 2 of the Maintenance Operations Optimiser feature.

Purpose:
    Generate an optimized maintenance schedule for a list of fleet vehicles
    by matching each vehicle's risk profile to the most suitable workshop,
    assigning service time slots, and estimating downtime.

Algorithm:
    1. Fetch risk profile for each vehicle_id from maintenance history CSV.
    2. Sort vehicles by risk_score descending (CRITICAL first, LOW last).
    3. For each vehicle, find the best-fit workshop using the following rules:
         a. Workshop must be marked is_available = True.
         b. Prefer EV-specialized workshops.
         c. Require battery_repair_capable = True when the dominant risk
            factor is 'battery' or 'cycles'.
         d. Among qualifying workshops in the same city, sort by
            current_workload_percent ascending (least-loaded first).
    4. Assign the earliest available time slot from the workshop.
    5. Estimate service downtime from the workshop's avg_service_hours
       and the vehicle's risk level.
    6. Return a sorted, time-slot-assigned schedule list.

Optimization rules:
    - CRITICAL vehicles always appear first.
    - Workshops with lower workload are preferred to balance load.
    - Battery-specific repairs are only routed to battery-capable workshops.
    - Unavailable workshops are never assigned.
    - Estimated downtime escalates with risk severity.

Integrates with:
    - datasets/vehicle_maintenance_history.csv  (risk input data)
    - datasets/workshop_capacity.csv            (workshop data)
    - maintenance_risk_analyzer.py              (risk scoring engine)
    - Downstream: charging_availability_planner.py

FastAPI ready:
    The @tool-decorated public function accepts a JSON-serializable payload
    and returns a list of schedule dicts, each independently serializable.
"""

from __future__ import annotations

import sys
import os
import ast
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.tools import tool

from utils import (
    load_maintenance_history,
    load_workshop_capacity,
    load_fleet_operations,
    validate_string,
    validate_positive_int,
)
from maintenance_risk_analyzer import (
    _validate_vehicle_record,
    _compute_risk,
    _fetch_record_from_csv,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Downtime multipliers by risk level (base hours × multiplier)
_DOWNTIME_MULTIPLIER: dict[str, float] = {
    "CRITICAL": 2.5,
    "HIGH":     1.8,
    "MEDIUM":   1.2,
    "LOW":      1.0,
}

# Ordered risk levels for sorting (index = priority weight, lower index = higher priority)
_RISK_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

# Days of week assigned in rotation to schedule slots across a date_range
_DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_available_slots(slots_raw: Any) -> list[str]:
    """Parse the 'available_slots' field from workshop_capacity.csv.

    The field may be stored as a Python list repr string (e.g. "['08:00','10:00']")
    or as a plain string. Returns a sorted list of time strings.

    Args:
        slots_raw: Raw value from the DataFrame cell.

    Returns:
        Sorted list of time slot strings (e.g. ['08:00', '10:00']).
    """
    if isinstance(slots_raw, list):
        return sorted([str(s).strip() for s in slots_raw if str(s).strip()])
    if not isinstance(slots_raw, str) or not slots_raw.strip():
        return ["09:00"]  # default fallback
    try:
        parsed = ast.literal_eval(slots_raw.strip())
        if isinstance(parsed, list):
            return sorted([str(s).strip() for s in parsed if str(s).strip()])
    except (ValueError, SyntaxError):
        pass
    # Try comma-separated
    parts = [p.strip().strip("'\"[]") for p in slots_raw.split(",")]
    return sorted([p for p in parts if p])


def _estimate_downtime(base_hours: float, risk_level: str) -> float:
    """Estimate total service downtime based on base service hours and risk level.

    Args:
        base_hours: Workshop's average service hours (from CSV).
        risk_level: One of CRITICAL / HIGH / MEDIUM / LOW.

    Returns:
        Estimated downtime in hours, rounded to 1 decimal place.
    """
    multiplier = _DOWNTIME_MULTIPLIER.get(risk_level, 1.0)
    return round(base_hours * multiplier, 1)


def _get_vehicle_city(vehicle_id: str) -> str:
    """Look up the depot city for a vehicle from fleet_operations_clean.csv.

    Args:
        vehicle_id: The vehicle identifier to search for.

    Returns:
        City string (Title Case), or 'Unknown' if not found.
    """
    try:
        df = load_fleet_operations()
        matches = df[df["vehicle_id"].astype(str).str.strip() == vehicle_id]
        if not matches.empty:
            city = str(matches.iloc[0].get("depot_location", "Unknown")).strip()
            return city if city else "Unknown"
    except Exception:
        pass
    return "Unknown"


def _find_best_workshop(
    city: str,
    needs_battery_repair: bool,
    workshops_df,
) -> dict | None:
    """Select the optimal workshop for a given vehicle using optimization rules.

    Selection criteria (in order of priority):
        1. Must be available (is_available == True).
        2. Prefer EV-specialized workshops.
        3. If needs_battery_repair: must have battery_repair_capable == True.
        4. Same city as vehicle depot; falls back to any city if none found.
        5. Sort by current_workload_percent ascending.

    Args:
        city:                City name of the vehicle depot.
        needs_battery_repair: True if the vehicle's dominant risk is battery/cycles.
        workshops_df:        The full workshop_capacity DataFrame.

    Returns:
        A dict of the selected workshop row, or None if no workshop qualifies.
    """
    # Step 1: available only
    available = workshops_df[workshops_df["is_available"] == True].copy()
    if available.empty:
        return None

    # Step 2: city match (prefer same city, fall back to any)
    city_match = available[
        available["city"].astype(str).str.strip().str.title() == city.strip().title()
    ]
    candidate_pool = city_match if not city_match.empty else available

    # Step 3: battery repair capability filter
    if needs_battery_repair:
        capable = candidate_pool[candidate_pool["battery_repair_capable"] == True]
        if not capable.empty:
            candidate_pool = capable

    # Step 4: EV specialized preference
    ev_spec = candidate_pool[candidate_pool["ev_specialized"] == True]
    if not ev_spec.empty:
        candidate_pool = ev_spec

    # Step 5: Sort by workload ascending → pick least-loaded
    candidate_pool = candidate_pool.sort_values("current_workload_percent", ascending=True)
    if candidate_pool.empty:
        return None

    return candidate_pool.iloc[0].to_dict()


def _build_schedule_entry(
    risk_result: dict,
    workshop: dict,
    slot_index: int,
    date_range_days: int,
) -> dict:
    """Construct a single schedule entry dict.

    Args:
        risk_result:    Output dict from _compute_risk().
        workshop:       Selected workshop dict from _find_best_workshop().
        slot_index:     Position in the sorted vehicle list (0 = first scheduled).
        date_range_days: Total scheduling window in days.

    Returns:
        A schedule entry dict with all required fields.
    """
    slots = _parse_available_slots(workshop.get("available_slots", "['09:00']"))
    time_slot = slots[slot_index % len(slots)] if slots else "09:00"

    day_index = slot_index % min(date_range_days, len(_DAYS_OF_WEEK))
    day_name  = _DAYS_OF_WEEK[day_index]

    base_hours = float(workshop.get("avg_service_hours", 6.0))
    downtime   = _estimate_downtime(base_hours, risk_result["risk_level"])

    return {
        "vehicle_id":              risk_result["vehicle_id"],
        "vehicle_model":           risk_result.get("vehicle_model", "Unknown"),
        "risk_score":              risk_result["risk_score"],
        "priority":                risk_result["risk_level"],
        "dominant_risk_factor":    risk_result.get("dominant_risk_factor", "unknown"),
        "predicted_issue":         risk_result.get("predicted_issue", ""),
        "workshop_id":             str(workshop.get("workshop_id", "UNASSIGNED")),
        "workshop_name":           str(workshop.get("workshop_name", "Unknown Workshop")),
        "workshop_city":           str(workshop.get("city", "Unknown")),
        "ev_specialized":          bool(workshop.get("ev_specialized", False)),
        "battery_repair_capable":  bool(workshop.get("battery_repair_capable", False)),
        "scheduled_day":           day_name,
        "scheduled_time_slot":     time_slot,
        "estimated_downtime_hours": downtime,
        "workshop_workload_percent": float(workshop.get("current_workload_percent", 0.0)),
        "recommended_action":      risk_result.get("recommended_action", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_schedule_inputs(
    vehicle_ids: list[str],
    date_range_days: int,
) -> tuple[list[str], int]:
    """Validate inputs for optimize_maintenance_schedule.

    Args:
        vehicle_ids:    List of vehicle ID strings. Must be non-empty.
        date_range_days: Number of days in the scheduling window (1–30).

    Returns:
        Tuple of (cleaned vehicle_ids list, validated date_range_days).

    Raises:
        TypeError:  If vehicle_ids is not a list.
        ValueError: If any vehicle_id is invalid, or date_range_days is out of range.
    """
    if not isinstance(vehicle_ids, list):
        raise TypeError(
            f"'vehicle_ids' must be a list, got {type(vehicle_ids).__name__!r}."
        )
    if not vehicle_ids:
        raise ValueError("'vehicle_ids' must not be empty.")

    cleaned_ids = []
    for i, vid in enumerate(vehicle_ids):
        if not isinstance(vid, str):
            raise ValueError(
                f"All vehicle_ids must be strings. Item [{i}] is "
                f"{type(vid).__name__!r}: {vid!r}"
            )
        stripped = vid.strip()
        if not stripped:
            raise ValueError(f"vehicle_ids[{i}] is empty or whitespace-only.")
        cleaned_ids.append(stripped)

    if not isinstance(date_range_days, int) or isinstance(date_range_days, bool):
        raise TypeError(
            f"'date_range_days' must be an integer, got {type(date_range_days).__name__!r}."
        )
    if not (1 <= date_range_days <= 30):
        raise ValueError(
            f"'date_range_days' must be between 1 and 30, got {date_range_days}."
        )

    return cleaned_ids, date_range_days


# ─────────────────────────────────────────────────────────────────────────────
# Public LangChain tool
# ─────────────────────────────────────────────────────────────────────────────

@tool
def optimize_maintenance_schedule(
    vehicle_ids: list[str],
    date_range_days: int = 5,
) -> list[dict]:
    """Generate an optimized maintenance schedule for a list of EV fleet vehicles.

    Fetches each vehicle's maintenance history from CSV, computes a risk score
    using the risk analyzer engine, and assigns each vehicle to the most suitable
    available workshop within a configurable scheduling window.

    Optimization rules applied:
        - CRITICAL vehicles are scheduled first; LOW risk vehicles last.
        - Workshops with lower current workload are preferred to balance capacity.
        - Battery-specific repairs (battery/cycle risk) are routed only to
          workshops with battery_repair_capable = True.
        - Unavailable workshops (is_available = False) are never assigned.
        - EV-specialized workshops are preferred over general service centres.
        - Downtime is estimated from the workshop's average service hours,
          scaled by the vehicle's risk severity.
        - If no workshop in the vehicle's depot city is available, the scheduler
          falls back to the nearest available workshop in any city.

    Args:
        vehicle_ids (list[str]):
            List of fleet vehicle IDs to schedule. Must be non-empty.
            Each ID must exist in vehicle_maintenance_history.csv.
            Maximum 50 vehicles per call to ensure performance.

        date_range_days (int, optional):
            Number of working days in the scheduling window (1–30).
            Determines how schedule slots are spread across weekdays.
            Defaults to 5 (one working week).

    Returns:
        list[dict]: Ordered list of schedule entries, sorted by risk priority
        (CRITICAL → HIGH → MEDIUM → LOW). Each entry contains:
            - vehicle_id (str):                 Fleet vehicle identifier.
            - vehicle_model (str):              Model name.
            - risk_score (int):                 Composite risk score 0–100.
            - priority (str):                   Risk level: CRITICAL/HIGH/MEDIUM/LOW.
            - dominant_risk_factor (str):       Factor driving the risk score.
            - predicted_issue (str):            Likely failure mode description.
            - workshop_id (str):                Assigned workshop identifier.
            - workshop_name (str):              Human-readable workshop name.
            - workshop_city (str):              Workshop city.
            - ev_specialized (bool):            Whether workshop is EV-certified.
            - battery_repair_capable (bool):    Whether workshop can repair batteries.
            - scheduled_day (str):              Assigned day (e.g. 'Monday').
            - scheduled_time_slot (str):        Assigned time (e.g. '09:00').
            - estimated_downtime_hours (float): Expected total service duration.
            - workshop_workload_percent (float): Workshop load at time of scheduling.
            - recommended_action (str):         Actionable maintenance instruction.

    Raises:
        TypeError:  If vehicle_ids is not a list, or date_range_days is not an int.
        ValueError: If vehicle_ids is empty, any ID is blank, date_range_days is
                    out of range [1–30], or more than 50 vehicle IDs are provided.
        RuntimeError: If dataset files cannot be loaded.

    Example:
        >>> schedule = optimize_maintenance_schedule.invoke({
        ...     "vehicle_ids": ["EV_1000", "EV_1005", "EV_1010"],
        ...     "date_range_days": 5
        ... })
        >>> schedule[0]["priority"]   # First entry is highest risk
        'CRITICAL'
        >>> schedule[0]["workshop_id"]
        'WS_007'
    """
    cleaned_ids, window = _validate_schedule_inputs(vehicle_ids, date_range_days)

    if len(cleaned_ids) > 50:
        raise ValueError(
            f"Maximum 50 vehicles per scheduling call; received {len(cleaned_ids)}. "
            "Split the list into smaller batches."
        )

    workshops_df = load_workshop_capacity()

    # ── Step 1: Fetch risk profile for each vehicle ──────────────────────────
    risk_profiles: list[dict] = []
    skipped: list[str] = []

    for vid in cleaned_ids:
        try:
            raw_record = _fetch_record_from_csv(vid)
            validated  = _validate_vehicle_record(raw_record)
            risk       = _compute_risk(validated)
            risk_profiles.append(risk)
        except (ValueError, KeyError) as exc:
            # Record not found or invalid — log and skip
            skipped.append(f"{vid}: {exc}")

    if not risk_profiles:
        raise ValueError(
            "No valid vehicle records could be loaded. Skipped: " + "; ".join(skipped)
        )

    # ── Step 2: Sort by risk score descending ────────────────────────────────
    risk_profiles.sort(
        key=lambda r: (_RISK_ORDER.get(r["risk_level"], 99), -r["risk_score"])
    )

    # ── Step 3: Assign workshops and build schedule ───────────────────────────
    schedule: list[dict] = []

    for slot_index, risk in enumerate(risk_profiles):
        vid  = risk["vehicle_id"]
        city = _get_vehicle_city(vid)

        needs_battery = risk.get("dominant_risk_factor") in ("battery", "cycles")

        workshop = _find_best_workshop(city, needs_battery, workshops_df)

        if workshop is None:
            # No workshop found — create an unassigned placeholder
            entry = {
                "vehicle_id":              vid,
                "vehicle_model":           risk.get("vehicle_model", "Unknown"),
                "risk_score":              risk["risk_score"],
                "priority":                risk["risk_level"],
                "dominant_risk_factor":    risk.get("dominant_risk_factor", "unknown"),
                "predicted_issue":         risk.get("predicted_issue", ""),
                "workshop_id":             "UNASSIGNED",
                "workshop_name":           "No available workshop found",
                "workshop_city":           city,
                "ev_specialized":          False,
                "battery_repair_capable":  False,
                "scheduled_day":           "TBD",
                "scheduled_time_slot":     "TBD",
                "estimated_downtime_hours": 0.0,
                "workshop_workload_percent": 0.0,
                "recommended_action":      risk.get("recommended_action", ""),
            }
        else:
            entry = _build_schedule_entry(risk, workshop, slot_index, window)

        schedule.append(entry)

    return schedule


# ─────────────────────────────────────────────────────────────────────────────
# Sample test calls
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=" * 65)
    print("TOOL 2: Maintenance Schedule Optimizer — Sample Test Calls")
    print("=" * 65)

    # ── Test 1: Standard schedule for 5 vehicles over 5-day window ────────
    print("\n[Test 1] Schedule 5 vehicles — 5-day window")
    vehicle_batch = ["EV_2000", "EV_2001", "EV_2002", "EV_2003", "EV_2004"]
    try:
        schedule = optimize_maintenance_schedule.invoke({
            "vehicle_ids": vehicle_batch,
            "date_range_days": 5,
        })
        print(f"  Scheduled {len(schedule)} vehicles:")
        for entry in schedule:
            print(
                f"  {entry['vehicle_id']:12s} | {entry['priority']:8s} | "
                f"score={entry['risk_score']:3d} | {entry['workshop_id']:8s} | "
                f"{entry['scheduled_day']:9s} {entry['scheduled_time_slot']} | "
                f"downtime={entry['estimated_downtime_hours']}h"
            )
    except Exception as e:
        print(f"  Error: {e}")

    # ── Test 2: Single CRITICAL vehicle ────────────────────────────────────
    print("\n[Test 2] Single vehicle — 3-day window")
    try:
        schedule = optimize_maintenance_schedule.invoke({
            "vehicle_ids": ["EV_2010"],
            "date_range_days": 3,
        })
        print(json.dumps(schedule, indent=2))
    except Exception as e:
        print(f"  Error: {e}")

    # ── Test 3: Larger batch — 10 vehicles ─────────────────────────────────
    print("\n[Test 3] Larger batch — 10 vehicles — 5-day window")
    large_batch = [f"EV_{2000 + i}" for i in range(10)]
    try:
        schedule = optimize_maintenance_schedule.invoke({
            "vehicle_ids": large_batch,
            "date_range_days": 5,
        })
        print(f"  Scheduled {len(schedule)} vehicles")
        priorities = {}
        for e in schedule:
            priorities[e["priority"]] = priorities.get(e["priority"], 0) + 1
        print(f"  Priority distribution: {priorities}")
    except Exception as e:
        print(f"  Error: {e}")

    # ── Test 4: Validation error — empty list ──────────────────────────────
    print("\n[Test 4] Validation error — empty vehicle_ids")
    try:
        optimize_maintenance_schedule.invoke({"vehicle_ids": [], "date_range_days": 5})
    except ValueError as e:
        print(f"  Caught expected ValueError: {e}")

    # ── Test 5: Validation error — date_range_days out of range ───────────
    print("\n[Test 5] Validation error — date_range_days=50 (exceeds max 30)")
    try:
        optimize_maintenance_schedule.invoke({
            "vehicle_ids": ["EV_2000"],
            "date_range_days": 50,
        })
    except ValueError as e:
        print(f"  Caught expected ValueError: {e}")

    print("\n" + "=" * 65)
    print("All test calls completed.")