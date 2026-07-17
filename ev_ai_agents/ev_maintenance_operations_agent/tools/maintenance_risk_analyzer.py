"""
maintenance_risk_analyzer.py
=============================
Tool 1 of the Maintenance Operations Optimiser feature.

Purpose:
    Analyze an EV fleet vehicle maintenance record and produce a risk assessment
    that identifies maintenance priority level, predicted failure issue, and the
    recommended corrective action.

Scoring model:
    A 100-point composite risk score is computed from six weighted factors:

    Factor                  Weight  Degradation Thresholds
    ─────────────────────── ──────  ─────────────────────────────────────────
    Battery health (%)         30   <60 → 30 | <70 → 20 | <80 → 10 | <90 → 5
    Vehicle age (years)        20   >8 → 20  | >5 → 12  | >3 → 6   | >1 → 2
    Total km driven            20   >200k→20 | >120k→12 | >60k→6   | >20k→2
    Charging cycles            15   >1500→15 | >1000→10 | >500→5
    Fault code severity        10   P0→10    | P1→7     | P2→3     | NONE→0
    Days since last service     5   >365→5   | >180→3   | >90→1

Risk levels:
    0–25   → LOW
    26–50  → MEDIUM
    51–75  → HIGH
    76–100 → CRITICAL

Integrates with:
    - datasets/vehicle_maintenance_history.csv
    - Downstream: maintenance_schedule_optimizer.py (consumes risk output)

FastAPI ready:
    The @tool-decorated public function accepts a dict payload compatible
    with a Pydantic BaseModel schema and returns a serializable dict.
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, date
from typing import Any

# Allow direct execution and imports from sibling modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.tools import tool

from utils import (
    load_maintenance_history,
    validate_string,
    validate_float,
    validate_positive_int,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Fault code severity mapping  (prefix → severity label, score contribution)
_FAULT_SEVERITY: dict[str, tuple[str, int]] = {
    "P0": ("Critical",  10),
    "P1": ("Major",      7),
    "P2": ("Minor",      3),
}

# Predicted issue catalogue keyed by highest-scoring factor
_ISSUE_CATALOGUE: dict[str, str] = {
    "battery":   "Battery cell degradation or BMS fault — capacity loss risk",
    "age":       "Age-related component wear — brake, suspension, or connector fatigue",
    "mileage":   "High-mileage wear — motor brushes, tyre, and drive-train components",
    "cycles":    "Charging system degradation — port corrosion or DC converter strain",
    "fault":     "Active fault code — immediate diagnostic inspection required",
    "overdue":   "Service overdue — undetected wear accumulation risk",
    "none":      "No dominant risk factor — preventive check recommended",
}

_RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


# ─────────────────────────────────────────────────────────────────────────────
# Private scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_battery(health_pct: float) -> tuple[int, str]:
    """Compute battery-health risk contribution.

    Args:
        health_pct: Battery state-of-health percentage (0–100).

    Returns:
        Tuple of (points_awarded, factor_key).
    """
    if health_pct < 60:
        return 30, "battery"
    if health_pct < 70:
        return 20, "battery"
    if health_pct < 80:
        return 10, "battery"
    if health_pct < 90:
        return 5, "battery"
    return 0, "none"


def _score_age(age_years: float) -> tuple[int, str]:
    """Compute vehicle age risk contribution.

    Args:
        age_years: Vehicle age in decimal years.

    Returns:
        Tuple of (points_awarded, factor_key).
    """
    if age_years > 8:
        return 20, "age"
    if age_years > 5:
        return 12, "age"
    if age_years > 3:
        return 6, "age"
    if age_years > 1:
        return 2, "age"
    return 0, "none"


def _score_mileage(total_km: float) -> tuple[int, str]:
    """Compute total mileage risk contribution.

    Args:
        total_km: Cumulative kilometres driven.

    Returns:
        Tuple of (points_awarded, factor_key).
    """
    if total_km > 200_000:
        return 20, "mileage"
    if total_km > 120_000:
        return 12, "mileage"
    if total_km > 60_000:
        return 6, "mileage"
    if total_km > 20_000:
        return 2, "mileage"
    return 0, "none"


def _score_cycles(cycles: int) -> tuple[int, str]:
    """Compute charging-cycle risk contribution.

    Args:
        cycles: Total battery charge cycles completed.

    Returns:
        Tuple of (points_awarded, factor_key).
    """
    if cycles > 1500:
        return 15, "cycles"
    if cycles > 1000:
        return 10, "cycles"
    if cycles > 500:
        return 5, "cycles"
    return 0, "none"


def _score_fault(fault_code: str) -> tuple[int, str]:
    """Compute fault code severity risk contribution.

    Args:
        fault_code: Fault code string (e.g. 'P0_MOTOR_FAILURE', 'NONE').

    Returns:
        Tuple of (points_awarded, factor_key).
    """
    code_upper = fault_code.strip().upper()
    if code_upper in ("NONE", "", "NO_FAULT"):
        return 0, "none"
    for prefix, (_, pts) in _FAULT_SEVERITY.items():
        if code_upper.startswith(prefix):
            return pts, "fault"
    # Unknown fault code — treat as minor
    return 3, "fault"


def _score_service_overdue(days_since_service: int) -> tuple[int, str]:
    """Compute service-overdue risk contribution.

    Args:
        days_since_service: Number of days elapsed since the last service.

    Returns:
        Tuple of (points_awarded, factor_key).
    """
    if days_since_service > 365:
        return 5, "overdue"
    if days_since_service > 180:
        return 3, "overdue"
    if days_since_service > 90:
        return 1, "overdue"
    return 0, "none"


def _map_risk_level(score: int) -> str:
    """Map a numeric risk score to a risk level label.

    Args:
        score: Integer risk score in range [0, 100].

    Returns:
        One of: 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'.
    """
    if score >= 76:
        return "CRITICAL"
    if score >= 51:
        return "HIGH"
    if score >= 26:
        return "MEDIUM"
    return "LOW"


def _recommended_action(risk_level: str, dominant_factor: str, fault_code: str) -> str:
    """Generate a human-readable recommended action.

    Args:
        risk_level:       One of LOW / MEDIUM / HIGH / CRITICAL.
        dominant_factor:  The factor that contributed most to the score.
        fault_code:       Raw fault code string from the vehicle record.

    Returns:
        A concise action recommendation string.
    """
    fault_upper = fault_code.strip().upper()
    has_fault = fault_upper not in ("NONE", "", "NO_FAULT")

    if risk_level == "CRITICAL":
        if has_fault:
            return (
                f"IMMEDIATE workshop booking required. Active fault detected: {fault_code}. "
                "Suspend vehicle from operational duty until diagnostic is cleared."
            )
        return (
            "Schedule emergency maintenance within 24–48 hours. "
            "Vehicle should not be assigned long-distance routes."
        )
    if risk_level == "HIGH":
        if dominant_factor == "battery":
            return "Book battery health inspection within 7 days. Run full BMS diagnostic cycle."
        if dominant_factor == "mileage":
            return "Schedule full service check — motor, tyres, drive-train — within 14 days."
        if dominant_factor == "overdue":
            return "Service is significantly overdue. Book workshop slot within 7 days."
        return "Schedule preventive maintenance within 7–14 days. Reduce route load if possible."
    if risk_level == "MEDIUM":
        return (
            "Add to next scheduled maintenance cycle. "
            "Monitor battery health and fault codes weekly."
        )
    return (
        "No immediate action required. "
        "Continue standard maintenance interval. Next service check in 90 days."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_vehicle_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise all fields in a vehicle maintenance record.

    Required keys and their types:
        vehicle_id            (str)   — unique vehicle identifier
        vehicle_model         (str)   — model name
        battery_health_percent (float) — state-of-health, 0–100
        vehicle_age_years     (float) — age in decimal years, >= 0
        total_km_driven       (float) — cumulative km, >= 0
        charging_cycles       (int)   — total charge cycles, >= 0
        fault_code            (str)   — fault code or 'NONE'
        last_service_date     (str)   — ISO date string 'YYYY-MM-DD'

    Args:
        record: Raw input dict from the caller.

    Returns:
        A validated, normalised copy of the record with a computed
        'days_since_service' integer key added.

    Raises:
        TypeError:  If record is not a dict.
        KeyError:   If a required key is missing.
        ValueError: If any field fails validation.
    """
    if not isinstance(record, dict):
        raise TypeError(
            f"vehicle_record must be a dict, got {type(record).__name__!r}."
        )

    required = [
        "vehicle_id", "vehicle_model", "battery_health_percent",
        "vehicle_age_years", "total_km_driven", "charging_cycles",
        "fault_code", "last_service_date",
    ]
    for key in required:
        if key not in record:
            raise KeyError(
                f"Required key '{key}' is missing from vehicle_record. "
                f"Required keys: {required}"
            )

    validated: dict[str, Any] = {}
    validated["vehicle_id"]     = validate_string(record["vehicle_id"],     "vehicle_id")
    validated["vehicle_model"]  = validate_string(record["vehicle_model"],  "vehicle_model")

    validated["battery_health_percent"] = validate_float(
        record["battery_health_percent"], "battery_health_percent", min_val=0.0, max_val=100.0
    )
    validated["vehicle_age_years"] = validate_float(
        record["vehicle_age_years"], "vehicle_age_years", min_val=0.0
    )
    validated["total_km_driven"] = validate_float(
        record["total_km_driven"], "total_km_driven", min_val=0.0
    )

    cycles_raw = record["charging_cycles"]
    if not isinstance(cycles_raw, (int, float)) or isinstance(cycles_raw, bool):
        raise ValueError("'charging_cycles' must be a non-negative integer.")
    validated["charging_cycles"] = int(cycles_raw)
    if validated["charging_cycles"] < 0:
        raise ValueError("'charging_cycles' must be >= 0.")

    validated["fault_code"] = validate_string(record["fault_code"], "fault_code")

    # Parse last_service_date and compute days_since_service
    date_raw = record["last_service_date"]
    if isinstance(date_raw, str):
        try:
            svc_date = datetime.strptime(date_raw.strip(), "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(
                f"'last_service_date' must be in 'YYYY-MM-DD' format, got: {date_raw!r}"
            )
    elif isinstance(date_raw, (datetime, date)):
        svc_date = date_raw if isinstance(date_raw, date) else date_raw.date()
    else:
        raise ValueError(
            f"'last_service_date' must be a string or date, got {type(date_raw).__name__!r}."
        )

    today = datetime.today().date()
    if svc_date > today:
        raise ValueError(
            f"'last_service_date' ({date_raw}) cannot be in the future."
        )
    validated["last_service_date"]   = svc_date.strftime("%Y-%m-%d")
    validated["days_since_service"]  = (today - svc_date).days

    return validated


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_risk(validated_record: dict[str, Any]) -> dict[str, Any]:
    """Compute the risk assessment from a validated vehicle record.

    Args:
        validated_record: Output of _validate_vehicle_record().

    Returns:
        Risk assessment dict with keys:
            vehicle_id, risk_score, risk_level,
            score_breakdown, predicted_issue, recommended_action.
    """
    vid    = validated_record["vehicle_id"]
    health = validated_record["battery_health_percent"]
    age    = validated_record["vehicle_age_years"]
    km     = validated_record["total_km_driven"]
    cycles = validated_record["charging_cycles"]
    fault  = validated_record["fault_code"]
    days   = validated_record["days_since_service"]

    # Score each factor
    bat_pts,  bat_factor  = _score_battery(health)
    age_pts,  age_factor  = _score_age(age)
    km_pts,   km_factor   = _score_mileage(km)
    cyc_pts,  cyc_factor  = _score_cycles(cycles)
    flt_pts,  flt_factor  = _score_fault(fault)
    svc_pts,  svc_factor  = _score_service_overdue(days)

    total_score = bat_pts + age_pts + km_pts + cyc_pts + flt_pts + svc_pts
    # Clamp to [0, 100]
    total_score = max(0, min(100, total_score))

    # Identify dominant factor (highest contributor)
    factor_scores = {
        bat_factor: bat_pts,
        age_factor: age_pts,
        km_factor:  km_pts,
        cyc_factor: cyc_pts,
        flt_factor: flt_pts,
        svc_factor: svc_pts,
    }
    # Pick the factor with the highest points (fault always wins ties)
    dominant = max(
        factor_scores,
        key=lambda f: (factor_scores[f], 1 if f == "fault" else 0)
    )
    if dominant == "none":
        dominant = "none"

    risk_level  = _map_risk_level(total_score)
    issue_desc  = _ISSUE_CATALOGUE.get(dominant, _ISSUE_CATALOGUE["none"])
    action      = _recommended_action(risk_level, dominant, fault)

    return {
        "vehicle_id":          vid,
        "vehicle_model":       validated_record["vehicle_model"],
        "risk_score":          total_score,
        "risk_level":          risk_level,
        "score_breakdown": {
            "battery_health_score":   bat_pts,
            "vehicle_age_score":      age_pts,
            "mileage_score":          km_pts,
            "charging_cycle_score":   cyc_pts,
            "fault_severity_score":   flt_pts,
            "service_overdue_score":  svc_pts,
        },
        "dominant_risk_factor":  dominant,
        "predicted_issue":       issue_desc,
        "recommended_action":    action,
        "days_since_service":    days,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Optional: CSV-based lookup helper (for batch pipeline use)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_record_from_csv(vehicle_id: str) -> dict[str, Any]:
    """Look up a vehicle maintenance record from the history CSV by vehicle_id.

    Args:
        vehicle_id: Validated vehicle ID string.

    Returns:
        A dict matching the expected vehicle_record schema.

    Raises:
        ValueError: If the vehicle_id is not found in the dataset.
    """
    df = load_maintenance_history()
    matches = df[df["vehicle_id"].astype(str).str.strip() == vehicle_id]
    if matches.empty:
        available = ", ".join(df["vehicle_id"].astype(str).tolist()[:10]) + "…"
        raise ValueError(
            f"vehicle_id='{vehicle_id}' not found in vehicle_maintenance_history.csv. "
            f"Sample IDs: {available}"
        )
    row = matches.iloc[0]
    return {
        "vehicle_id":              str(row["vehicle_id"]),
        "vehicle_model":           str(row["vehicle_model"]),
        "battery_health_percent":  float(row["battery_health_percent"]),
        "vehicle_age_years":       float(row["vehicle_age_years"]),
        "total_km_driven":         float(row["total_km_driven"]),
        "charging_cycles":         int(row["charging_cycles"]),
        "fault_code":              str(row["fault_code"]),
        "last_service_date":       str(row["last_service_date"])[:10],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public LangChain tool — dict input (for agent call)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def analyze_maintenance_risk(vehicle_record: dict) -> dict:
    """Analyze an EV fleet vehicle and return a maintenance risk assessment.

    Evaluates battery health, vehicle age, cumulative mileage, charging cycles,
    active fault codes, and service history to compute a composite risk score.
    The score drives maintenance scheduling priority within the Maintenance
    Operations Optimiser.

    Scoring model (100-point scale):
        - Battery health degradation  : up to 30 points
        - Vehicle age                 : up to 20 points
        - Total km driven             : up to 20 points
        - Charging cycles             : up to 15 points
        - Fault code severity         : up to 10 points
        - Days since last service     : up to  5 points

    Risk levels:
        0–25   → LOW       (routine monitoring, next scheduled service)
        26–50  → MEDIUM    (flag for next maintenance cycle)
        51–75  → HIGH      (schedule within 7–14 days)
        76–100 → CRITICAL  (immediate action, suspend from long routes)

    Args:
        vehicle_record (dict): A dict containing the following keys:
            - vehicle_id (str):              Unique fleet vehicle identifier.
            - vehicle_model (str):           Vehicle model name.
            - battery_health_percent (float): State-of-health, range 0–100.
            - vehicle_age_years (float):     Age in decimal years, >= 0.
            - total_km_driven (float):       Cumulative km driven, >= 0.
            - charging_cycles (int):         Total full charge cycles completed, >= 0.
            - fault_code (str):              Active fault code (e.g. 'P0_MOTOR_FAILURE')
                                             or 'NONE' if no active fault.
            - last_service_date (str):       ISO date of last service 'YYYY-MM-DD'.

    Returns:
        dict: Risk assessment containing:
            - vehicle_id (str):            Echo of the input identifier.
            - vehicle_model (str):         Echo of the model name.
            - risk_score (int):            Composite score 0–100 (higher = more risk).
            - risk_level (str):            One of: 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'.
            - score_breakdown (dict):      Per-factor point contributions.
            - dominant_risk_factor (str):  The factor contributing most to the score.
            - predicted_issue (str):       Description of the most likely failure mode.
            - recommended_action (str):    Concrete next-step recommendation.
            - days_since_service (int):    Computed days since last_service_date.

    Raises:
        TypeError:  If vehicle_record is not a dict.
        KeyError:   If a required key is missing from vehicle_record.
        ValueError: If any field fails type or range validation.

    Example:
        >>> record = {
        ...     "vehicle_id": "EV_1001",
        ...     "vehicle_model": "Tata Nexon EV",
        ...     "battery_health_percent": 64.0,
        ...     "vehicle_age_years": 6.5,
        ...     "total_km_driven": 145000.0,
        ...     "charging_cycles": 1200,
        ...     "fault_code": "P1_BMS_FAULT",
        ...     "last_service_date": "2025-09-01",
        ... }
        >>> result = analyze_maintenance_risk.invoke({"vehicle_record": record})
        >>> result["risk_level"]
        'CRITICAL'
    """
    validated = _validate_vehicle_record(vehicle_record)
    return _compute_risk(validated)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function: risk from vehicle_id (CSV lookup)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_risk_by_id(vehicle_id: str) -> dict:
    """Look up a vehicle by ID from the maintenance CSV and run risk analysis.

    This is a convenience wrapper for batch processing and pipeline use.
    It is NOT a @tool — use analyze_maintenance_risk for agent/API calls.

    Args:
        vehicle_id: The vehicle ID to look up (e.g. 'EV_1003').

    Returns:
        Risk assessment dict (same schema as analyze_maintenance_risk output).

    Raises:
        ValueError: If the vehicle_id is not found in the CSV.
    """
    vid = vehicle_id.strip()
    record = _fetch_record_from_csv(vid)
    validated = _validate_vehicle_record(record)
    return _compute_risk(validated)


# ─────────────────────────────────────────────────────────────────────────────
# Sample test calls
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=" * 65)
    print("TOOL 1: Maintenance Risk Analyzer — Sample Test Calls")
    print("=" * 65)

    # ── Test 1: CRITICAL risk — old vehicle, bad battery, active P0 fault ──
    print("\n[Test 1] CRITICAL risk vehicle")
    record_critical = {
        "vehicle_id":              "EV_TEST_001",
        "vehicle_model":           "Eicher EV Cargo",
        "battery_health_percent":  55.0,
        "vehicle_age_years":       9.2,
        "total_km_driven":         215000.0,
        "charging_cycles":         1650,
        "fault_code":              "P0_MOTOR_FAILURE",
        "last_service_date":       "2025-06-01",
    }
    result = analyze_maintenance_risk.invoke({"vehicle_record": record_critical})
    print(json.dumps(result, indent=2))

    # ── Test 2: HIGH risk — older vehicle, degraded battery, overdue ──────
    print("\n[Test 2] HIGH risk vehicle")
    record_high = {
        "vehicle_id":              "EV_TEST_002",
        "vehicle_model":           "MG ZS EV",
        "battery_health_percent":  72.0,
        "vehicle_age_years":       6.0,
        "total_km_driven":         130000.0,
        "charging_cycles":         950,
        "fault_code":              "NONE",
        "last_service_date":       "2025-10-15",
    }
    result = analyze_maintenance_risk.invoke({"vehicle_record": record_high})
    print(json.dumps(result, indent=2))

    # ── Test 3: MEDIUM risk — moderate age, acceptable battery ───────────
    print("\n[Test 3] MEDIUM risk vehicle")
    record_medium = {
        "vehicle_id":              "EV_TEST_003",
        "vehicle_model":           "Tata Nexon EV",
        "battery_health_percent":  83.0,
        "vehicle_age_years":       4.0,
        "total_km_driven":         70000.0,
        "charging_cycles":         520,
        "fault_code":              "P2_THERMAL_WARNING",
        "last_service_date":       "2026-03-10",
    }
    result = analyze_maintenance_risk.invoke({"vehicle_record": record_medium})
    print(json.dumps(result, indent=2))

    # ── Test 4: LOW risk — new vehicle, healthy battery, no faults ────────
    print("\n[Test 4] LOW risk vehicle")
    record_low = {
        "vehicle_id":              "EV_TEST_004",
        "vehicle_model":           "BYD Atto 3",
        "battery_health_percent":  96.0,
        "vehicle_age_years":       0.8,
        "total_km_driven":         12000.0,
        "charging_cycles":         85,
        "fault_code":              "NONE",
        "last_service_date":       "2026-05-20",
    }
    result = analyze_maintenance_risk.invoke({"vehicle_record": record_low})
    print(json.dumps(result, indent=2))

    # ── Test 5: CSV lookup by vehicle_id ──────────────────────────────────
    print("\n[Test 5] CSV lookup — analyze_risk_by_id('EV_2005')")
    try:
        result = analyze_risk_by_id("EV_2005")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"  Error: {e}")

    # ── Test 6: Validation error — missing key ────────────────────────────
    print("\n[Test 6] Validation error — missing 'fault_code' key")
    try:
        bad_record = {
            "vehicle_id":              "EV_BAD",
            "vehicle_model":           "Unknown Model",
            "battery_health_percent":  80.0,
            "vehicle_age_years":       3.0,
            "total_km_driven":         50000.0,
            "charging_cycles":         300,
            # fault_code intentionally omitted
            "last_service_date":       "2026-01-01",
        }
        analyze_maintenance_risk.invoke({"vehicle_record": bad_record})
    except KeyError as e:
        print(f"  Caught expected KeyError: {e}")
    except Exception as e:
        print(f"  Caught unexpected error ({type(e).__name__}): {e}")

    print("\n" + "=" * 65)
    print("All test calls completed.")