"""
utils.py — Shared utilities for Maintenance Operations Optimiser tools.

Provides:
- Project path resolution (BASE_DIR, dataset paths)
- Lazy-loading, cached dataset loaders
- Common validation helpers reused across the three tools
"""

from __future__ import annotations

import os
import functools
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────

# This file lives at:
#   features/maintenance_operations_optimiser/tools/utils.py
# Project root is four levels up.
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = _THIS_DIR                                          # …/tools/
_MOO_DIR   = os.path.dirname(_TOOLS_DIR)                       # …/maintenance_operations_optimiser/
_FEAT_DIR  = os.path.dirname(_MOO_DIR)                         # …/features/
BASE_DIR   = os.path.dirname(_FEAT_DIR)                        # project root

DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
LOCAL_DATASET_DIR = os.path.join(_MOO_DIR, "dataset")


def _dataset_path(filename: str) -> str:
    """Return absolute path to a raw dataset file."""
    return os.path.join(DATASETS_DIR, filename)


def _local_dataset_path(filename: str) -> str:
    """Return absolute path to a maintenance-optimised dataset file."""
    return os.path.join(LOCAL_DATASET_DIR, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loading cached dataset loaders
# ─────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def load_maintenance_history() -> pd.DataFrame:
    """Load vehicle_maintenance_history.csv (cached after first call).

    Returns:
        DataFrame with one row per vehicle maintenance record.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        RuntimeError: If the file cannot be parsed as CSV.
    """
    path = _local_dataset_path("vehicle_maintenance_history.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"vehicle_maintenance_history.csv not found at: {path}\n"
            "Generate the dataset or place it in the local dataset/ directory."
        )
    try:
        df = pd.read_csv(path, parse_dates=["last_service_date"])
        return df
    except Exception as exc:
        raise RuntimeError(f"Failed to load vehicle_maintenance_history.csv: {exc}") from exc


@functools.lru_cache(maxsize=None)
def load_workshop_capacity() -> pd.DataFrame:
    """Load workshop_capacity.csv (cached after first call).

    Returns:
        DataFrame with one row per workshop.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        RuntimeError: If the file cannot be parsed as CSV.
    """
    path = _local_dataset_path("workshop_capacity.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"workshop_capacity.csv not found at: {path}\n"
            "Generate the dataset or place it in the local dataset/ directory."
        )
    try:
        df = pd.read_csv(path)
        # Normalise boolean columns
        for col in ["is_available", "ev_specialized", "battery_repair_capable"]:
            if col in df.columns:
                df[col] = df[col].map(
                    {"True": True, "False": False, True: True, False: False}
                ).fillna(False).astype(bool)
        return df
    except Exception as exc:
        raise RuntimeError(f"Failed to load workshop_capacity.csv: {exc}") from exc


@functools.lru_cache(maxsize=None)
def load_fleet_operations() -> pd.DataFrame:
    """Load fleet_operations_maintenance.csv (cached after first call).

    Returns:
        DataFrame with one row per fleet vehicle.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        RuntimeError: If the file cannot be parsed as CSV.
    """
    path = _local_dataset_path("fleet_operations_maintenance.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"fleet_operations_maintenance.csv not found at: {path}\n"
            "Run the cleaning pipeline first."
        )
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load fleet_operations_maintenance.csv: {exc}") from exc


@functools.lru_cache(maxsize=None)
def load_charging_stations() -> pd.DataFrame:
    """Load charging_station_maint.csv (cached after first call).

    Returns:
        DataFrame with one row per charging station.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        RuntimeError: If the file cannot be parsed as CSV.
    """
    path = _local_dataset_path("charging_station_maint.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"charging_station_maint.csv not found at: {path}\n"
            "Run the cleaning pipeline first."
        )
    try:
        df = pd.read_csv(path)
        for col in ["is_fast_dc", "fast_charger_flag"]:
            if col in df.columns:
                df[col] = df[col].map(
                    {"True": True, "False": False, True: True, False: False}
                ).fillna(False).astype(bool)
        return df
    except Exception as exc:
        raise RuntimeError(f"Failed to load charging_station_maint.csv: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Shared validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def validate_string(value: object, field_name: str) -> str:
    """Validate that a value is a non-empty string.

    Args:
        value: The raw input value.
        field_name: Name of the field (used in error messages).

    Returns:
        Stripped, non-empty string.

    Raises:
        ValueError: If value is not a string or is empty/whitespace.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"'{field_name}' must be a string, got {type(value).__name__!r}."
        )
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"'{field_name}' must not be empty or whitespace-only.")
    return stripped


def validate_float(value: object, field_name: str, min_val: float = None,
                   max_val: float = None) -> float:
    """Validate that a value is a finite float within an optional range.

    Args:
        value: The raw input value (int or float accepted).
        field_name: Name of the field (used in error messages).
        min_val: Optional inclusive lower bound.
        max_val: Optional inclusive upper bound.

    Returns:
        Validated float.

    Raises:
        ValueError: If the value fails any check.
    """
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"'{field_name}' must be numeric, got {type(value).__name__!r}."
        )
    fval = float(value)
    if fval != fval:  # NaN check
        raise ValueError(f"'{field_name}' must not be NaN.")
    if min_val is not None and fval < min_val:
        raise ValueError(
            f"'{field_name}' must be >= {min_val}, got {fval}."
        )
    if max_val is not None and fval > max_val:
        raise ValueError(
            f"'{field_name}' must be <= {max_val}, got {fval}."
        )
    return fval


def validate_positive_int(value: object, field_name: str) -> int:
    """Validate that a value is a positive integer.

    Args:
        value: The raw input value.
        field_name: Name of the field (used in error messages).

    Returns:
        Validated positive integer.

    Raises:
        ValueError: If the value is not a positive integer.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            f"'{field_name}' must be an integer, got {type(value).__name__!r}."
        )
    if value <= 0:
        raise ValueError(f"'{field_name}' must be > 0, got {value}.")
    return value