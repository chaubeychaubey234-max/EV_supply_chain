import os
import pandas as pd

# Resolve paths relative to this file's location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")

def get_csv_path(filename: str) -> str:
    """Return the absolute path for a dataset file."""
    return os.path.join(DATASETS_DIR, filename)

def load_supply_chain() -> pd.DataFrame:
    """Load the EV supply chain mapping dataset."""
    return pd.read_csv(get_csv_path("ev_supply_chain.csv"))

def load_minerals_risk() -> pd.DataFrame:
    """Load the critical minerals geopolitical risk dataset."""
    return pd.read_csv(get_csv_path("critical_minerals_risk.csv"))

def load_battery_quality() -> pd.DataFrame:
    """Load the battery manufacturing quality dataset."""
    return pd.read_csv(get_csv_path("battery_quality.csv"))

# --- Helper Queries for Agents ---

def query_supplier_demographics(supplier_id: str) -> dict:
    """Query demographics for a supplier from ev_supply_chain.csv."""
    df = load_supply_chain()
    sub = df[df["supplier_id"].str.upper() == supplier_id.upper()]
    if sub.empty:
        return {}
    row = sub.iloc[0]
    return {
        "supplier_id": supplier_id.upper(),
        "supplier_name": row["supplier_name"],
        "country": row["country"],
        "supplier_tier": int(row["supplier_tier"]),
        "materials": list(sub["material"].unique()),
        "battery_types": list(sub["battery_type"].unique())
    }

def query_material_traceability(batch_id: str) -> list:
    """Retrieve full batch lineage from ev_supply_chain.csv."""
    df = load_supply_chain()
    sub = df[df["batch_id"].str.upper() == batch_id.upper()]
    if sub.empty:
        return []
    records = []
    for _, row in sub.iterrows():
        records.append({
            "cell_id": row["cell_id"],
            "pack_id": row["pack_id"],
            "vehicle_id": row["vehicle_id"],
            "material": row["material"]
        })
    return records

def query_geopolitical_mineral_risk(material: str, country: str) -> dict:
    """Retrieve risk profile for a specific mineral and country from critical_minerals_risk.csv."""
    df = load_minerals_risk()
    sub = df[
        (df["material"].str.lower() == material.lower()) & 
        (df["country"].str.lower() == country.lower())
    ]
    if sub.empty:
        return {}
    row = sub.iloc[0]
    return {
        "material": row["material"],
        "country": row["country"],
        "production_share": float(row["production_share"]),
        "global_supply_percentage": float(row["global_supply_percentage"]),
        "dependency_score": float(row["dependency_score"]),
        "political_risk_score": float(row["political_risk_score"]),
        "export_dependency": float(row["export_dependency"]),
        "risk_level": row["risk_level"]
    }

def query_batch_quality(batch_id: str) -> dict:
    """Retrieve quality data for a material batch from battery_quality.csv."""
    df = load_battery_quality()
    sub = df[df["batch_id"].str.upper() == batch_id.upper()]
    if sub.empty:
        return {}
    row = sub.iloc[0]
    return {
        "batch_id": batch_id.upper(),
        "supplier_id": row["supplier_id"],
        "inspection_count": int(row["inspection_count"]),
        "defects_found": int(row["defects_found"]),
        "defect_rate": float(row["defect_rate"]),
        "defect_type": row["defect_type"]
    }
