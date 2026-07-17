import os
import pandas as pd
from langchain.tools import tool
from . import ToolError


# ---------------------------------------------------------------------------
# Load actual datasets from the datasets/ directory
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASETS_DIR = os.path.join(_BASE_DIR, "..", "..", "datasets")

_supply_chain_df = pd.read_csv(os.path.join(_DATASETS_DIR, "ev_supply_chain.csv"))
_minerals_risk_df = pd.read_csv(os.path.join(_DATASETS_DIR, "critical_minerals_risk.csv"))
_battery_quality_df = pd.read_csv(os.path.join(_DATASETS_DIR, "battery_quality.csv"))


def _build_supplier_db() -> dict:
    """Build supplier profiles from ev_supply_chain.csv."""
    db = {}
    for sup_id, group in _supply_chain_df.groupby("supplier_id"):
        row = group.iloc[0]
        materials = group["material"].unique().tolist()
        db[sup_id] = {
            "supplier_name": row["supplier_name"],
            "country": row["country"],
            "tier": int(row["supplier_tier"]),
            "materials_supplied": materials,
            "battery_types": group["battery_type"].unique().tolist(),
        }
    return db


_SUPPLIER_DB = _build_supplier_db()


# ---------------------------------------------------------------------------
# Region mapping for geography tool
# ---------------------------------------------------------------------------
_REGION_MAP = {
    "China": "Asia-Pacific",
    "South Korea": "Asia-Pacific",
    "Japan": "Asia-Pacific",
    "Australia": "Asia-Pacific",
    "Belgium": "Europe",
    "Germany": "Europe",
    "Finland": "Europe",
    "Russia": "Europe",
    "Congo": "Africa",
    "South Africa": "Africa",
    "Chile": "South America",
    "Argentina": "South America",
    "USA": "North America",
    "Canada": "North America",
}


@tool
def get_supplier_profile(supplier_id: str) -> dict:
    """Retrieve the full profile for an EV supply-chain supplier.

    Given a supplier ID (e.g. 'SUP-001'), returns structured data including
    the supplier's name, country of operation, supply-chain tier, materials
    supplied, and battery types produced.

    Use this tool when you need comprehensive supplier information for
    due-diligence, onboarding checks, or risk assessment inputs.
    """
    try:
        if not supplier_id or not isinstance(supplier_id, str):
            raise ToolError("supplier_id must be a non-empty string")

        profile = _SUPPLIER_DB.get(supplier_id.upper())
        if profile is None:
            raise ToolError(
                f"Supplier '{supplier_id}' not found. "
                f"Valid IDs: {', '.join(_SUPPLIER_DB.keys())}"
            )

        return {
            "supplier_id": supplier_id.upper(),
            "supplier_name": profile["supplier_name"],
            "country": profile["country"],
            "tier": profile["tier"],
            "materials_supplied": profile["materials_supplied"],
            "battery_types": profile["battery_types"],
            "status": "active",
        }
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Failed to fetch supplier profile: {exc}") from exc


@tool
def get_supplier_tier(supplier_id: str) -> dict:
    """Return the supply-chain tier classification for a given supplier.

    Tier 1 suppliers deliver directly to OEMs (e.g. cell/pack manufacturers).
    Tier 2+ suppliers provide raw or intermediate materials upstream.

    Use this to understand a supplier's position in the multi-tier EV
    battery supply chain.
    """
    try:
        if not supplier_id or not isinstance(supplier_id, str):
            raise ToolError("supplier_id must be a non-empty string")

        profile = _SUPPLIER_DB.get(supplier_id.upper())
        if profile is None:
            raise ToolError(
                f"Supplier '{supplier_id}' not found. "
                f"Valid IDs: {', '.join(_SUPPLIER_DB.keys())}"
            )

        tier = profile["tier"]
        tier_label = {
            1: "Tier 1 – Direct OEM Supplier",
            2: "Tier 2 – Upstream Raw Material Supplier",
            3: "Tier 3 – Sub-component Supplier",
        }.get(tier, f"Tier {tier}")

        return {
            "supplier_id": supplier_id.upper(),
            "supplier_name": profile["supplier_name"],
            "tier": tier,
            "tier_label": tier_label,
            "materials_supplied": profile["materials_supplied"],
        }
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Failed to fetch supplier tier: {exc}") from exc


@tool
def get_supplier_geography(supplier_id: str) -> dict:
    """Return the geographic details for a supplier, including country
    and the associated sourcing region classification.

    Regions are mapped as:
      Asia-Pacific, Europe, Africa, South America, North America.

    Use this when evaluating geographic diversification or geopolitical
    exposure within the supply chain.
    """
    try:
        if not supplier_id or not isinstance(supplier_id, str):
            raise ToolError("supplier_id must be a non-empty string")

        profile = _SUPPLIER_DB.get(supplier_id.upper())
        if profile is None:
            raise ToolError(
                f"Supplier '{supplier_id}' not found. "
                f"Valid IDs: {', '.join(_SUPPLIER_DB.keys())}"
            )

        country = profile["country"]

        return {
            "supplier_id": supplier_id.upper(),
            "supplier_name": profile["supplier_name"],
            "country": country,
            "region": _REGION_MAP.get(country, "Unknown"),
            "materials_supplied": profile["materials_supplied"],
            "tier": profile["tier"],
        }
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Failed to fetch supplier geography: {exc}") from exc
