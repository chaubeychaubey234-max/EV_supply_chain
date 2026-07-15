import random
from langchain.tools import tool
from supply_chain_tools import ToolError


# ---------------------------------------------------------------------------
# Mock supplier database – realistic EV battery supply chain actors
# ---------------------------------------------------------------------------
_SUPPLIER_DB = {
    "SUP-001": {
        "supplier_name": "Ganfeng Lithium Co.",
        "country": "China",
        "tier": 1,
        "materials_supplied": ["lithium", "LFP"],
        "certifications": ["ISO 9001", "IATF 16949"],
        "capacity_gwh": 25.0,
    },
    "SUP-002": {
        "supplier_name": "Umicore NV",
        "country": "Belgium",
        "tier": 1,
        "materials_supplied": ["cobalt", "nickel", "NMC"],
        "certifications": ["ISO 14001", "IATF 16949"],
        "capacity_gwh": 18.5,
    },
    "SUP-003": {
        "supplier_name": "Glencore Congo SARL",
        "country": "Congo",
        "tier": 2,
        "materials_supplied": ["cobalt"],
        "certifications": ["RMI"],
        "capacity_gwh": 8.0,
    },
    "SUP-004": {
        "supplier_name": "Pilbara Minerals Ltd",
        "country": "Australia",
        "tier": 2,
        "materials_supplied": ["lithium"],
        "certifications": ["ISO 9001"],
        "capacity_gwh": 12.0,
    },
    "SUP-005": {
        "supplier_name": "CATL Battery Co.",
        "country": "China",
        "tier": 1,
        "materials_supplied": ["LFP", "NMC"],
        "certifications": ["ISO 9001", "IATF 16949", "ISO 14001"],
        "capacity_gwh": 100.0,
    },
    "SUP-006": {
        "supplier_name": "Norilsk Nickel",
        "country": "Russia",
        "tier": 2,
        "materials_supplied": ["nickel", "cobalt"],
        "certifications": ["ISO 9001"],
        "capacity_gwh": 15.0,
    },
    "SUP-007": {
        "supplier_name": "SQM Chile",
        "country": "Chile",
        "tier": 2,
        "materials_supplied": ["lithium"],
        "certifications": ["ISO 9001", "ISO 14001"],
        "capacity_gwh": 10.0,
    },
    "SUP-008": {
        "supplier_name": "LG Energy Solution",
        "country": "South Korea",
        "tier": 1,
        "materials_supplied": ["NMC", "nickel"],
        "certifications": ["ISO 9001", "IATF 16949", "ISO 14001"],
        "capacity_gwh": 60.0,
    },
}


@tool
def get_supplier_profile(supplier_id: str) -> dict:
    """Retrieve the full profile for an EV supply-chain supplier.

    Given a supplier ID (e.g. 'SUP-001'), returns structured data including
    the supplier's name, country of operation, supply-chain tier, materials
    supplied, certifications, and manufacturing capacity in GWh.

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
            "certifications": profile["certifications"],
            "capacity_gwh": profile["capacity_gwh"],
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
        region_map = {
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

        return {
            "supplier_id": supplier_id.upper(),
            "supplier_name": profile["supplier_name"],
            "country": country,
            "region": region_map.get(country, "Unknown"),
            "materials_supplied": profile["materials_supplied"],
            "tier": profile["tier"],
        }
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Failed to fetch supplier geography: {exc}") from exc
