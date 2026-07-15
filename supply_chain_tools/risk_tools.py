import random
from langchain.tools import tool
from supply_chain_tools import ToolError


# ---------------------------------------------------------------------------
# Geopolitical risk profiles – realistic EV supply chain context
# ---------------------------------------------------------------------------
_GEO_RISK_PROFILES = {
    "China": {
        "risk_score": 72,
        "factors": [
            "Export controls on critical minerals",
            "US-China trade tensions",
            "Dominance in lithium refining (60%+ global share)",
            "Regulatory opacity",
        ],
        "sanctions_active": False,
        "trade_restrictions": True,
    },
    "Congo": {
        "risk_score": 88,
        "factors": [
            "Artisanal mining & child labor concerns",
            "Political instability in Katanga province",
            "Conflict mineral designation",
            "Limited infrastructure & logistics risk",
        ],
        "sanctions_active": False,
        "trade_restrictions": False,
    },
    "Russia": {
        "risk_score": 91,
        "factors": [
            "Active international sanctions",
            "Supply disruption due to conflict",
            "Nickel & palladium export restrictions",
            "Banking and payment channel risks",
        ],
        "sanctions_active": True,
        "trade_restrictions": True,
    },
    "Australia": {
        "risk_score": 15,
        "factors": [
            "Stable democratic governance",
            "Strong mining regulation",
            "Long lead times for permitting",
        ],
        "sanctions_active": False,
        "trade_restrictions": False,
    },
    "Chile": {
        "risk_score": 28,
        "factors": [
            "Nationalization debates on lithium",
            "Water scarcity in Atacama region",
            "Stable trade partnerships",
        ],
        "sanctions_active": False,
        "trade_restrictions": False,
    },
    "Belgium": {
        "risk_score": 10,
        "factors": [
            "EU regulatory compliance",
            "Strong IP protection",
            "High labor cost",
        ],
        "sanctions_active": False,
        "trade_restrictions": False,
    },
    "South Korea": {
        "risk_score": 18,
        "factors": [
            "Geopolitical tension with North Korea",
            "Strong semiconductor & battery ecosystem",
            "Favorable trade agreements",
        ],
        "sanctions_active": False,
        "trade_restrictions": False,
    },
}


def _classify_risk(score: float) -> str:
    """Classify a numeric risk score into Low / Medium / High."""
    if score <= 30:
        return "Low"
    elif score <= 65:
        return "Medium"
    return "High"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def calculate_supplier_risk_score(supplier_data: dict) -> dict:
    """Calculate a composite risk score (0-100) for a supplier.

    Expects a dict with keys:
      - country (str): Supplier's country of operation
      - tier (int): Supply-chain tier (1 = direct OEM supplier)
      - materials_supplied (list[str]): e.g. ['lithium', 'cobalt']

    The score blends three components:
      1. Geographic risk – derived from geopolitical profiles
      2. Dependency risk – higher for critical single-source materials
      3. Quality risk – random variance simulating inspection outcomes

    Use this tool to evaluate whether a supplier poses acceptable risk
    for procurement decisions.
    """
    try:
        if not isinstance(supplier_data, dict):
            raise ToolError("supplier_data must be a dictionary")

        country = supplier_data.get("country", "Unknown")
        tier = supplier_data.get("tier", 1)
        materials = supplier_data.get("materials_supplied", [])

        # --- Geographic risk component (0-40) ---
        geo_profile = _GEO_RISK_PROFILES.get(country, {})
        geo_base = geo_profile.get("risk_score", 50)
        geo_component = (geo_base / 100) * 40

        # --- Dependency risk component (0-35) ---
        critical_materials = {"cobalt", "lithium", "nickel"}
        critical_count = len(set(materials) & critical_materials)
        dependency_base = min(critical_count * 30, 100)
        tier_multiplier = 1.0 if tier == 1 else 0.75
        dependency_component = (dependency_base / 100) * 35 * tier_multiplier

        # --- Quality risk component (0-25) – simulated ---
        quality_component = random.uniform(2, 25)

        composite = round(geo_component + dependency_component + quality_component, 1)
        composite = min(composite, 100.0)

        return {
            "supplier_country": country,
            "risk_score": composite,
            "risk_level": _classify_risk(composite),
            "components": {
                "geographic_risk": round(geo_component, 1),
                "dependency_risk": round(dependency_component, 1),
                "quality_risk": round(quality_component, 1),
            },
            "sanctions_flag": geo_profile.get("sanctions_active", False),
            "recommendation": (
                "Immediate review required"
                if composite > 70
                else "Monitor closely" if composite > 40 else "Acceptable"
            ),
        }
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Risk score calculation failed: {exc}") from exc


@tool
def detect_geopolitical_risk(country: str) -> dict:
    """Assess the geopolitical risk profile for a given country.

    Returns a risk score (0-100), contributing risk factors, and flags
    for active sanctions or trade restrictions. Covers key EV supply chain
    geographies: China, Congo, Russia, Australia, Chile, Belgium, South Korea.

    Use this when evaluating sourcing decisions or diversification strategy
    across different supplier geographies.
    """
    try:
        if not country or not isinstance(country, str):
            raise ToolError("country must be a non-empty string")

        profile = _GEO_RISK_PROFILES.get(country)
        if profile is None:
            # Generate a plausible default for unlisted countries
            score = random.randint(20, 60)
            return {
                "country": country,
                "risk_score": score,
                "risk_level": _classify_risk(score),
                "factors": ["No detailed profile available – estimated risk"],
                "sanctions_active": False,
                "trade_restrictions": False,
                "data_confidence": "Low",
            }

        return {
            "country": country,
            "risk_score": profile["risk_score"],
            "risk_level": _classify_risk(profile["risk_score"]),
            "factors": profile["factors"],
            "sanctions_active": profile["sanctions_active"],
            "trade_restrictions": profile["trade_restrictions"],
            "data_confidence": "High",
        }
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Geopolitical risk detection failed: {exc}") from exc


@tool
def detect_supplier_concentration(suppliers_list: list) -> dict:
    """Detect concentration risk across a set of suppliers.

    Accepts a list of supplier dicts, each containing at minimum:
      - supplier_id (str)
      - country (str)
      - materials_supplied (list[str])

    Returns a Herfindahl-style concentration index (0-1), per-country and
    per-material breakdowns, and flags for dangerous single-source dependencies.

    Use this to evaluate whether the supply base is dangerously concentrated
    in a single geography or material source.
    """
    try:
        if not isinstance(suppliers_list, list) or len(suppliers_list) == 0:
            raise ToolError("suppliers_list must be a non-empty list of supplier dicts")

        country_counts: dict[str, int] = {}
        material_counts: dict[str, int] = {}
        total = len(suppliers_list)

        for sup in suppliers_list:
            c = sup.get("country", "Unknown")
            country_counts[c] = country_counts.get(c, 0) + 1
            for mat in sup.get("materials_supplied", []):
                material_counts[mat] = material_counts.get(mat, 0) + 1

        # Herfindahl-Hirschman style index on country shares
        hhi = sum((cnt / total) ** 2 for cnt in country_counts.values())
        hhi = round(hhi, 4)

        # Single-source detection
        single_source_materials = [
            mat for mat, cnt in material_counts.items() if cnt == 1
        ]

        concentration_level = (
            "Critical" if hhi > 0.5 else "High" if hhi > 0.3 else "Moderate" if hhi > 0.15 else "Low"
        )

        return {
            "total_suppliers": total,
            "concentration_index": hhi,
            "concentration_level": concentration_level,
            "country_distribution": country_counts,
            "material_distribution": material_counts,
            "single_source_materials": single_source_materials,
            "diversification_recommendation": (
                "Urgent: diversify supplier base"
                if hhi > 0.5
                else "Consider adding alternative sources"
                if hhi > 0.25
                else "Supply base is reasonably diversified"
            ),
        }
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Supplier concentration analysis failed: {exc}") from exc


@tool
def detect_quality_deviation(inspection_data: dict) -> dict:
    """Detect quality deviations from incoming inspection data.

    Expects a dict with optional keys:
      - batch_id (str): The batch being inspected
      - supplier_id (str): Supplier under inspection
      - inspection_type (str): e.g. 'incoming', 'in-process', 'final'
      - sample_size (int): Number of units inspected

    Simulates a defect rate and compares against industry thresholds to
    determine whether a deviation has occurred. Returns severity level
    and recommended corrective action.

    Use this when checking incoming material quality or evaluating supplier
    quality performance.
    """
    try:
        if not isinstance(inspection_data, dict):
            raise ToolError("inspection_data must be a dictionary")

        batch_id = inspection_data.get("batch_id", f"BATCH-{random.randint(1000,9999)}")
        supplier_id = inspection_data.get("supplier_id", "UNKNOWN")
        inspection_type = inspection_data.get("inspection_type", "incoming")
        sample_size = inspection_data.get("sample_size", random.randint(50, 500))

        # Simulate defect rate (realistic: 0.1% – 8%)
        defect_rate = round(random.uniform(0.1, 8.0), 2)

        # Threshold logic by inspection type
        thresholds = {
            "incoming": 2.0,
            "in-process": 1.5,
            "final": 0.5,
        }
        threshold = thresholds.get(inspection_type, 2.0)

        deviation_flag = defect_rate > threshold

        if defect_rate > threshold * 3:
            severity = "High"
            action = "Quarantine batch – escalate to supplier quality engineering"
        elif defect_rate > threshold:
            severity = "Medium"
            action = "Issue Supplier Corrective Action Request (SCAR)"
        else:
            severity = "Low"
            action = "No action required – within acceptable limits"

        return {
            "batch_id": batch_id,
            "supplier_id": supplier_id,
            "inspection_type": inspection_type,
            "sample_size": sample_size,
            "defect_rate_percent": defect_rate,
            "threshold_percent": threshold,
            "deviation_flag": deviation_flag,
            "severity_level": severity,
            "recommended_action": action,
        }
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Quality deviation detection failed: {exc}") from exc
