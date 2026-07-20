import os
import pandas as pd
from langchain.tools import tool
from . import ToolError


# ---------------------------------------------------------------------------
# Load actual datasets
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASETS_DIR = os.path.join(_BASE_DIR, "..", "..", "datasets")

_supply_chain_df = pd.read_csv(os.path.join(_DATASETS_DIR, "ev_supply_chain.csv"))
_minerals_risk_df = pd.read_csv(os.path.join(_DATASETS_DIR, "critical_minerals_risk.csv"))
_battery_quality_df = pd.read_csv(os.path.join(_DATASETS_DIR, "battery_quality.csv"))


def _classify_risk(score: float) -> str:
    """Classify a numeric risk score into Low / Medium / High."""
    if score <= 30:
        return "Low"
    elif score <= 65:
        return "Medium"
    return "High"


def _get_geo_risk(country: str) -> dict:
    """Look up geopolitical risk from critical_minerals_risk.csv for a country."""
    rows = _minerals_risk_df[_minerals_risk_df["country"] == country]
    if rows.empty:
        return {
            "risk_score": 50,
            "risk_level": "Medium",
            "dependency_score": 50.0,
            "political_risk_score": 50.0,
            "factors": ["No detailed profile available – estimated risk"],
            "data_confidence": "Low",
        }
    # Use the max political_risk_score across all materials for the country
    row = rows.loc[rows["political_risk_score"].idxmax()]
    factors = []
    if float(row["production_share"]) > 0.5:
        factors.append(f"Dominant production share ({row['production_share']:.0%})")
    if float(row["dependency_score"]) > 70:
        factors.append(f"High dependency score ({row['dependency_score']})")
    if float(row["export_dependency"]) > 0.7:
        factors.append(f"High export dependency ({row['export_dependency']:.0%})")
    high_risk_rows = rows[rows["risk_level"] == "High"]
    if not high_risk_rows.empty:
        factors.append(f"High-risk materials: {', '.join(high_risk_rows['material'].tolist())}")

    return {
        "risk_score": float(row["political_risk_score"]),
        "risk_level": row["risk_level"],
        "dependency_score": float(row["dependency_score"]),
        "political_risk_score": float(row["political_risk_score"]),
        "factors": factors if factors else ["Stable sourcing environment"],
        "data_confidence": "High",
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def calculate_supplier_risk_score(supplier_id: str) -> dict:
    """Calculate a composite risk score (0-100) for a supplier.

    Expects a supplier_id (str) like 'SUP-001'.

    The score blends three components:
      1. Geographic risk – derived from critical_minerals_risk.csv
      2. Dependency risk – from dependency_score in the dataset
      3. Quality risk – from actual battery_quality.csv defect rates

    Use this tool to evaluate whether a supplier poses acceptable risk
    for procurement decisions.
    """
    try:
        if not isinstance(supplier_id, str):
            raise ToolError("supplier_id must be a string")

        supplier_id = supplier_id.upper()
        # Find supplier data
        sc_rows = _supply_chain_df[_supply_chain_df["supplier_id"] == supplier_id]
        if sc_rows.empty:
            raise ToolError(f"Supplier {supplier_id} not found in dataset.")
            
        row = sc_rows.iloc[0]
        country = row["country"]
        tier = int(row["supplier_tier"])
        materials = sc_rows["material"].unique().tolist()

        # --- Geographic risk component (0-40) from critical_minerals_risk.csv ---
        geo_info = _get_geo_risk(country)
        geo_base = geo_info["risk_score"]
        geo_component = (geo_base / 100) * 40

        # --- Dependency risk component (0-35) from critical_minerals_risk.csv ---
        dep_scores = []
        for mat in materials:
            mat_rows = _minerals_risk_df[
                (_minerals_risk_df["material"] == mat) &
                (_minerals_risk_df["country"] == country)
            ]
            if not mat_rows.empty:
                dep_scores.append(float(mat_rows.iloc[0]["dependency_score"]))
        dependency_base = max(dep_scores) if dep_scores else 30.0
        tier_multiplier = 1.0 if tier == 1 else 0.75
        dependency_component = (dependency_base / 100) * 35 * tier_multiplier

        # --- Quality risk component (0-25) from battery_quality.csv ---
        quality_component = 10.0  # default
        if supplier_id:
            q_rows = _battery_quality_df[_battery_quality_df["supplier_id"] == supplier_id]
            if not q_rows.empty:
                avg_defect_rate = q_rows["defect_rate"].mean()
                # Scale defect rate (0-0.05 range) to 0-25 score
                quality_component = min(round((avg_defect_rate / 0.05) * 25, 1), 25.0)

        composite = round(geo_component + dependency_component + quality_component, 1)
        composite = min(composite, 100.0)

        return {
            "supplier_id": supplier_id,
            "supplier_country": country,
            "risk_score": composite,
            "risk_level": _classify_risk(composite),
            "components": {
                "geographic_risk": round(geo_component, 1),
                "dependency_risk": round(dependency_component, 1),
                "quality_risk": round(quality_component, 1),
            },
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

    Returns a risk score (0-100), contributing risk factors, and dependency
    metrics. Data is sourced from the critical_minerals_risk.csv dataset.

    Use this when evaluating sourcing decisions or diversification strategy
    across different supplier geographies.
    """
    try:
        if not country or not isinstance(country, str):
            raise ToolError("country must be a non-empty string")

        geo_info = _get_geo_risk(country)

        # Get all materials sourced from this country
        country_rows = _minerals_risk_df[_minerals_risk_df["country"] == country]
        materials_from_country = country_rows["material"].tolist() if not country_rows.empty else []
        risk_levels = country_rows["risk_level"].tolist() if not country_rows.empty else []

        return {
            "country": country,
            "risk_score": geo_info["risk_score"],
            "risk_level": geo_info["risk_level"],
            "dependency_score": geo_info["dependency_score"],
            "factors": geo_info["factors"],
            "materials_sourced": materials_from_country,
            "material_risk_levels": dict(zip(materials_from_country, risk_levels)),
            "data_confidence": geo_info["data_confidence"],
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
def assess_battery_quality(supplier_id: str = "", batch_id: str = "") -> dict:
    """Detect quality deviations using actual battery_quality.csv data.

    Expects either:
      - batch_id (str): The batch being inspected
      - supplier_id (str): Supplier under inspection

    Looks up real defect data from battery_quality.csv and compares against
    industry thresholds to determine whether a deviation has occurred.

    Use this when checking incoming material quality or evaluating supplier
    quality performance.
    """
    try:

        # Look up actual data from battery_quality.csv
        if batch_id:
            rows = _battery_quality_df[_battery_quality_df["batch_id"] == batch_id]
        elif supplier_id:
            rows = _battery_quality_df[_battery_quality_df["supplier_id"] == supplier_id]
        else:
            raise ToolError("Must provide either batch_id or supplier_id")

        if rows.empty:
            raise ToolError(
                f"No quality data found. "
                f"Valid batch IDs: {', '.join(_battery_quality_df['batch_id'].tolist())}"
            )

        row = rows.iloc[0]
        defect_rate = float(row["defect_rate"]) * 100  # Convert to percentage
        threshold = 2.0  # Industry standard threshold percentage

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
            "batch_id": row["batch_id"],
            "supplier_id": row["supplier_id"],
            "inspection_count": int(row["inspection_count"]),
            "defects_found": int(row["defects_found"]),
            "defect_rate_percent": round(defect_rate, 3),
            "defect_type": row["defect_type"],
            "threshold_percent": threshold,
            "deviation_flag": deviation_flag,
            "severity_level": severity,
            "recommended_action": action,
        }
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Quality deviation detection failed: {exc}") from exc
