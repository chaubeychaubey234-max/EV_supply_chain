import os
import sys
import re
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

# Ensure the workspace root is in sys.path to import supply_chain_tools
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import state and datasets
from .state import SupplyChainState
from .datasets import (
    query_supplier_demographics,
    query_material_traceability,
    query_geopolitical_mineral_risk,
    query_batch_quality,
    load_supply_chain
)

# Import existing tools from tools folder
from .tools.supplier_tools import (
    get_supplier_profile,
    get_supplier_tier,
    get_supplier_geography
)
from .tools.risk_tools import (
    calculate_supplier_risk_score,
    detect_geopolitical_risk,
    detect_supplier_concentration,
    detect_quality_deviation
)
from .tools.traceability_tools import (
    trace_material_batch,
    map_cell_to_pack,
    map_pack_to_vehicle,
    verify_traceability_completeness
)

# Define structured output schema for parameter extraction
class QueryParameters(BaseModel):
    supplier_id: Optional[str] = Field(None, description="The supplier ID (e.g. SUP-001, SUP-002, etc.)")
    batch_id: Optional[str] = Field(None, description="The batch ID (e.g. BAT-2024-001, BAT-2024-002, etc.)")
    material: Optional[str] = Field(None, description="Critical mineral or battery chemistry name (e.g. lithium, cobalt, nickel, NMC, LFP)")
    country: Optional[str] = Field(None, description="Country name (e.g. China, Congo, Russia, Australia, Chile, Belgium, South Korea)")

def get_llm():
    """Helper to instantiate Google Gemini LLM if API key is present."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.2, 
            google_api_key=api_key
        )
    return None

def fallback_extract(query: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Resilient regex extraction fallback if LLM is unavailable."""
    query_upper = query.upper()
    supplier_id = None
    batch_id = None
    material = None
    country = None
    
    # Extract SUP-XXX
    sup_match = re.search(r'SUP-\d{3}', query_upper)
    if sup_match:
        supplier_id = sup_match.group(0)
        
    # Extract BAT-XXXX-XXX
    bat_match = re.search(r'BAT-\d{4}-\d{3}', query_upper)
    if bat_match:
        batch_id = bat_match.group(0)
        
    # Extract Material
    for m in ['lithium', 'cobalt', 'nickel', 'lfp', 'nmc', 'graphite']:
        if m in query.lower():
            material = m.upper() if m in ['lfp', 'nmc'] else m.capitalize()
            break
            
    # Extract Country
    for c in ['china', 'congo', 'russia', 'australia', 'chile', 'belgium', 'south korea']:
        if c in query.lower():
            if c == 'south korea':
                country = 'South Korea'
            else:
                country = c.capitalize()
            break
            
    return supplier_id, batch_id, material, country


# ---------------------------------------------------------------------------
# LangGraph Nodes
# ---------------------------------------------------------------------------

def extract_parameters_node(state: SupplyChainState):
    """Router & Parameter Extraction Node. Extracts entities and enriches parameters from database."""
    query = state["query"]
    messages = list(state.get("messages", []))
    messages.append({"role": "system", "content": "Initializing parameter extraction node..."})
    
    supplier_id = None
    batch_id = None
    material = None
    country = None
    
    llm = get_llm()
    if llm:
        try:
            structured_llm = llm.with_structured_output(QueryParameters)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert entity extraction system. Extract EV battery supply chain entities (supplier_id, batch_id, material, country) from the user query."),
                ("user", "{query}")
            ])
            chain = prompt | structured_llm
            extracted = chain.invoke({"query": query})
            
            supplier_id = extracted.supplier_id
            batch_id = extracted.batch_id
            material = extracted.material
            country = extracted.country
        except Exception as e:
            messages.append({"role": "warning", "content": f"LLM parameter extraction failed: {e}. Falling back to regex."})
            supplier_id, batch_id, material, country = fallback_extract(query)
    else:
        messages.append({"role": "info", "content": "Gemini API key not found. Using regex parameter extraction."})
        supplier_id, batch_id, material, country = fallback_extract(query)
        
    # --- Database Enrichment ---
    # Case 1: Supplier ID is available -> find country, materials, and a representative batch_id
    if supplier_id:
        supplier_id = supplier_id.upper()
        demographics = query_supplier_demographics(supplier_id)
        if demographics:
            if not country:
                country = demographics["country"]
            if not material and demographics["materials"]:
                material = demographics["materials"][0]
            
            # Find a representative batch for this supplier in our dataset
            df = load_supply_chain()
            matching_batches = df[df["supplier_id"].str.upper() == supplier_id]["batch_id"].unique()
            if len(matching_batches) > 0 and not batch_id:
                batch_id = matching_batches[0]
                
    # Case 2: Batch ID is available -> resolve supplier, country, and material
    if batch_id:
        batch_id = batch_id.upper()
        quality = query_batch_quality(batch_id)
        if quality:
            if not supplier_id:
                supplier_id = quality["supplier_id"].upper()
                demographics = query_supplier_demographics(supplier_id)
                if demographics:
                    if not country:
                        country = demographics["country"]
                    if not material and demographics["materials"]:
                        material = demographics["materials"][0]
        else:
            # Fallback lookup from supply chain df
            df = load_supply_chain()
            matching_rows = df[df["batch_id"].str.upper() == batch_id]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                if not supplier_id:
                    supplier_id = row["supplier_id"]
                if not country:
                    country = row["country"]
                if not material:
                    material = row["material"]
                    
    messages.append({
        "role": "info",
        "content": f"Extracted & Enriched Entities: supplier_id={supplier_id}, batch_id={batch_id}, material={material}, country={country}"
    })
    
    return {
        "supplier_id": supplier_id,
        "batch_id": batch_id,
        "material": material,
        "country": country,
        "messages": messages
    }


def analyze_supplier_node(state: SupplyChainState):
    """Supplier Specialist Agent Node. Profiles demographics, capacity, and tiers."""
    supplier_id = state.get("supplier_id")
    messages = list(state.get("messages", []))
    
    if not supplier_id:
        return {
            "supplier_analysis": {"status": "skipped", "reason": "No supplier ID specified in query context."},
            "messages": messages + [{"role": "system", "content": "Supplier Agent: Skipped (No Supplier ID)"}]
        }
        
    messages.append({"role": "system", "content": f"Supplier Agent: Analysing supplier profile for {supplier_id}..."})
    
    # 1. Invoke Tools
    try:
        profile = get_supplier_profile.invoke({"supplier_id": supplier_id})
        tier = get_supplier_tier.invoke({"supplier_id": supplier_id})
        geography = get_supplier_geography.invoke({"supplier_id": supplier_id})
    except Exception as e:
        # Fallback to direct DB query if tool execution errors out
        demo = query_supplier_demographics(supplier_id)
        profile = {"supplier_id": supplier_id, "supplier_name": demo.get("supplier_name", "Unknown"), "country": demo.get("country", "Unknown"), "capacity_gwh": 10.0}
        tier = {"tier_label": f"Tier {demo.get('supplier_tier', 1)}"}
        geography = {"region": "Unknown"}
        messages.append({"role": "warning", "content": f"Supplier Agent: Tool invocation error, used local DB lookup: {e}"})
        
    analysis = {
        "profile": profile,
        "tier": tier,
        "geography": geography,
        "status": "success"
    }
    
    # 2. Leverage Gemini for detailed reasoning if available
    llm = get_llm()
    if llm:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a senior supply-chain intelligence expert. Summarize the supplier profile, capacity, tier classification, and sourcing geography into a highly structured, analytical brief."),
                ("user", "Profile: {profile}\nTier: {tier}\nGeography: {geography}")
            ])
            chain = prompt | llm
            brief = chain.invoke({"profile": profile, "tier": tier, "geography": geography})
            analysis["brief"] = brief.content
        except Exception as e:
            analysis["brief"] = f"Failed to generate LLM summary: {e}"
    else:
        analysis["brief"] = f"Supplier {profile.get('supplier_name')} is a {tier.get('tier_label')} actor operating in {profile.get('country')} ({geography.get('region', 'Unknown')} sourcing region) with a manufacturing capacity of {profile.get('capacity_gwh', 'N/A')} GWh."
        
    return {
        "supplier_analysis": analysis,
        "messages": messages
    }


def analyze_traceability_node(state: SupplyChainState):
    """Traceability Specialist Agent Node. Traces material batch lineage and passport readiness."""
    batch_id = state.get("batch_id")
    supplier_id = state.get("supplier_id")
    messages = list(state.get("messages", []))
    
    if not batch_id:
        return {
            "traceability_analysis": {"status": "skipped", "reason": "No batch ID available to trace lineage."},
            "messages": messages + [{"role": "system", "content": "Traceability Agent: Skipped (No Batch ID)"}]
        }
        
    messages.append({"role": "system", "content": f"Traceability Agent: Tracing material batch {batch_id}..."})
    
    try:
        trace_data = trace_material_batch.invoke({"batch_id": batch_id})
        completeness = verify_traceability_completeness.invoke({"batch_data": trace_data})
    except Exception as e:
        # Fallback to database queries
        db_trace = query_material_traceability(batch_id)
        trace_data = {
            "batch_id": batch_id,
            "material": db_trace[0]["material"] if db_trace else "Unknown",
            "supplier_id": supplier_id or "Unknown",
            "trace_paths": db_trace,
            "total_cells": len(db_trace)
        }
        completeness = {
            "completeness_percent": 100.0 if db_trace else 0.0,
            "traceability_status": "Complete" if db_trace else "Incomplete",
            "eu_battery_passport_ready": True if db_trace else False,
            "missing_links": []
        }
        messages.append({"role": "warning", "content": f"Traceability Agent: Tool invocation error, used local DB lookup: {e}"})
        
    analysis = {
        "trace_data": trace_data,
        "completeness": completeness,
        "status": "success"
    }
    
    llm = get_llm()
    if llm:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert in EV battery traceability, manufacturing compliance, and EU Battery Passport standards. Summarize the batch trace path (Supplier -> Cell -> Pack -> Vehicle) and evaluate its compliance readiness."),
                ("user", "Trace Lineage: {trace_data}\nCompliance Verification: {completeness}")
            ])
            chain = prompt | llm
            brief = chain.invoke({"trace_data": trace_data, "completeness": completeness})
            analysis["brief"] = brief.content
        except Exception as e:
            analysis["brief"] = f"Failed to generate LLM summary: {e}"
    else:
        analysis["brief"] = f"Batch {batch_id} ({trace_data.get('material')}) has a traceability completeness score of {completeness.get('completeness_percent', 0.0)}%. Status is: {completeness.get('traceability_status')}. EU Battery Passport Ready: {completeness.get('eu_battery_passport_ready')}."
        
    return {
        "traceability_analysis": analysis,
        "messages": messages
    }


def analyze_risk_node(state: SupplyChainState):
    """Risk Specialist Agent Node. Analyzes geopolitical risk, material concentration, and HHI index."""
    supplier_id = state.get("supplier_id")
    country = state.get("country")
    material = state.get("material")
    messages = list(state.get("messages", []))
    
    messages.append({"role": "system", "content": "Risk Agent: Running geopolitical & supplier concentration risk audit..."})
    
    # 1. Supplier risk calculation
    supplier_risk = {}
    if supplier_id:
        try:
            demo = query_supplier_demographics(supplier_id)
            supplier_risk = calculate_supplier_risk_score.invoke({"supplier_data": {
                "country": demo.get("country", country or "Unknown"),
                "tier": demo.get("supplier_tier", 1),
                "materials_supplied": demo.get("materials", [material] if material else [])
            }})
        except Exception as e:
            supplier_risk = {"risk_score": 50.0, "risk_level": "Medium", "recommendation": "Monitor closely"}
            messages.append({"role": "warning", "content": f"Risk Agent: Supplier risk score tool error: {e}"})
            
    # 2. Country geopolitical risk
    geo_risk = {}
    if country:
        try:
            geo_risk = detect_geopolitical_risk.invoke({"country": country})
        except Exception as e:
            geo_risk = {"risk_score": 45.0, "risk_level": "Medium", "factors": ["No details available"]}
            messages.append({"role": "warning", "content": f"Risk Agent: Geopolitical risk tool error: {e}"})
            
    # 3. Supplier concentration analysis
    concentration_risk = {}
    try:
        # Load all suppliers in database to calculate HHI
        df = load_supply_chain()
        suppliers_list = []
        unique_sups = df["supplier_id"].unique()
        for sid in unique_sups:
            demo = query_supplier_demographics(sid)
            if demo:
                suppliers_list.append({
                    "supplier_id": sid,
                    "country": demo["country"],
                    "materials_supplied": demo["materials"]
                })
        concentration_risk = detect_supplier_concentration.invoke({"suppliers_list": suppliers_list})
    except Exception as e:
        concentration_risk = {"concentration_index": 0.35, "concentration_level": "High", "diversification_recommendation": "Consider adding alternative sources"}
        messages.append({"role": "warning", "content": f"Risk Agent: Concentration risk tool error: {e}"})
        
    analysis = {
        "supplier_risk": supplier_risk,
        "geopolitical_risk": geo_risk,
        "concentration_risk": concentration_risk,
        "status": "success"
    }
    
    llm = get_llm()
    if llm:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a specialist in supply chain risk management and geopolitical exposure. Synthesize the supplier's risk score, country risk parameters, and the general supply base concentration HHI metrics into a detailed risk advisory."),
                ("user", "Supplier Risk Score: {supplier_risk}\nCountry Risk Profile: {geo_risk}\nConcentration Index (HHI): {concentration}")
            ])
            chain = prompt | llm
            brief = chain.invoke({"supplier_risk": supplier_risk, "geo_risk": geo_risk, "concentration": concentration_risk})
            analysis["brief"] = brief.content
        except Exception as e:
            analysis["brief"] = f"Failed to generate LLM summary: {e}"
    else:
        analysis["brief"] = f"Country {country} Geopolitical Risk Score: {geo_risk.get('risk_score', 'N/A')} ({geo_risk.get('risk_level', 'N/A')}). Sourcing base concentration: {concentration_risk.get('concentration_level')} (Index: {concentration_risk.get('concentration_index')})."
        
    return {
        "risk_analysis": analysis,
        "messages": messages
    }


def analyze_quality_node(state: SupplyChainState):
    """Quality Specialist Agent Node. Analyzes manufacturing defect rates and triggers corrective actions."""
    batch_id = state.get("batch_id")
    supplier_id = state.get("supplier_id")
    messages = list(state.get("messages", []))
    
    if not batch_id:
        return {
            "quality_analysis": {"status": "skipped", "reason": "No batch ID available to check quality metrics."},
            "messages": messages + [{"role": "system", "content": "Quality Agent: Skipped (No Batch ID)"}]
        }
        
    messages.append({"role": "system", "content": f"Quality Agent: Auditing quality metrics for batch {batch_id}..."})
    
    try:
        # 1. Query database records
        db_quality = query_batch_quality(batch_id)
        
        # 2. Call deviation tool
        sample_size = db_quality.get("inspection_count", 500)
        deviation = detect_quality_deviation.invoke({
            "inspection_data": {
                "batch_id": batch_id,
                "supplier_id": supplier_id or db_quality.get("supplier_id", "UNKNOWN"),
                "sample_size": sample_size,
                "inspection_type": "incoming"
            }
        })
    except Exception as e:
        db_quality = {"batch_id": batch_id, "defect_rate": 0.02, "defect_type": "Unknown"}
        deviation = {"deviation_flag": True, "severity_level": "Medium", "recommended_action": "Issue SCAR"}
        messages.append({"role": "warning", "content": f"Quality Agent: Tool deviation checks failed: {e}"})
        
    analysis = {
        "db_quality": db_quality,
        "deviation": deviation,
        "status": "success"
    }
    
    llm = get_llm()
    if llm:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an EV battery manufacturing quality manager. Review the inspection log and deviation outputs for the batch, identify the quality threat, and outline the necessary QA/QE countermeasures."),
                ("user", "Inspection Record: {quality}\nQuality Deviation Flag & Actions: {deviation}")
            ])
            chain = prompt | llm
            brief = chain.invoke({"quality": db_quality, "deviation": deviation})
            analysis["brief"] = brief.content
        except Exception as e:
            analysis["brief"] = f"Failed to generate LLM summary: {e}"
    else:
        analysis["brief"] = f"Batch {batch_id} defect rate: {db_quality.get('defect_rate', 0.0)*100}%. Defect category: '{db_quality.get('defect_type')}'. QA Deviation Flagged: {deviation.get('deviation_flag')} (Severity: {deviation.get('severity_level')}). Action: {deviation.get('recommended_action')}."
        
    return {
        "quality_analysis": analysis,
        "messages": messages
    }


def synthesize_report_node(state: SupplyChainState):
    """Orchestrator Synthesis Node. Compiles and formats a unified EV supply chain intelligence report."""
    query = state["query"]
    supplier_analysis = state.get("supplier_analysis")
    traceability_analysis = state.get("traceability_analysis")
    risk_analysis = state.get("risk_analysis")
    quality_analysis = state.get("quality_analysis")
    messages = list(state.get("messages", []))
    
    messages.append({"role": "system", "content": "Orchestrator: Compiling unified intelligence report..."})
    
    llm = get_llm()
    if llm:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a Principal EV Supply Chain Analyst at a leading electric vehicle manufacturer.
Your task is to synthesize a professional, unified intelligence report from the findings of your specialized agents.
Use rich markdown styling. Include clear headings, structured bullet points, key metrics blocks, and tables. 

Ensure the report includes:
1. Executive Summary & Sourcing Verdict
2. Supplier Profiling (Tier, capacity, geo region)
3. End-to-End Material Traceability Lineage (Supplier -> Cell -> Pack -> Vehicle)
4. Geopolitical, Sourcing Concentration & Mineral Risk Scorecard (Include HHI and Country details)
5. Incoming Quality & Defect Deviation Audit (Flagging any quarantines or corrective actions)
6. Actionable Procurement Recommendations (Strategic diversification or supplier corrections)

Ensure you base the report on the actual agent telemetry details provided. Do not invent facts."""),
                ("user", """User Sourcing Query: {query}

--- AGENTS TELEMETRY ---
Supplier Agent Summary:
{supplier_data}

Traceability Agent Summary:
{traceability_data}

Risk Agent Summary:
{risk_data}

Quality Agent Summary:
{quality_data}""")
            ])
            chain = prompt | llm
            report = chain.invoke({
                "query": query,
                "supplier_data": supplier_analysis,
                "traceability_data": traceability_analysis,
                "risk_data": risk_analysis,
                "quality_data": quality_analysis
            })
            unified_report = report.content
        except Exception as e:
            messages.append({"role": "error", "content": f"LLM synthesis failed: {e}"})
            unified_report = f"LLM synthesis failed. Fallback report generated.\n\n{generate_fallback_report(state)}"
    else:
        messages.append({"role": "info", "content": "LLM not available. Generating fallback template report."})
        unified_report = generate_fallback_report(state)
        
    return {
        "unified_report": unified_report,
        "messages": messages
    }


def generate_fallback_report(state: SupplyChainState) -> str:
    """Fallback generator that constructs a structured Markdown report from tool outcomes."""
    sup = state.get("supplier_analysis", {})
    trace = state.get("traceability_analysis", {})
    risk = state.get("risk_analysis", {})
    qual = state.get("quality_analysis", {})
    
    sup_profile = sup.get("profile", {})
    sup_tier = sup.get("tier", {})
    sup_geo = sup.get("geography", {})
    
    trace_data = trace.get("trace_data", {})
    comp = trace.get("completeness", {})
    
    s_risk = risk.get("supplier_risk", {})
    g_risk = risk.get("geopolitical_risk", {})
    c_risk = risk.get("concentration_risk", {})
    
    db_q = qual.get("db_quality", {})
    dev_q = qual.get("deviation", {})
    
    report = f"""# EV Supply Chain Sourcing Intelligence & Risk Audit Report

**Query Evaluated**: "{state['query']}"

---

## 1. Executive Summary & Verdict
This audit compiles multi-tier diagnostics for **{sup_profile.get('supplier_name', 'N/A')}** and associated material batches. 
*   **Supplier Risk Classification**: `{s_risk.get('risk_level', 'UNKNOWN')}` (Composite Score: {s_risk.get('risk_score', 'N/A')})
*   **Traceability Completeness**: `{comp.get('traceability_status', 'N/A')}` ({comp.get('completeness_percent', 0)}%)
*   **Quality Inspection Action**: `{dev_q.get('recommended_action', 'N/A')}`

---

## 2. Supplier Profile & Geography
*   **Supplier Name**: {sup_profile.get('supplier_name', 'N/A')} ({sup_profile.get('supplier_id', 'N/A')})
*   **Sourcing Country**: {sup_profile.get('country', 'N/A')}
*   **Supply Chain Tier**: {sup_tier.get('tier_label', 'N/A')}
*   **Materials Supplied**: {", ".join(sup_profile.get('materials_supplied', []))}
*   **Manufacturing Capacity**: {sup_profile.get('capacity_gwh', 'N/A')} GWh

---

## 3. Sourcing Risk & Geopolitical Scorecard
| Sourcing Element | Risk Level | Metric Value | Risk Factors |
| :--- | :--- | :--- | :--- |
| **Supplier Risk Score** | {s_risk.get('risk_level', 'N/A')} | {s_risk.get('risk_score', 'N/A')} / 100 | Geo Component: {s_risk.get('components', {}).get('geographic_risk', 0)}, Dependency Component: {s_risk.get('components', {}).get('dependency_risk', 0)} |
| **Country Geopolitical Risk** | {g_risk.get('risk_level', 'N/A')} | {g_risk.get('risk_score', 'N/A')} / 100 | {", ".join(g_risk.get('factors', [])) if g_risk.get('factors') else 'N/A'} |
| **Sourcing Concentration (HHI)** | {c_risk.get('concentration_level', 'N/A')} | {c_risk.get('concentration_index', 'N/A')} | {c_risk.get('diversification_recommendation', 'N/A')} |

*   **Sanctions Flag**: `{"ACTIVE" if g_risk.get("sanctions_active") or s_risk.get("sanctions_flag") else "NONE"}`
*   **Trade Restrictions Flag**: `{"ACTIVE" if g_risk.get("trade_restrictions") else "NONE"}`

---

## 4. End-to-End Material Traceability Lineage
*   **Material Batch**: {trace_data.get('batch_id', 'N/A')}
*   **Traceability completeness**: {comp.get('completeness_percent', 0)}%
*   **EU Battery Passport Status**: `{"COMPLIANT" if comp.get("eu_battery_passport_ready") else "NON-COMPLIANT"}`
*   **Material Sourcing Tree**:
    *   **Extraction Location**: {trace_data.get('origin_country', 'N/A')}
    *   **Total Cells Tracked**: {trace_data.get('total_cells', 0)} cells
    *   **Sourced Material**: {trace_data.get('material', 'N/A')} ({trace_data.get('purity_grade', 'N/A')})

### Mapped Downstream Trace Mappings:
"""
    if trace_data.get("trace_paths"):
        for path in trace_data["trace_paths"]:
            report += f"\n*   **Cell ID**: `{path.get('cell_id')}` → **Pack ID**: `{path.get('pack_id')}` → **Vehicle VIN**: `{path.get('vehicle_id')}` ({path.get('vehicle_model')})"
    else:
        report += "\n*   No active traceability links found in the current lineage."
        
    report += f"""

---

## 5. Manufacturing Quality & Defects Deviation Audit
*   **Batch Under Audit**: {db_q.get('batch_id', 'N/A')}
*   **Sample Inspected**: {db_q.get('inspection_count', 0)} units
*   **Defect Count**: {db_q.get('defects_found', 0)} units
*   **Inspection Defect Rate**: {db_q.get('defect_rate', 0.0)*100:.2f}%
*   **Inspection Defect Class**: {db_q.get('defect_type', 'N/A')}
*   **Quality Deviation Flagged**: `{"YES" if dev_q.get("deviation_flag") else "NO"}` (Severity: {dev_q.get('severity_level', 'N/A')})
*   **Corrective Recommendation**: {dev_q.get('recommended_action', 'N/A')}

---

## 6. Actionable Procurement Advisory
1.  **Supplier Actions**: Sourcing from `{sup_profile.get('supplier_name', 'N/A')}` indicates a `{s_risk.get('risk_level', 'UNKNOWN')}` procurement exposure. Quality deviations are `{dev_q.get('severity_level', 'N/A')}`.
2.  **Mitigation Plan**:
    *   *If defect rate is high*: Execute the recommended QA action: "{dev_q.get('recommended_action')}".
    *   *If HHI concentration is high*: Sourcing HHI concentration is `{c_risk.get('concentration_level')}`. {c_risk.get('diversification_recommendation')}.
"""
    return report


# ---------------------------------------------------------------------------
# State Graph Construction
# ---------------------------------------------------------------------------

workflow = StateGraph(SupplyChainState)

# Add specialist nodes
workflow.add_node("extract_parameters", extract_parameters_node)
workflow.add_node("analyze_supplier", analyze_supplier_node)
workflow.add_node("analyze_traceability", analyze_traceability_node)
workflow.add_node("analyze_risk", analyze_risk_node)
workflow.add_node("analyze_quality", analyze_quality_node)
workflow.add_node("synthesize_report", synthesize_report_node)

# Set execution flow edges
workflow.set_entry_point("extract_parameters")
workflow.add_edge("extract_parameters", "analyze_supplier")
workflow.add_edge("analyze_supplier", "analyze_traceability")
workflow.add_edge("analyze_traceability", "analyze_risk")
workflow.add_edge("analyze_risk", "analyze_quality")
workflow.add_edge("analyze_quality", "synthesize_report")
workflow.add_edge("synthesize_report", END)

# Compile graph
supply_chain_app = workflow.compile()
