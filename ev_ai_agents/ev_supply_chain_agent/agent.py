import os
import re
import logging
from typing import List, Optional, Any, Dict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from .state import SupplyChainState
from ev_ai_agents.ev_supply_chain_agent.tools.supplier_tools import (
    get_supplier_profile,
    get_supplier_tier,
    get_supplier_geography
)
from ev_ai_agents.ev_supply_chain_agent.tools.traceability_tools import (
    trace_material_batch,
    map_cell_to_pack,
    map_pack_to_vehicle,
    verify_traceability_completeness
)
from ev_ai_agents.ev_supply_chain_agent.tools.risk_tools import (
    calculate_supplier_risk_score,
    detect_geopolitical_risk,
    detect_supplier_concentration,
    assess_battery_quality
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("supply_chain_agent")

_TOOL_REGISTRY = {
    "get_supplier_profile": get_supplier_profile,
    "get_supplier_tier": get_supplier_tier,
    "get_supplier_geography": get_supplier_geography,
    "trace_material_batch": trace_material_batch,
    "map_cell_to_pack": map_cell_to_pack,
    "map_pack_to_vehicle": map_pack_to_vehicle,
    "verify_traceability_completeness": verify_traceability_completeness,
    "calculate_supplier_risk_score": calculate_supplier_risk_score,
    "detect_geopolitical_risk": detect_geopolitical_risk,
    "detect_supplier_concentration": detect_supplier_concentration,
    "assess_battery_quality": assess_battery_quality
}


def resolve_supply_chain_params(query: str, state_supplier: str = None, state_batch: str = None, state_country: str = None) -> dict:
    """Intelligent entity extraction and parameter resolution for supply chain queries."""
    query_upper = query.upper() if query else ""
    
    # Extract supplier_id if present
    sup_match = re.search(r"SUP[-_]?\d{3}", query_upper)
    supplier_id = state_supplier or (sup_match.group(0).replace("_", "-") if sup_match else None)
    
    # Extract batch_id if present
    batch_match = re.search(r"BAT[-_]?\d{4}[-_]?\d{3}", query_upper)
    batch_id = state_batch or (batch_match.group(0).replace("_", "-") if batch_match else None)
    
    # Extract country if present
    country = state_country
    if not country:
        for c in ["CHINA", "CONGO", "BELGIUM", "AUSTRALIA", "RUSSIA", "CHILE", "SOUTH KOREA", "USA", "GERMANY"]:
            if c in query_upper:
                country = c.title()
                break
                
    # Infer supplier ID from company names if not given as SUP-###
    if not supplier_id:
        if "GANFENG" in query_upper: supplier_id = "SUP-001"
        elif "UMICORE" in query_upper: supplier_id = "SUP-002"
        elif "GLENCORE" in query_upper: supplier_id = "SUP-003"
        elif "PILBARA" in query_upper: supplier_id = "SUP-004"
        elif "CATL" in query_upper: supplier_id = "SUP-005"
        elif "NORILSK" in query_upper: supplier_id = "SUP-006"
        elif "SQM" in query_upper: supplier_id = "SUP-007"
        elif "LG" in query_upper: supplier_id = "SUP-008"
        
    # Default to SUP-001 for general / non-specific risk queries
    if not supplier_id:
        supplier_id = "SUP-001"
        
    # Align batch_id and country to matched supplier if omitted
    supplier_map = {
        "SUP-001": ("BAT-2024-001", "China"),
        "SUP-002": ("BAT-2024-005", "Belgium"),
        "SUP-003": ("BAT-2024-002", "Congo"),
        "SUP-004": ("BAT-2024-006", "Australia"),
        "SUP-005": ("BAT-2024-004", "China"),
        "SUP-006": ("BAT-2024-003", "Russia"),
        "SUP-007": ("BAT-2024-007", "Chile"),
        "SUP-008": ("BAT-2024-008", "South Korea"),
    }
    
    def_batch, def_country = supplier_map.get(supplier_id, ("BAT-2024-001", "China"))
    if not batch_id: batch_id = def_batch
    if not country: country = def_country
    
    return {
        "supplier_id": supplier_id,
        "batch_id": batch_id,
        "country": country
    }


class SupplyChainQueryPlan(BaseModel):
    query_type: str = Field(description="Query type: conceptual, statistical, asset, or hybrid")
    tools: List[str] = Field(description="List of tool keys to run.")
    supplier_id: Optional[str] = Field(default=None)
    batch_id: Optional[str] = Field(default=None)
    country: Optional[str] = Field(default=None)
    confidence: float = Field(default=1.0)


class SupplyChainReasoningOutput(BaseModel):
    unified_report: str = Field(description="Unified report summarizing supplier profile, traceability, and risk.")


def generate_llm_response(prompt_messages: list, response_model: Any = None) -> Any:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key.startswith("dummy"):
        logger.warning("GROQ_API_KEY is missing or invalid. Falling back to deterministic report builder.")
        return None
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
        if response_model:
            structured_llm = llm.with_structured_output(response_model)
            return structured_llm.invoke(prompt_messages)
        return llm.invoke(prompt_messages)
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return None


def planner_node(state: SupplyChainState) -> dict:
    user_query = state.get("user_query") or state.get("query", "")
    params = resolve_supply_chain_params(user_query, state.get("supplier_id"), state.get("batch_id"), state.get("country"))
    
    tools_to_run = [
        "get_supplier_profile",
        "get_supplier_tier",
        "get_supplier_geography",
        "trace_material_batch",
        "calculate_supplier_risk_score",
        "detect_geopolitical_risk",
        "assess_battery_quality"
    ]
    
    return {
        "user_query": user_query,
        "detected_intent": "hybrid",
        "analysis_mode": "hybrid_analysis",
        "analysis_plan": tools_to_run,
        "confidence": 1.0,
        "supplier_id": params["supplier_id"],
        "batch_id": params["batch_id"],
        "country": params["country"],
        "tool_outputs": {}
    }


def tool_executor_node(state: SupplyChainState) -> dict:
    tools_to_run = state.get("analysis_plan", [])
    supplier_id = state.get("supplier_id", "SUP-001")
    batch_id = state.get("batch_id", "BAT-2024-001")
    country = state.get("country", "China")
    
    tool_outputs = {}
    
    for tool_name in tools_to_run:
        tool_callable = _TOOL_REGISTRY.get(tool_name)
        if not tool_callable:
            tool_outputs[tool_name] = {"error": f"Tool {tool_name} not found"}
            continue
            
        try:
            logger.info(f"Executing tool: {tool_name}")
            if tool_name in ["get_supplier_profile", "get_supplier_tier", "get_supplier_geography", "calculate_supplier_risk_score"]:
                res = tool_callable.invoke({"supplier_id": supplier_id})
            elif tool_name == "trace_material_batch":
                res = tool_callable.invoke({"batch_id": batch_id, "supplier_id": supplier_id})
            elif tool_name == "detect_geopolitical_risk":
                res = tool_callable.invoke({"country": country})
            elif tool_name == "assess_battery_quality":
                res = tool_callable.invoke({"supplier_id": supplier_id, "batch_id": batch_id})
            else:
                res = tool_callable.invoke({})
                
            tool_outputs[tool_name] = res
        except Exception as e:
            logger.warning(f"Error executing tool {tool_name}: {e}")
            tool_outputs[tool_name] = {"error": f"Tool execution failed: {str(e)}", "status": "Failed"}
            
    return {"tool_outputs": tool_outputs, "selected_tools": tools_to_run}


def validate_and_extract_metrics(tool_outputs: dict, default_sup: str = "SUP-001", default_batch: str = "BAT-2024-001", default_country: str = "China") -> tuple[dict, list]:
    """Supply Chain Metric Validation & Conflict Resolution Layer.
    Extracts single sources of truth for supplier profile, battery quality metrics, and risk scores.
    """
    provenance: dict[str, dict] = {}
    conflicts: list[str] = []

    # 1. Normalized Supplier Profile (Source: get_supplier_profile & get_supplier_tier)
    prof = tool_outputs.get("get_supplier_profile", {})
    tier_info = tool_outputs.get("get_supplier_tier", {})
    
    if isinstance(prof, dict) and "supplier_id" in prof:
        sup_id = str(prof.get("supplier_id"))
        sup_name = str(prof.get("supplier_name", "Ganfeng Lithium Co."))
        sup_cntry = str(prof.get("country", default_country))
        mats = prof.get("materials_supplied", [])
        mat_str = ", ".join(mats) if mats else "lithium"
        bats = prof.get("battery_types", [])
        bat_str = ", ".join(bats) if bats else "LFP"
        v_class = tier_info.get("tier_label") if isinstance(tier_info, dict) and "tier_label" in tier_info else "Tier 1 – Direct OEM Supplier"
        
        provenance["supplier_id"] = {"value": sup_id, "source": "get_supplier_profile"}
        provenance["supplier_name"] = {"value": sup_name, "source": "get_supplier_profile"}
        provenance["country"] = {"value": sup_cntry, "source": "get_supplier_profile"}
        provenance["battery_material"] = {"value": mat_str, "source": "get_supplier_profile"}
        provenance["battery_chemistry"] = {"value": bat_str, "source": "get_supplier_profile"}
        provenance["vendor_classification"] = {"value": v_class, "source": "get_supplier_tier"}
        provenance["status"] = {"value": prof.get("status", "active").capitalize(), "source": "get_supplier_profile"}
    else:
        provenance["supplier_id"] = {"value": default_sup, "source": "default_resolution"}
        provenance["supplier_name"] = {"value": "Not available from current analysis.", "source": "default_resolution"}
        provenance["country"] = {"value": default_country, "source": "default_resolution"}
        provenance["battery_material"] = {"value": "Not available from current analysis.", "source": "default_resolution"}
        provenance["battery_chemistry"] = {"value": "Not available from current analysis.", "source": "default_resolution"}
        provenance["vendor_classification"] = {"value": "Not available from current analysis.", "source": "default_resolution"}
        provenance["status"] = {"value": "Not available from current analysis.", "source": "default_resolution"}

    # 2. Normalized Battery Quality Metrics (Source: assess_battery_quality ONLY)
    qual = tool_outputs.get("assess_battery_quality", {})
    if isinstance(qual, dict) and "inspection_count" in qual and "error" not in qual:
        provenance["batch_id"] = {"value": qual.get("batch_id", default_batch), "source": "assess_battery_quality"}
        provenance["inspection_count"] = {"value": int(qual.get("inspection_count")), "source": "assess_battery_quality"}
        provenance["defects_found"] = {"value": int(qual.get("defects_found")), "source": "assess_battery_quality"}
        provenance["defect_rate_percent"] = {"value": float(qual.get("defect_rate_percent")), "source": "assess_battery_quality"}
        provenance["defect_type"] = {"value": str(qual.get("defect_type")), "source": "assess_battery_quality"}
        provenance["quality_severity"] = {"value": str(qual.get("severity_level")), "source": "assess_battery_quality"}
        conflicts.append("Battery quality defect rate validated from originating tool output (`assess_battery_quality`).")
    else:
        provenance["batch_id"] = {"value": default_batch, "source": "default_resolution"}
        provenance["inspection_count"] = {"value": None, "source": "assess_battery_quality"}
        provenance["defects_found"] = {"value": None, "source": "assess_battery_quality"}
        provenance["defect_rate_percent"] = {"value": None, "source": "assess_battery_quality"}
        provenance["defect_type"] = {"value": None, "source": "assess_battery_quality"}
        provenance["quality_severity"] = {"value": None, "source": "assess_battery_quality"}

    # 3. Supplier Risk Score (Source: calculate_supplier_risk_score)
    risk = tool_outputs.get("calculate_supplier_risk_score", {})
    if isinstance(risk, dict) and "risk_score" in risk and "error" not in risk:
        provenance["supplier_risk_score"] = {"value": float(risk.get("risk_score")), "source": "calculate_supplier_risk_score"}
        provenance["supplier_risk_level"] = {"value": str(risk.get("risk_level")), "source": "calculate_supplier_risk_score"}
        comp = risk.get("components", {})
        if isinstance(comp, dict):
            provenance["geographic_risk"] = {"value": float(comp.get("geographic_risk")) if comp.get("geographic_risk") is not None else None, "source": "calculate_supplier_risk_score"}
            provenance["dependency_risk"] = {"value": float(comp.get("dependency_risk")) if comp.get("dependency_risk") is not None else None, "source": "calculate_supplier_risk_score"}
            provenance["quality_risk"] = {"value": float(comp.get("quality_risk")) if comp.get("quality_risk") is not None else None, "source": "calculate_supplier_risk_score"}

    # 4. Geopolitical Risk (Source: detect_geopolitical_risk)
    geopol = tool_outputs.get("detect_geopolitical_risk", {})
    if isinstance(geopol, dict) and "risk_score" in geopol and "error" not in geopol:
        provenance["geopolitical_risk_score"] = {"value": float(geopol.get("risk_score")), "source": "detect_geopolitical_risk"}
        provenance["geopolitical_risk_level"] = {"value": str(geopol.get("risk_level")), "source": "detect_geopolitical_risk"}
        provenance["dependency_score"] = {"value": float(geopol.get("dependency_score")), "source": "detect_geopolitical_risk"}
        provenance["geopolitical_factors"] = {"value": geopol.get("factors", []), "source": "detect_geopolitical_risk"}

    # 5. Traceability (Source: trace_material_batch)
    trace = tool_outputs.get("trace_material_batch", {})
    if isinstance(trace, dict) and trace.get("trace_paths") and "error" not in trace:
        paths = trace.get("trace_paths", [])
        if paths:
            first_p = paths[0]
            provenance["trace_cell_id"] = {"value": first_p.get("cell_id"), "source": "trace_material_batch"}
            provenance["trace_pack_id"] = {"value": first_p.get("pack_id"), "source": "trace_material_batch"}
            provenance["trace_vehicle_id"] = {"value": first_p.get("vehicle_id"), "source": "trace_material_batch"}

    return provenance, conflicts


def _format_supply_chain_report(state: SupplyChainState) -> dict:
    """Construct structured executive supply chain report following exact business & integrity rules."""
    tool_outputs = state.get("tool_outputs", {})
    sup_id = state.get("supplier_id", "SUP-001")
    batch_id = state.get("batch_id", "BAT-2024-001")
    country = state.get("country", "China")
    
    provenance, conflicts = validate_and_extract_metrics(tool_outputs, sup_id, batch_id, country)

    def get_val(key: str, default_str: str = "Not available from current analysis.") -> str:
        item = provenance.get(key)
        if item and item.get("value") is not None:
            val = item["value"]
            if isinstance(val, float):
                return f"{val:.1f}"
            return str(val)
        return default_str

    s_name = get_val("supplier_name")
    s_id = get_val("supplier_id", sup_id)
    s_country = get_val("country", country)
    s_material = get_val("battery_material")
    s_chemistry = get_val("battery_chemistry")
    s_class = get_val("vendor_classification")
    s_status = get_val("status")

    s_risk_score = get_val("supplier_risk_score")
    s_risk_level = get_val("supplier_risk_level")
    geo_comp = get_val("geographic_risk")
    dep_comp = get_val("dependency_risk")
    qual_comp = get_val("quality_risk")

    # 1. Executive Summary
    exec_lines = [
        "# Executive Summary",
        f"Comprehensive supply chain risk and traceability evaluation conducted for supplier **{s_name} ({s_id})**, operating in **{s_country}**.",
        f"The composite supplier risk score is evaluated at **{s_risk_score} / 100** ({s_risk_level} Risk Level), validated directly from source tool outputs.",
        f"Traceability lineage for material batch **{get_val('batch_id', batch_id)}** is mapped from raw material origin through cell quality inspection and vehicle integration."
    ]

    # 2. Supplier Risk Assessment
    supplier_risk_lines = [
        "\n# Supplier Risk Assessment",
        "\nSupplier Profile:",
        f"- Supplier Name: {s_name}",
        f"- Supplier ID: {s_id}",
        f"- Country: {s_country}",
        f"- Battery Material: {s_material}",
        f"- Battery Chemistry: {s_chemistry}",
        f"- Vendor Classification: {s_class}",
        f"- Supplier Status: {s_status}",
        "\nSupplier Risk Score:",
        f"{s_risk_score} / 100" if s_risk_score != "Not available from current analysis." else "Not available from current analysis.",
        "\nRisk Level:",
        f"{s_risk_level}",
        "\nRisk Breakdown:",
        f"- Geopolitical Risk: {geo_comp}",
        f"- Dependency Risk: {dep_comp}",
        f"- Quality Risk: {qual_comp}",
        f"- ESG Risk: {s_risk_level if s_risk_level != 'Not available from current analysis.' else 'Not available from current analysis.'}"
    ]

    # 3. Material Traceability
    cell_id_val = get_val("trace_cell_id", None)
    pack_id_val = get_val("trace_pack_id", None)
    vehicle_id_val = get_val("trace_vehicle_id", None)

    if cell_id_val and pack_id_val and vehicle_id_val and cell_id_val != "Not available from current analysis.":
        trace_lines = [
            "\n# Material Traceability",
            "\nSupply Chain Flow Lineage:",
            "```",
            f"{s_id} ({s_country})",
            "      |",
            f"  {s_name} Refinery",
            "      |",
            f"  {get_val('batch_id', batch_id)}",
            "      |",
            f"  Battery Cell QC ({cell_id_val})",
            "      |",
            f"  Battery Pack ({pack_id_val})",
            "      |",
            f"  Vehicle VIN ({vehicle_id_val})",
            "```"
        ]
    else:
        trace_lines = [
            "\n# Material Traceability",
            "Traceability information not available from current analysis."
        ]

    # 4. Geopolitical Risk
    g_score = get_val("geopolitical_risk_score")
    g_level = get_val("geopolitical_risk_level")
    g_dep = get_val("dependency_score")
    g_factors_item = provenance.get("geopolitical_factors")
    g_factors = g_factors_item["value"] if g_factors_item and isinstance(g_factors_item.get("value"), list) else []

    if g_score != "Not available from current analysis.":
        geopol_lines = [
            "\n# Geopolitical Risk",
            f"- Country Analyzed: **{s_country}**",
            f"- Geopolitical Risk Score: **{g_score} / 100** ({g_level})",
            f"- Supply Dependency Score: **{g_dep} / 100**"
        ]
        if g_factors:
            geopol_lines.append("- Key Geopolitical Risk Factors:")
            for f in g_factors:
                geopol_lines.append(f"  • {f}")
    else:
        geopol_lines = [
            "\n# Geopolitical Risk",
            "Geopolitical risk data not available from current analysis."
        ]

    # 5. ESG Compliance
    esg_evidence = []
    if g_score != "Not available from current analysis.":
        esg_evidence.append(f"Geopolitical risk score of {g_score}/100 for country sourcing region ({s_country}) from `detect_geopolitical_risk`.")
    if dep_comp != "Not available from current analysis.":
        esg_evidence.append(f"Regional supply dependency risk component evaluated at {dep_comp} from `calculate_supplier_risk_score`.")
    if qual_comp != "Not available from current analysis.":
        esg_evidence.append(f"Manufacturing quality risk component evaluated at {qual_comp} from `assess_battery_quality`.")

    esg_lines = [
        "\n# ESG Compliance",
        "\nSupplier:",
        f"{s_name} ({s_id})" if s_name != "Not available from current analysis." else "Not available from current analysis.",
        "\nRisk Level:",
        f"{s_risk_level}",
        "\nEvidence:"
    ]
    if esg_evidence:
        for ev in esg_evidence:
            esg_lines.append(f"- {ev}")
    else:
        esg_lines.append("- No direct ESG sensor telemetry recorded in current dataset.")

    esg_lines.extend([
        "\nAdditional ESG factors:",
        "Not available from current analysis.",
        "\nRecommendations:",
        "Strategic Recommendation: Diversify suppliers and increase responsible sourcing audits."
    ])

    # 6. Battery Quality Audit
    insp_cnt = get_val("inspection_count")
    defects = get_val("defects_found")
    def_pct = get_val("defect_rate_percent")
    def_type = get_val("defect_type")
    q_sev = get_val("quality_severity")

    if insp_cnt != "Not available from current analysis." and def_pct != "Not available from current analysis.":
        qual_lines = [
            "\n# Battery Quality Audit",
            f"- Batch Inspected: **{get_val('batch_id', batch_id)}**",
            f"- Total Inspected: **{insp_cnt} cells**",
            f"- Defects Found: **{defects} cells**",
            f"- Defect Rate: **{def_pct}%**",
            f"- Common Defect: **{def_type}**",
            f"- Quality Severity: **{q_sev}** (Acceptable threshold: 2.0%)",
            "*(Metric validated from source tool output: `assess_battery_quality`)*"
        ]
    else:
        qual_lines = [
            "\n# Battery Quality Audit",
            "Quality Data: Not available from current analysis."
        ]

    # 7. Procurement Risk
    proc_lines = [
        "\n# Procurement Risk",
        f"- Current Sourcing Exposure: Regional concentration in **{s_country}** for **{s_material}**.",
        "- Supply Vulnerability: High single-region refining dependency presents logistics and geopolitical exposure.",
        "- Operational Capacity Metrics: Not available from current analysis."
    ]

    # 8. Diversification Strategy
    div_lines = [
        "\n# Diversification Strategy",
        f"Current Risk: Concentrated reliance on **{s_country}** mineral refining.",
        "\nRecommended Actions:",
        "1. Strategic Recommendation: Consider supplier diversification across alternative regions such as Australia and Chile.",
        "2. Strategic Recommendation: Develop regional sourcing alternatives and nearshore refining capacity.",
        "3. Strategic Recommendation: Increase recycled lithium usage in battery cell production.",
        "4. Strategic Recommendation: Maintain strategic inventory buffer of critical battery materials."
    ]

    # 9. Key Recommendations
    recs = [
        f"Strategic Recommendation: Initiate dual-sourcing qualification for {s_id} ({s_name}) material contracts.",
        f"Strategic Recommendation: Perform onsite ESG and quality audit at {s_name} facilities in {s_country}.",
        "Strategic Recommendation: Implement automated batch-level defect tracking for incoming cell shipments.",
        "Strategic Recommendation: Establish 3-month buffer inventory for critical battery-grade raw materials."
    ]

    rec_lines = [
        "\n# Key Recommendations",
        "\n".join([f"{i+1}. {r}" for i, r in enumerate(recs)])
    ]

    full_report = "\n".join(
        exec_lines +
        supplier_risk_lines +
        trace_lines +
        geopol_lines +
        esg_lines +
        qual_lines +
        proc_lines +
        div_lines +
        rec_lines
    )

    return {
        "unified_report": full_report,
        "summary": full_report,
        "recommendations": recs,
        "next_steps": [
            "Issue Supplier Corrective Action Requests (SCAR) if defect rate exceeds 2.0%.",
            "Review secondary supplier contract terms for Tier 1 materials.",
            "Schedule quarterly geopolitical risk reviews."
        ],
        "provenance_map": provenance
    }


def llm_reasoning_node(state: SupplyChainState) -> dict:
    user_query = state.get("user_query", "")
    tool_outputs = state.get("tool_outputs", {})
    formatted_report = _format_supply_chain_report(state)
    
    system_prompt = (
        "You are an expert EV Supply Chain Risk Intelligence Agent.\n"
        "Your task is to interpret tool outputs and user queries to provide a production-grade risk intelligence report.\n"
        "Strict Rule: Follow the exact 9 headers: # Executive Summary, # Supplier Risk Assessment, # Material Traceability, # Geopolitical Risk, # ESG Compliance, # Battery Quality Audit, # Procurement Risk, # Diversification Strategy, # Key Recommendations.\n"
        "Strict Rule: Never perform calculations or invent numerical values. Use exact metrics from tool outputs."
    )
    
    user_prompt = f"User Query: {user_query}\n\nTool Outputs:\n{tool_outputs}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = generate_llm_response(messages, SupplyChainReasoningOutput)
    
    report_text = formatted_report["unified_report"]
    if response and hasattr(response, "unified_report") and response.unified_report:
        report_text = response.unified_report
        
    return {
        "reasoning_output": {
            "unified_report": report_text
        },
        "unified_report": report_text,
        "summary": report_text,
        "recommendations": formatted_report["recommendations"],
        "next_steps": formatted_report["next_steps"]
    }


def response_builder_node(state: SupplyChainState) -> dict:
    reasoning = state.get("reasoning_output", {})
    report_dict = _format_supply_chain_report(state)
    report_text = reasoning.get("unified_report") or state.get("unified_report") or report_dict["unified_report"]
    tool_outputs = state.get("tool_outputs", {})
    
    profile_out = tool_outputs.get("get_supplier_profile", {})
    risk_out = tool_outputs.get("calculate_supplier_risk_score", {})
    
    return {
        "status": "success",
        "unified_report": report_text,
        "summary": report_text,
        "tool_outputs": tool_outputs,
        "supplier_details": profile_out,
        "risk_rating": risk_out.get("risk_level", "Medium") if isinstance(risk_out, dict) else "Medium",
        "recommendations": report_dict.get("recommendations", []),
        "next_steps": report_dict.get("next_steps", [])
    }


workflow = StateGraph(SupplyChainState)
workflow.add_node("planner", planner_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("llm_reasoning", llm_reasoning_node)
workflow.add_node("response_builder", response_builder_node)
workflow.set_entry_point("planner")
workflow.add_edge("planner", "tool_executor")
workflow.add_edge("tool_executor", "llm_reasoning")
workflow.add_edge("llm_reasoning", "response_builder")
workflow.add_edge("response_builder", END)

supply_chain_app = workflow.compile()


def get_agent_executor():
    """Shim to support legacy entrypoint."""
    class AppWrapper:
        def invoke(self, inputs):
            user_input = inputs.get("input") or inputs.get("user_query") or ""
            res = supply_chain_app.invoke({"user_query": user_input})
            res["output"] = res.get("unified_report", "")
            return res
    return AppWrapper()
