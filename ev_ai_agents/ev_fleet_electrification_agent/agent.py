"""
agent.py — Orchestration layer for the Fleet Electrification Readiness Agent.
"""

from __future__ import annotations

import logging
import sys
import os
import re
from typing import Any, List, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Resolve project root so imports work
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))                   # …/ev_fleet_electrification_agent/
_ROOT_DIR = os.path.dirname(os.path.dirname(_AGENT_DIR))                  # project root
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# Existing tool imports
from ev_ai_agents.ev_fleet_electrification_agent.tools.fleet_data_tool      import fetch_vehicle_data, analyze_fleet_csv, aggregate_fleet_statistics
from ev_ai_agents.ev_fleet_electrification_agent.tools.readiness_score_tool  import calculate_readiness_score
from ev_ai_agents.ev_fleet_electrification_agent.tools.ev_matching_tool      import recommend_ev_replacement
from ev_ai_agents.ev_fleet_electrification_agent.tools.roi_tool              import calculate_roi
from ev_ai_agents.ev_fleet_electrification_agent.tools.route_analysis_tool   import analyze_vehicle_route
from ev_ai_agents.ev_fleet_electrification_agent.tools.procurement_tool      import recommend_procurement

from dotenv import load_dotenv
load_dotenv()

from ev_ai_agents.ev_fleet_electrification_agent.state import AgentState

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Centralized Tool Registry
_TOOL_REGISTRY: dict[str, Any] = {
    "fleet_data_tool":            fetch_vehicle_data,
    "analyze_fleet_csv":          analyze_fleet_csv,
    "aggregate_fleet_statistics": aggregate_fleet_statistics,
    "readiness_score_tool":       calculate_readiness_score,
    "ev_matching_tool":           recommend_ev_replacement,
    "roi_tool":                   calculate_roi,
    "route_analysis_tool":        analyze_vehicle_route,
    "procurement_tool":           recommend_procurement,
}

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas for Structured LLM Outputs & Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class FleetQueryPlan(BaseModel):
    query_type: str = Field(description="Query type: asset, fleet, conceptual, statistical, or hybrid")
    requires_dataset: bool = Field(description="True if dataset access is required")
    tools: List[str] = Field(description="List of tool keys to run. Choose from: aggregate_fleet_statistics, fleet_data_tool, readiness_score_tool, ev_matching_tool, roi_tool, procurement_tool, route_analysis_tool, analyze_fleet_csv")
    requires_llm: bool = Field(description="True if LLM reasoning is required to synthesize the final answer")
    generic_description: str = Field(default="", description="Extracted generic vehicle/batch description if no specific ID is provided")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 representing plan confidence")

class FleetReasoningOutput(BaseModel):
    summary: str = Field(description="Summary of the fleet electrification analysis.")
    reasoning: str = Field(description="Detailed logic explaining the recommendations, ROI, and feasibility.")
    recommendations: List[str] = Field(description="List of actionable electrification recommendations.")
    risks: List[str] = Field(description="Key deployment or transition risks identified.")
    next_steps: List[str] = Field(description="Concrete immediate next steps for fleet managers.")

class LLMResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None

# ─────────────────────────────────────────────────────────────────────────────
# Isolated LLM Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_llm() -> ChatGroq:
    """Initialize the Groq model. Validates that GROQ_API_KEY exists."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key.startswith("dummy"):
        raise ValueError("GROQ_API_KEY not found or invalid in environment.")
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

def generate_llm_response(prompt_messages: list, response_model: Any = None) -> LLMResponse:
    """Invokes the LLM and catches expected configuration and API infrastructure errors.
    Propagates programming exceptions normally.
    """
    try:
        llm = get_llm()
        if response_model:
            structured_llm = llm.with_structured_output(response_model)
            res = structured_llm.invoke(prompt_messages)
            return LLMResponse(success=True, data=res, error=None)
        else:
            res = llm.invoke(prompt_messages)
            return LLMResponse(success=True, data=res, error=None)
    except (ValueError, Exception) as exc:
        exc_class_name = type(exc).__name__
        is_infra = False
        
        # Determine if it's an expected infrastructure error
        if isinstance(exc, ValueError):
            is_infra = True
        else:
            # Safely check module or package info without hard dependency failures
            mod_name = type(exc).__module__.lower()
            if "groq" in mod_name or "openai" in mod_name:
                is_infra = True
            elif "Connection" in exc_class_name or "Timeout" in exc_class_name or "RateLimit" in exc_class_name:
                is_infra = True
                
        if is_infra:
            log.warning(f"LLM infrastructure error caught: {exc_class_name} - {str(exc)}")
            return LLMResponse(success=False, data=None, error="LLM_NOT_CONFIGURED")
        else:
            # Propagate normal developer/programming errors
            raise exc

# ─────────────────────────────────────────────────────────────────────────────
# LangGraph Nodes
# ─────────────────────────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> dict:
    """Planner node: Requests an execution plan from the LLM and stores the LLMResponse."""
    user_query = state.user_query
    vehicle_id = state.vehicle_id
    
    mode_context = f"Target Vehicle ID: '{vehicle_id}' (Asset Mode)" if vehicle_id else "No Vehicle ID provided (Fleet Mode / General Electrification)"
    
    system_prompt = (
        "You are the Query Planner for a Fleet Electrification & Procurement Intelligence system.\n"
        "Analyze the user's natural language query and decide on an execution plan.\n"
        "Available tool capabilities in the registry:\n"
        "- 'aggregate_fleet_statistics': Aggregates fleet-wide metrics, readiness distribution, distance patterns, charging feasibility, savings potential, and carbon reduction impact across the fleet.\n"
        "- 'fleet_data_tool': Retrieve specifications and basic data for a specific vehicle_id.\n"
        "- 'readiness_score_tool': Calculate an electrification readiness score (0-100) for a vehicle.\n"
        "- 'ev_matching_tool': Recommends EV replacement models based on existing vehicle specs.\n"
        "- 'roi_tool': Calculates financial return on investment and estimated payback years for a vehicle.\n"
        "- 'procurement_tool': Recommends purchase/lease strategy and schedule for a vehicle.\n"
        "- 'route_analysis_tool': Performs route and range safety analysis for daily vehicle routes.\n"
        "- 'analyze_fleet_csv': Batch/bulk csv upload processing tool.\n"
        "\n"
        "Rules:\n"
        "1. If Target Vehicle ID is None or omitted, set query_type to 'fleet' or 'conceptual' and include 'aggregate_fleet_statistics' in tools.\n"
        "2. If Target Vehicle ID is provided, set query_type to 'asset' or 'hybrid' and select vehicle-specific tools.\n"
        "Classify query_type into one of: 'asset', 'fleet', 'conceptual', 'statistical', or 'hybrid'.\n"
        "Confidence must be a float between 0.0 and 1.0."
    )
    
    query_snippet = user_query[:300] + "..." if len(user_query) > 300 else user_query
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context: {mode_context}\nGenerate a query plan for user query: '{query_snippet}'")
    ]
    
    response = generate_llm_response(messages, FleetQueryPlan)
    
    updates: dict[str, Any] = {
        "planner_response": response
    }
    
    if response.success and response.data:
        plan: FleetQueryPlan = response.data
        updates["detected_intent"] = plan.query_type
        updates["analysis_plan"] = plan.tools
        updates["confidence"] = plan.confidence
        if vehicle_id:
            updates["analysis_mode"] = "asset"
        elif plan.query_type in ["fleet", "conceptual", "statistical"]:
            updates["analysis_mode"] = plan.query_type
        else:
            updates["analysis_mode"] = "fleet"
    else:
        # Fallback when LLM planner is unconfigured or failed
        if vehicle_id:
            updates["analysis_mode"] = "asset"
            updates["analysis_plan"] = ["fleet_data_tool", "readiness_score_tool", "ev_matching_tool", "roi_tool", "procurement_tool"]
        else:
            updates["analysis_mode"] = "fleet"
            updates["analysis_plan"] = ["aggregate_fleet_statistics"]
        updates["detected_intent"] = updates["analysis_mode"]
        updates["confidence"] = 1.0
        
    return updates

def tool_executor_node(state: AgentState) -> dict:
    """Tool Executor node: Executes the planned tools if planning was successful."""
    planner_res: Optional[LLMResponse] = state.planner_response
    vehicle_id = state.vehicle_id 
    tool_outputs: dict[str, Any] = {}
    
    # Determine tools to run
    if planner_res and planner_res.success and planner_res.data:
        tools_to_run = list(planner_res.data.tools)
    else:
        tools_to_run = []
        
    if not vehicle_id:
        # Fleet Mode: Ensure aggregate_fleet_statistics is included
        if not tools_to_run or "aggregate_fleet_statistics" not in tools_to_run:
            tools_to_run.append("aggregate_fleet_statistics")
    else:
        # Asset Mode: Ensure asset tools are present if tools_to_run was empty
        if not tools_to_run:
            tools_to_run = ["fleet_data_tool", "readiness_score_tool", "ev_matching_tool", "roi_tool", "procurement_tool"]
            
    log.info(f"Tool Executor Mode: {'Asset Mode (' + vehicle_id + ')' if vehicle_id else 'Fleet Mode'}")
    log.info(f"Selected tools to run: {tools_to_run}")

    if vehicle_id:
        # Asset Mode: Fetch vehicle data using tool (NO direct CSV reads in agent.py)
        fetch_tool = _TOOL_REGISTRY.get("fleet_data_tool")
        vehicle_record = {}
        if fetch_tool:
            try:
                res = fetch_tool.invoke({"vehicle_id": vehicle_id})
                if isinstance(res, dict) and "error" not in res:
                    vehicle_record = res
            except Exception as e:
                log.warning(f"Error fetching vehicle data for {vehicle_id}: {e}")
                
        # Resolve parameters based on tool dataset response
        daily_distance = float(vehicle_record.get("daily_distance_km", 0.0))
        charging_window = float(vehicle_record.get("charging_window_hours", 
                                vehicle_record.get("available_charging_window_hours", 8.0)))
        idle_minutes = float(vehicle_record.get("idle_time_minutes",
                             vehicle_record.get("avg_idle_minutes", 45.0)))
        stops = int(vehicle_record.get("stops_per_day", 10))
        route_type = str(vehicle_record.get("usage_pattern", 
                         vehicle_record.get("route_type", "mixed"))).lower()
        consistency = float(vehicle_record.get("route_consistency_score", 0.85))
        vehicle_age = float(vehicle_record.get("vehicle_age_years", 3.0))
        
        vtype = str(vehicle_record.get("vehicle_type", "")).lower()
        if "heavy" in vtype or "truck" in vtype:
            def_eff = 3.5
        elif "van" in vtype or "delivery" in vtype:
            def_eff = 9.5
        elif "bus" in vtype:
            def_eff = 4.8
        else:
            def_eff = 12.0
        fuel_efficiency = float(vehicle_record.get("fuel_efficiency_kmpl", def_eff))
        operating_hours = float(vehicle_record.get("operating_hours_per_day", 24.0 - charging_window))
        utilization = float(vehicle_record.get("utilization_rate", 0.75))
        payload = float(vehicle_record.get("payload_requirement_kg", 
                        vehicle_record.get("payload_capacity_kg", 
                        vehicle_record.get("payload_kg", 1000.0))))
                
        for tool_name in tools_to_run:
            tool_callable = _TOOL_REGISTRY.get(tool_name)
            if not tool_callable:
                log.error(f"Tool '{tool_name}' not found in registry.")
                tool_outputs[tool_name] = {"error": f"Tool '{tool_name}' not available in registry."}
                continue
                
            try:
                if tool_name == "readiness_score_tool":
                    params = {
                        "daily_distance_km": daily_distance,
                        "available_charging_window_hours": charging_window,
                        "avg_idle_minutes": idle_minutes,
                        "stops_per_day": stops,
                        "route_type": route_type,
                        "route_consistency_score": consistency,
                        "vehicle_age_years": vehicle_age,
                        "fuel_efficiency_kmpl": fuel_efficiency,
                        "operating_hours_per_day": operating_hours,
                        "utilization_rate": utilization,
                        "payload_kg": payload
                    }
                elif tool_name == "ev_matching_tool":
                    params = {
                        "daily_distance_km": daily_distance,
                        "available_charging_window_hours": charging_window,
                        "payload_kg": payload
                    }
                elif tool_name == "aggregate_fleet_statistics":
                    params = {}
                else:
                    params = {"vehicle_id": vehicle_id}
                    
                log.info(f"Tool inputs resolved for {tool_name}: {params}")
                result = tool_callable.invoke(params)
                log.info(f"Tool outputs returned for {tool_name}: {result}")
                tool_outputs[tool_name] = result
            except Exception as e:
                log.exception(f"Error executing tool {tool_name}")
                tool_outputs[tool_name] = {"error": f"Tool execution failed: {str(e)}"}
    else:
        # Fleet Mode: Filter tools to fleet-level tools only (aggregate_fleet_statistics, analyze_fleet_csv)
        fleet_tools = [t for t in tools_to_run if t in ["aggregate_fleet_statistics", "analyze_fleet_csv"]]
        if "aggregate_fleet_statistics" not in fleet_tools:
            fleet_tools.append("aggregate_fleet_statistics")
        tools_to_run = fleet_tools
        
        for tool_name in fleet_tools:
            tool_callable = _TOOL_REGISTRY.get(tool_name)
            if not tool_callable:
                log.error(f"Tool '{tool_name}' not found in registry.")
                tool_outputs[tool_name] = {"error": f"Tool '{tool_name}' not available in registry."}
                continue
                
            try:
                if tool_name == "aggregate_fleet_statistics":
                    params = {}
                elif tool_name == "analyze_fleet_csv":
                    params = {"csv_path": ""}
                else:
                    params = {}
                log.info(f"Tool inputs resolved for {tool_name}: {params}")
                result = tool_callable.invoke(params)
                log.info(f"Tool outputs returned for {tool_name}: {result}")
                tool_outputs[tool_name] = result
            except Exception as e:
                log.exception(f"Error executing tool {tool_name}")
                tool_outputs[tool_name] = {"error": f"Tool execution failed: {str(e)}"}
                
    return {"tool_outputs": tool_outputs, "selected_tools": tools_to_run}

# ─────────────────────────────────────────────────────────────────────────────
# Financial Viability & Decision Logic Helper
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_financial_viability(readiness_score: float, roi_pct: float, payback_years: float, daily_distance_km: float) -> dict:
    """Evaluate financial viability, priority, and procurement decision based on readiness and ROI rules."""
    if readiness_score >= 70.0:
        if roi_pct > 50.0 and payback_years < 5.0:
            return {
                "recommendation": "Proceed with EV conversion",
                "procurement_action": "Proceed with procurement",
                "priority": "HIGH",
                "viable": True,
                "reason_tag": "High financial ROI and fast payback."
            }
        elif roi_pct > 0.0 and payback_years <= 10.0:
            return {
                "recommendation": "Proceed with EV conversion",
                "procurement_action": "Proceed with procurement",
                "priority": "MEDIUM",
                "viable": True,
                "reason_tag": "Positive financial ROI within acceptable payback window."
            }
        else:
            return {
                "recommendation": "Technically suitable but financially not viable for immediate conversion",
                "procurement_action": "Delay procurement and monitor",
                "priority": "LOW",
                "viable": False,
                "reason_tag": f"Low annual utilization ({daily_distance_km:.1f} km/day) results in extended payback ({payback_years:.1f} yrs) and negative/weak ROI ({roi_pct:.1f}%)."
            }
    elif readiness_score >= 50.0:
        if roi_pct > 0.0 and payback_years <= 10.0:
            return {
                "recommendation": "Conditionally ready — requires operational route adjustments before conversion",
                "procurement_action": "Delay procurement and monitor",
                "priority": "MEDIUM",
                "viable": True,
                "reason_tag": "Moderate readiness but positive financial return."
            }
        else:
            return {
                "recommendation": "Conditionally ready — requires operational adjustments and financial improvement",
                "procurement_action": "Reject current replacement option",
                "priority": "LOW",
                "viable": False,
                "reason_tag": "Moderate readiness and weak financial return."
            }
    else:
        return {
            "recommendation": "Not currently suitable for electrification",
            "procurement_action": "Reject current replacement option",
            "priority": "LOW",
            "viable": False,
            "reason_tag": "Low technical readiness score."
        }


def _format_electrification_report(state: AgentState) -> dict:
    """Construct standardized 6-section electrification report following business decision rules."""
    tool_outputs = state.tool_outputs or {}
    v_id = state.vehicle_id
    
    fleet_data = tool_outputs.get("fleet_data_tool", {})
    readiness = tool_outputs.get("readiness_score_tool", {})
    ev_match = tool_outputs.get("ev_matching_tool", {})
    roi = tool_outputs.get("roi_tool", {})
    proc = tool_outputs.get("procurement_tool", {})
    agg_stats = tool_outputs.get("aggregate_fleet_statistics", {})

    if v_id and isinstance(fleet_data, dict) and "vehicle_id" in fleet_data:
        r_score = float(readiness.get("readiness_score", fleet_data.get("readiness_score", 70)))
        r_class = readiness.get("classification", "Ready")
        ev_model = ev_match.get("recommended_ev", fleet_data.get("recommended_ev_model", "Tata Ultra EV"))
        compat = float(ev_match.get("compatibility_score", 1.0)) * 100.0
        range_km = float(ev_match.get("estimated_range_km", 300.0))
        bat_cap = float(ev_match.get("battery_capacity_kwh", 150.0))
        price = float(ev_match.get("purchase_price_usd", roi.get("ev_purchase_price_usd", 120000.0)))
        
        annual_sav = float(roi.get("total_annual_savings_usd", fleet_data.get("estimated_annual_savings", 3092.0)))
        payback = float(roi.get("estimated_payback_years", 38.8))
        roi_val = float(roi.get("roi_percent_over_10_years", -74.2))
        daily_km = float(fleet_data.get("daily_distance_km", 46.0))

        fin_eval = _evaluate_financial_viability(r_score, roi_val, payback, daily_km)

        # 1. Electrification Readiness
        readiness_lines = [
            "# Electrification Readiness",
            f"Readiness Score: {int(r_score)}/100 ({r_class})",
            f"Recommendation: {fin_eval['recommendation']}",
            f"Transition Priority: {fin_eval['priority']}"
        ]

        # 2. EV Asset Replacement Match
        match_lines = [
            "\n# EV Asset Replacement Match",
            f"Recommended EV Model: {ev_model}",
            f"Compatibility Score: {compat:.0f}%",
            f"Battery Specs: {bat_cap:.0f} kWh | Estimated Range: {range_km:.0f} km",
            f"Acquisition Cost: ${price:,.0f}"
        ]
        if not fin_eval["viable"]:
            match_lines.append("Note: Alternative EV models with lower acquisition cost should be evaluated.")

        # 3. Financial Savings & ROI Analysis
        reasoning_text = (
            f"Although {v_id} is technically compatible with {ev_model} (Readiness Score: {int(r_score)}/100), "
            f"the low annual utilization ({daily_km:.1f} km/day) results in a {payback:.1f}-year payback period and "
            f"negative 10-year ROI ({roi_val:.1f}%). Immediate replacement is not economically justified."
            if not fin_eval["viable"] else
            f"Vehicle {v_id} demonstrates strong financial viability with annual operating savings of ${annual_sav:,.0f}, "
            f"a payback period of {payback:.1f} years, and a 10-year ROI of {roi_val:.1f}%. Immediate conversion is economically justified."
        )

        fin_lines = [
            "\n# Financial Savings & ROI Analysis",
            f"Annual Operating Savings: ${annual_sav:,.0f}",
            f"Estimated Payback Period: {payback:.1f} years",
            f"10-Year ROI: {roi_val:.1f}%",
            f"Financial Viability Status: {'FINANCIALLY VIABLE' if fin_eval['viable'] else 'FINANCIALLY UNVIABLE'}",
            f"\nBusiness Reasoning:\n\"{reasoning_text}\""
        ]

        # 4. Procurement Timeline
        proc_lines = [
            "\n# Procurement Timeline",
            f"Procurement Action: {fin_eval['procurement_action']}",
            f"Recommended Window: {proc.get('recommended_purchase_window', 'Q4 2025' if not fin_eval['viable'] else 'Q1 2025')}",
            f"Strategic Guidance: {'Delay procurement pending EV price reductions, operational route restructuring, or government subsidies.' if not fin_eval['viable'] else 'Proceed with vehicle order and depot charging installation.'}"
        ]

        # 5. Key AI Guidelines
        guideline_text = (
            f"Vehicle is technically ready, but immediate conversion is not financially recommended due to negative ROI ({roi_val:.1f}%). Evaluate lower-cost EV alternatives or delay procurement."
            if not fin_eval["viable"] else
            f"Vehicle {v_id} is both technically ready (Score: {int(r_score)}/100) and financially attractive (ROI: {roi_val:.1f}%). Proceed with immediate Q1 procurement and depot charger setup."
        )
        ai_lines = [
            "\n# Key AI Guidelines",
            guideline_text
        ]

        # 6. Tactical Next Steps
        next_steps = [
            "Evaluate alternative lower-cost EV models or second-life commercial EVs." if not fin_eval["viable"] else "Finalize purchase order for recommended EV model.",
            "Reassess financial payback if daily utilization or route distance increases.",
            "Monitor regional EV fleet subsidies and tax incentives to improve payback timeline.",
            "Assess depot charger capacity before committing capital."
        ]
        next_lines = [
            "\n# Tactical Next Steps",
            "\n".join([f"- {s}" for s in next_steps])
        ]

        full_summary = "\n".join(readiness_lines + match_lines + fin_lines + proc_lines + ai_lines + next_lines)
        
        recs = [
            f"Recommendation for {v_id}: {fin_eval['recommendation']} (Priority: {fin_eval['priority']})",
            f"Procurement Action: {fin_eval['procurement_action']}",
        ]
        if not fin_eval["viable"]:
            recs.append("Alternative EV models with lower acquisition cost should be evaluated.")

        return {
            "summary": full_summary,
            "recommendations": recs,
            "next_steps": next_steps
        }

    else:
        # Fleet-Wide Mode Summary
        tot_evs = agg_stats.get("total_vehicles_analyzed", 1000)
        ready_pct = agg_stats.get("readiness_distribution", {}).get("Ready_pct", 65.0)
        tot_sav = agg_stats.get("financial_impact", {}).get("total_annual_savings_usd", 2500000.0)
        co2_red = agg_stats.get("environmental_impact", {}).get("total_carbon_reduction_percent", 58.0)

        full_summary = "\n".join([
            "# Electrification Readiness",
            f"Fleet-Wide Readiness: {ready_pct:.1f}% of fleet analyzed ({tot_evs} vehicles) is technically suitable for electrification.",
            "Transition Priority: Phased rollout based on combined technical readiness and financial ROI.",
            "\n# EV Asset Replacement Match",
            "Category Matches: Light Commercial, Heavy Duty, Passenger, and Delivery Van categories evaluated against active EV models.",
            "\n# Financial Savings & ROI Analysis",
            f"Total Fleet Annual Savings: ${tot_sav:,.0f}",
            f"Fleet Carbon Reduction: {co2_red:.1f}%",
            "Business Reasoning:\n\"Fleet conversion should prioritize high-utilization routes (>120 km/day) where fuel savings quickly offset initial EV acquisition costs.\"",
            "\n# Procurement Timeline",
            "Procurement Action: Phased deployment — Phase 1 (High ROI) in Q1-Q2, Phase 2 (Medium ROI) in Q3-Q4, Phase 3 (Unviable/Low ROI) deferred.",
            "\n# Key AI Guidelines",
            "Prioritize vehicle replacement based on joint technical readiness (Score >= 70) and positive 10-year ROI. Defer low-utilization vehicle replacements until cheaper models or incentives become available.",
            "\n# Tactical Next Steps",
            "- Select top 20% high-ROI vehicles for immediate Phase 1 procurement.",
            "- Conduct depot electrical infrastructure audit for fast-charging installations.",
            "- Re-evaluate low-utilization vehicles annually against falling battery pack prices."
        ])

        return {
            "summary": full_summary,
            "recommendations": [
                "Proceed with Phase 1 procurement for vehicles with ROI > 50% and payback < 5 years.",
                "Delay procurement for low-utilization vehicles with negative 10-year ROI.",
                "Evaluate alternative lower-cost EV models for low-mileage urban delivery routes."
            ],
            "next_steps": [
                "Audit high-priority depot charging capacity.",
                "Apply for state and federal EV fleet purchase subsidies.",
                "Initiate vendor RFPs for recommended EV commercial models."
            ]
        }


def llm_reasoning_node(state: AgentState) -> dict:
    """LLM Reasoning node: Asks the LLM to interpret tool outputs or conceptually answer queries."""
    user_query = state.user_query
    analysis_mode = state.analysis_mode
    tool_outputs = state.tool_outputs
    vehicle_id = state.vehicle_id
    
    formatted_report = _format_electrification_report(state)

    system_prompt = (
        "You are an expert consultant in fleet electrification and EV procurement.\n"
        "Your job is to interpret tool outputs and provide realistic business recommendations.\n"
        "CRITICAL BUSINESS REASONING & DECISION RULES:\n"
        "1. Financial Viability Decision Logic: Evaluate BOTH technical readiness AND financial attractiveness.\n"
        "   - IF readiness >= 70 AND ROI > 0 AND payback <= 10 years -> Recommendation: 'Proceed with EV conversion'\n"
        "   - IF readiness >= 70 BUT (ROI <= 0 OR payback > 10 years) -> Recommendation: 'Technically suitable but financially not viable for immediate conversion'\n"
        "   - IF readiness < 50 -> Recommendation: 'Not currently suitable for electrification'\n"
        "2. Procurement Action: Dynamically choose 'Proceed with procurement' (if ROI attractive), 'Delay procurement and monitor' (if technical suitable but weak economics), or 'Reject current replacement option' (if technical/financial unviable).\n"
        "3. Priority Classification: HIGH (ROI > 50% & payback < 5 yrs), MEDIUM (ROI > 0% & payback 5-10 yrs), LOW (ROI <= 0 or payback > 10 yrs or readiness < 70).\n"
        "4. Business Reasoning: Explain WHY the decision was made using utilization (km/day), payback period, and 10-year ROI.\n"
        "5. Alternative EVs: If recommended EV is expensive/unviable for low utilization, explicitly state: 'Alternative EV models with lower acquisition cost should be evaluated.'\n"
        "6. Response Headers: Include the 6 required sections: # Electrification Readiness, # EV Asset Replacement Match, # Financial Savings & ROI Analysis, # Procurement Timeline, # Key AI Guidelines, # Tactical Next Steps.\n"
        "Strict Rule: NEVER perform calculations or invent numbers. Use exact metrics from tool outputs."
    )
    
    user_prompt = (
        f"User Query: {user_query}\n"
        f"Mode: {analysis_mode} ({'Asset: ' + vehicle_id if vehicle_id else 'Fleet-wide Analysis'})\n\n"
        f"Tool Outputs:\n{tool_outputs}"
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = generate_llm_response(messages, FleetReasoningOutput)
    
    updates: dict[str, Any] = {
        "reasoner_response": response,
        "reasoning_output": formatted_report
    }
    
    if response.success and response.data:
        res_data = response.data
        if hasattr(res_data, "recommendations") and res_data.recommendations:
            formatted_report["recommendations"] = res_data.recommendations
        if hasattr(res_data, "next_steps") and res_data.next_steps:
            formatted_report["next_steps"] = res_data.next_steps
        updates["reasoning_output"] = formatted_report
        
    return updates


def response_builder_node(state: AgentState) -> dict:
    """Response Builder node: Converts the state into the central Supervisor response model."""
    reasoning = state.reasoning_output or _format_electrification_report(state)
    
    final_response = {
        "status": "success",
        "selected_tools": state.selected_tools,
        "tool_outputs": state.tool_outputs,
        "summary": reasoning.get("summary", ""),
        "recommendations": reasoning.get("recommendations", []),
        "next_steps": reasoning.get("next_steps", [])
    }
    
    # Inspect tool outputs for any error keys
    failed = False
    for k, v in state.tool_outputs.items():
        if isinstance(v, dict) and "error" in v:
            failed = True
            break
            
    if failed:
        final_response["status"] = "partial"
        
    return {
        "final_response": final_response,
        "execution_status": final_response["status"]
    }

# ─────────────────────────────────────────────────────────────────────────────
# Standardized LangGraph Pipeline Setup
# ─────────────────────────────────────────────────────────────────────────────

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("llm_reasoning", llm_reasoning_node)
workflow.add_node("response_builder", response_builder_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "tool_executor")
workflow.add_edge("tool_executor", "llm_reasoning")
workflow.add_edge("llm_reasoning", "response_builder")
workflow.add_edge("response_builder", END)

fleet_app = workflow.compile()

# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(user_query: str, vehicle_id: Optional[str] = None) -> dict[str, Any]:
    """Execute the Fleet Electrification Readiness Agent using LangGraph.

    Args:
        user_query: Natural-language question from the user.
        vehicle_id: Optional fleet vehicle identifier. When provided, asset-specific analysis is performed.
                    When omitted (None), generic fleet-planning workflow is executed using aggregate statistics.

    Returns:
        Structured response dictionary matching Supervisor expectations.
    """
    log.info("=" * 60)
    log.info("Fleet Electrification Agent (LangGraph Workflow) — START")
    log.info("  Query     : %s", user_query)
    log.info("  Vehicle   : %s", vehicle_id)
    log.info("=" * 60)
    
    # 1. Validate inputs
    if not isinstance(user_query, str) or not user_query.strip():
        log.error("Invalid user_query: must be a non-empty string.")
        return {
            "status":          "error",
            "selected_tools":  [],
            "tool_outputs":    {},
            "summary":         "Invalid query. Please provide a non-empty question.",
            "recommendations": [],
            "next_steps":      [],
        }

    # Normalize vehicle_id
    clean_vid: Optional[str] = None
    if isinstance(vehicle_id, str) and vehicle_id.strip():
        clean_vid = vehicle_id.strip()

    # 2. Invoke StateGraph workflow
    state = AgentState(
        user_query=user_query.strip(),
        vehicle_id=clean_vid,
        execution_status="pending"
    )
    
    try:
        res = fleet_app.invoke(state)
        if isinstance(res, dict):
            final_res = res.get("final_response", {})
        else:
            final_res = getattr(res, "final_response", {})
            
        log.info("Agent completed with status: %s", final_res.get("status"))
        log.info("Final response: %s", final_res)
        return final_res
    except Exception as e:
        log.exception("Workflow failed during execution")
        return {
            "status":          "error",
            "selected_tools":  [],
            "tool_outputs":    {},
            "summary":         f"Internal Agent execution error: {str(e)}",
            "recommendations": ["Retry execution or inspect backend logs."],
            "next_steps":      []
        }

# ─────────────────────────────────────────────────────────────────────────────
# Local test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("Fleet Electrification Agent — Test Run (Fleet Mode)")
    print("=" * 60)
    res_fleet = run_agent("Evaluate my delivery fleet for electrification and estimate annual savings.")
    print(f"Status         : {res_fleet['status']}")
    print(f"Selected Tools : {res_fleet['selected_tools']}")
    print(f"Summary        : {res_fleet['summary']}")
    print("Recommendations:")
    for i, rec in enumerate(res_fleet.get("recommendations", []), 1):
        print(f"  {i}. {rec}")

    print()
    print("=" * 60)
    print("Fleet Electrification Agent — Test Run (Asset Mode)")
    print("=" * 60)
    res_asset = run_agent("Evaluate VH_15592 for electrification", vehicle_id="VH_15592")
    print(f"Status         : {res_asset['status']}")
    print(f"Selected Tools : {res_asset['selected_tools']}")
    print(f"Summary        : {res_asset['summary']}")
    print("Recommendations:")
    for i, rec in enumerate(res_asset.get("recommendations", []), 1):
        print(f"  {i}. {rec}")

