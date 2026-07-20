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
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context: {mode_context}\nGenerate a query plan for user query: '{user_query}'")
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

def llm_reasoning_node(state: AgentState) -> dict:
    """LLM Reasoning node: Asks the LLM to interpret tool outputs or conceptually answer queries."""
    planner_res: Optional[LLMResponse] = state.planner_response
    if planner_res is None or not planner_res.success:
        log.info("LLM Reasoning: Skipping reasoning because planner was unavailable (success=False).")
        return {
            "reasoner_response": LLMResponse(success=False, error="LLM_NOT_CONFIGURED")
        }
        
    user_query = state.user_query
    detected_intent = state.detected_intent
    analysis_mode = state.analysis_mode
    tool_outputs = state.tool_outputs
    vehicle_id = state.vehicle_id
    
    log.info(f"LLM reasoning input (Query): {user_query}")
    log.info(f"LLM reasoning input (Tool Outputs): {tool_outputs}")
    
    system_prompt = (
        "You are an expert consultant in fleet electrification and EV procurement.\n"
        "Your job is to interpret tool outputs and explain them clearly, justify recommendations, and suggest next steps.\n"
        "Strict Rule: You must NEVER perform calculations or invent numerical values. All metrics, scores, financial savings, carbon reduction, and costs must come directly from tool outputs.\n"
        "Strict Rule: If analyzing a specific vehicle (Asset Mode), focus on that vehicle's readiness score, EV match, ROI, and procurement window.\n"
        "Strict Rule: If analyzing the overall fleet (Fleet Mode), discuss fleet-wide readiness distribution, average daily distances, charging feasibility, total savings, and carbon reduction impact based on the aggregate statistics tool outputs."
    )
    
    user_prompt = (
        f"User Query: {user_query}\n"
        f"Mode: {analysis_mode} ({'Asset: ' + vehicle_id if vehicle_id else 'Fleet-wide Analysis'})\n\n"
        f"Tool Outputs:\n"
    )
    for t_name, t_out in tool_outputs.items():
        user_prompt += f"--- {t_name} ---\n{t_out}\n\n"
        
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = generate_llm_response(messages, FleetReasoningOutput)
    
    updates: dict[str, Any] = {
        "reasoner_response": response
    }
    if response.success and response.data:
        updates["reasoning_output"] = response.data.model_dump()
        
    return updates

def response_builder_node(state: AgentState) -> dict:
    """Response Builder node: Converts the state into the central Supervisor response model."""
    planner_res: Optional[LLMResponse] = state.planner_response
    reasoner_res: Optional[LLMResponse] = state.reasoner_response
    
    # Check if LLM is unconfigured
    llm_configured = True
    error_msg = ""
    if planner_res and not planner_res.success and planner_res.error == "LLM_NOT_CONFIGURED":
        llm_configured = False
        error_msg = "LLM is not configured. Add the GROQ_API_KEY to enable AI reasoning."
    elif reasoner_res and not reasoner_res.success and reasoner_res.error == "LLM_NOT_CONFIGURED":
        llm_configured = False
        error_msg = "LLM is not configured. Add the GROQ_API_KEY to enable AI reasoning."
        
    if not llm_configured:
        final_response = {
            "status": "error",
            "selected_tools": [],
            "tool_outputs": {},
            "summary": error_msg,
            "recommendations": ["Add the GROQ_API_KEY to enable AI reasoning."],
            "next_steps": ["Configure GROQ_API_KEY in the environment."]
        }
        return {
            "final_response": final_response,
            "execution_status": "error"
        }
        
    # Success/Partial path: Merging LLM narrative and raw tool outputs
    reasoning = state.reasoning_output or {}
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

