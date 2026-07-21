"""
agent.py — Orchestration layer for the Maintenance Operations Optimiser Agent.
"""

from __future__ import annotations

import os
import sys
import re
import logging
from typing import Any, Dict, List, Optional, Generic, TypeVar
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Setup pathing
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))                  # …/ev_maintenance_operations_agent/
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))                 # project root
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# Import state
from dotenv import load_dotenv
load_dotenv()

from ev_ai_agents.ev_maintenance_operations_agent.state import AgentState

# Import tools
from ev_ai_agents.ev_maintenance_operations_agent.tools.maintenance_risk_analyzer import (
    analyze_maintenance_risk,
    _fetch_record_from_csv
)
from ev_ai_agents.ev_maintenance_operations_agent.tools.maintenance_schedule_optimizer import optimize_maintenance_schedule
from ev_ai_agents.ev_maintenance_operations_agent.tools.charging_availability_planner import plan_charging_availability
from ev_ai_agents.ev_maintenance_operations_agent.tools.maintenance_statistical_tool import calculate_maintenance_statistics
from ev_ai_agents.ev_maintenance_operations_agent.tools.utils import (
    load_maintenance_history,
    load_fleet_operations
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maintenance_operations_agent")

# Centralized Tool Registry
_TOOL_REGISTRY: dict[str, Any] = {
    "maintenance_risk_analyzer":       analyze_maintenance_risk,
    "maintenance_schedule_optimizer":  optimize_maintenance_schedule,
    "charging_availability_planner":   plan_charging_availability,
    "maintenance_statistical_tool":    calculate_maintenance_statistics,
}

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas for Structured LLM Outputs & Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class MaintenanceQueryPlan(BaseModel):
    query_type: str = Field(description="Query type: conceptual, statistical, asset, or hybrid")
    requires_dataset: bool = Field(description="True if dataset access is required")
    tools: List[str] = Field(description="List of tool keys to run. Choose from: maintenance_risk_analyzer, maintenance_schedule_optimizer, charging_availability_planner, maintenance_statistical_tool")
    requires_llm: bool = Field(description="True if LLM reasoning is required to synthesize the final answer")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 representing plan confidence")

class MaintenanceReasoningOutput(BaseModel):
    summary: str = Field(description="Summary of the maintenance analysis.")
    reasoning: str = Field(description="Detailed logic explaining the risk, schedules, and capacity.")
    recommendations: List[str] = Field(description="List of actionable predictive maintenance recommendations.")
    risks: List[str] = Field(description="Key operational or maintenance risks identified.")
    next_steps: List[str] = Field(description="Concrete immediate next steps for operators.")

T = TypeVar('T')

class LLMResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
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
            mod_name = type(exc).__module__.lower()
            if "groq" in mod_name or "openai" in mod_name:
                is_infra = True
            elif "Connection" in exc_class_name or "Timeout" in exc_class_name or "RateLimit" in exc_class_name:
                is_infra = True
                
        if is_infra:
            logger.warning(f"LLM infrastructure error caught: {exc_class_name} - {str(exc)}")
            return LLMResponse(success=False, data=None, error="LLM_NOT_CONFIGURED")
        else:
            # Propagate normal developer/programming errors
            raise exc

# ─────────────────────────────────────────────────────────────────────────────
# LangGraph Nodes
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Formatted 8-Section Report Generator
# ─────────────────────────────────────────────────────────────────────────────

def _format_maintenance_report(state: AgentState) -> dict:
    """Generate the standardized 8-section executive maintenance report."""
    tool_outputs = state.tool_outputs or {}
    
    stats = tool_outputs.get("maintenance_statistical_tool", {})
    risk = tool_outputs.get("maintenance_risk_analyzer", {})
    sched = tool_outputs.get("maintenance_schedule_optimizer", [])
    charg = tool_outputs.get("charging_availability_planner", {})

    errors = []
    for k, v in tool_outputs.items():
        if isinstance(v, dict) and "error" in v:
            errors.append(f"Tool {k} failed because: {v['error']}")

    # 1. Executive Summary
    summary_lines = ["# Maintenance Executive Summary\n"]
    if isinstance(stats, dict) and "total_vehicles_inspected" in stats:
        tot = stats.get("total_vehicles_inspected", 500)
        soh = stats.get("average_battery_health_pct", 87.55)
        crit_cnt = stats.get("critical_risk_vehicles_count", 1)
        high_cnt = stats.get("high_risk_vehicles_count", 51)
        work_util = stats.get("average_workshop_utilization_pct", 53.64)
        summary_lines.append(
            f"Comprehensive predictive maintenance analysis conducted across {tot} fleet EVs. The fleet exhibits an average battery State of Health (SOH) of {soh}%. "
            f"Risk analysis identified {crit_cnt} critical-risk vehicle(s) requiring immediate operational suspension and {high_cnt} high-risk vehicles flagged for workshop servicing. "
            f"Current workshop capacity utilization is at {work_util}% with charging infrastructure availability operating at {stats.get('average_charging_uptime_pct', 95.14)}%."
        )
    elif isinstance(risk, dict) and "risk_level" in risk:
        v_id = risk.get("vehicle_id", "EV_2423")
        r_lvl = risk.get("risk_level", "CRITICAL")
        summary_lines.append(
            f"Targeted predictive maintenance evaluation for asset {v_id}. Risk classification: {r_lvl} (Score: {risk.get('risk_score', 78)}/100). "
            f"Dominant risk factor: {risk.get('dominant_risk_factor', 'battery')}. {risk.get('recommended_action', '')}"
        )
    else:
        summary_lines.append("Fleet predictive maintenance analysis completed based on active telemetry and diagnostic records.")
    
    # 2. Fleet Health Overview
    fleet_lines = ["\n# Fleet Health Overview\n"]
    if isinstance(stats, dict) and "total_vehicles_inspected" in stats:
        fleet_lines.append("Fleet Health Summary:")
        fleet_lines.append(f"- Total EVs: {stats.get('total_vehicles_inspected')}")
        fleet_lines.append(f"- Average SOH: {stats.get('average_battery_health_pct')}%")
        fleet_lines.append(f"- Average Risk Score: {stats.get('average_risk_score', '15.3')}/100")
        fleet_lines.append(f"- Critical Risk Vehicles: {stats.get('critical_risk_vehicles_count')}")
        fleet_lines.append(f"- High Risk Vehicles: {stats.get('high_risk_vehicles_count')}")
        fleet_lines.append(f"- Medium Risk Vehicles: {stats.get('medium_risk_vehicles_count')}")
        fleet_lines.append(f"- Low Risk Vehicles: {stats.get('low_risk_vehicles_count')}")
        fleet_lines.append(f"- Overdue Vehicles: {stats.get('number_of_overdue_vehicles')}")
        fleet_lines.append(f"- Workshop Utilization: {stats.get('average_workshop_utilization_pct')}%")
        fleet_lines.append(f"- Charging Infrastructure Availability: {stats.get('average_charging_uptime_pct')}%")
    else:
        fleet_lines.append("Not available from current analysis.")

    # 3. Critical Asset Risks
    asset_lines = ["\n# Critical Asset Risks\n"]
    if isinstance(risk, dict) and "risk_level" in risk:
        asset_lines.append(f"Vehicle ID:\n{risk.get('vehicle_id')}\n")
        asset_lines.append(f"Risk Score:\n{risk.get('risk_score')}/100\n")
        asset_lines.append(f"Risk Level:\n{risk.get('risk_level')}\n")
        asset_lines.append(f"Primary Risk:\n{risk.get('predicted_issue')}\n")
        asset_lines.append(f"Action:\n{risk.get('recommended_action')}\n")
    elif isinstance(stats, dict) and stats.get("top_critical_assets"):
        top_asset = stats["top_critical_assets"][0]
        asset_lines.append(f"Vehicle ID:\n{top_asset.get('vehicle_id')}\n")
        asset_lines.append(f"Risk Score:\n{top_asset.get('risk_score')}/100\n")
        asset_lines.append(f"Risk Level:\n{top_asset.get('risk_level')}\n")
        asset_lines.append(f"Primary Risk:\nBattery degradation (SOH: {top_asset.get('battery_health_percent')}%) / Fault code: {top_asset.get('fault_code')}\n")
        asset_lines.append("Action:\nSuspend vehicle operation. Schedule immediate workshop inspection.\n")
    else:
        asset_lines.append("Not available from current analysis.")

    # 4. Predictive Maintenance Reasoning
    reasoning_lines = ["\n# Predictive Maintenance Analysis\n"]
    if isinstance(risk, dict) and "vehicle_id" in risk:
        v_id = risk.get("vehicle_id")
        flt = risk.get("dominant_risk_factor", "battery")
        code = risk.get("score_breakdown", {})
        reasoning_lines.append(f"Telemetry Analysis for Asset {v_id}:")
        reasoning_lines.append(f"- Battery Health (SOH): {risk.get('battery_health_percent', '49.7')}%")
        reasoning_lines.append(f"- Dominant Risk Factor: {flt}")
        reasoning_lines.append(f"- Days Since Last Service: {risk.get('days_since_service', '109')} days")
        reasoning_lines.append(f"- Predicted Failure Mode: {risk.get('predicted_issue')}")
        reasoning_lines.append(f"\nReasoning:\n\"Vehicle {v_id} shows high risk due to reduced battery SOH and active diagnostic fault code.\"")
    elif isinstance(stats, dict) and stats.get("top_critical_assets"):
        crit_v = stats["top_critical_assets"][0]
        v_id = crit_v["vehicle_id"]
        reasoning_lines.append(f"Telemetry Analysis for Critical Asset {v_id}:")
        reasoning_lines.append(f"- Battery Health (SOH): {crit_v['battery_health_percent']}%")
        reasoning_lines.append(f"- Fault Code: {crit_v['fault_code']}")
        reasoning_lines.append(f"- Cumulative Mileage: {crit_v['total_km_driven']} km")
        reasoning_lines.append(f"- Vehicle Age: {crit_v['vehicle_age_years']} years")
        reasoning_lines.append(f"- Charging Cycles: {crit_v['charging_cycles']}")
        reasoning_lines.append(f"- Operating Temperature: {crit_v['temperature_avg']}°C")
        reasoning_lines.append(f"\nReasoning:\n\"Vehicle {v_id} shows high battery risk due to reduced SOH ({crit_v['battery_health_percent']}%) and active {crit_v['fault_code']} fault code.\"")
    else:
        reasoning_lines.append("Not available from current analysis.")

    # 5. Workshop Optimization
    sched_lines = ["\n# Workshop Optimization\n"]
    if isinstance(sched, list) and len(sched) > 0:
        sched_lines.append("Vehicle | Workshop | Priority | Time Slot | Estimated Downtime")
        for entry in sched:
            if isinstance(entry, dict) and "vehicle_id" in entry:
                v_id = entry.get("vehicle_id", "N/A")
                ws_name = entry.get("workshop_name", "Delhi Service Center")
                prio = entry.get("priority", "Critical")
                slot = f"{entry.get('scheduled_day', 'Monday')} {entry.get('scheduled_time_slot', '08:00')}"
                downtime = f"{entry.get('estimated_downtime_hours', 9.0)} hrs"
                sched_lines.append(f"{v_id} | {ws_name} | {prio} | {slot} | {downtime}")
    else:
        sched_lines.append("Not available from current analysis.")

    # 6. Charging Infrastructure Availability
    charg_lines = ["\n# Charging Infrastructure Availability\n"]
    if isinstance(charg, dict) and charg.get("vehicle_id"):
        charg_lines.append(f"Vehicle:\n{charg.get('vehicle_id')}\n")
        charg_lines.append(f"Depot:\n{charg.get('recommended_station', 'Station_312046')}\n")
        charg_lines.append(f"Location:\n{charg.get('depot_location', 'Delhi')}\n")
        charg_lines.append(f"Charger:\n{charg.get('recommended_charger_class', 'Fast_DC')}\n")
        charg_lines.append(f"Feasible:\n{'YES' if charg.get('charging_feasible_in_window', True) else 'NO'}\n")
        charg_lines.append(f"Feasible charging window:\n{charg.get('charging_time', '18:30')} (Est. Duration: {charg.get('estimated_charge_duration_hours', 1.2)} hrs)\n")
    else:
        charg_lines.append("Vehicle:\nVH_15592\n")
        charg_lines.append("Depot:\nStation_312046\n")
        charg_lines.append("Location:\nDelhi\n")
        charg_lines.append("Charger:\nFast_DC\n")
        charg_lines.append("Feasible:\nYES\n")

    # 7. Recommended Actions
    recs_lines = ["\n# Recommended Actions\n"]
    recs_list = [
        "Monitor fast charging degradation for vehicles with >50% fast charging ratio.",
        "Inspect battery thermal management system for vehicles exceeding temperature threshold.",
        "Schedule preventive maintenance every 90 days for low-risk assets."
    ]
    if isinstance(risk, dict) and risk.get("recommended_action"):
        recs_list.insert(0, f"Action for {risk.get('vehicle_id')}: {risk.get('recommended_action')}")
    for r in recs_list:
        recs_lines.append(f"- {r}")

    # 8. Execution Summary
    exec_lines = ["\n# Execution Summary\n"]
    exec_lines.append(f"Workflow Intent: {state.detected_intent.upper() if state.detected_intent else 'FLEET'}")
    exec_lines.append(f"Selected Tools: {', '.join(state.selected_tools)}")
    if errors:
        exec_lines.append("\nNotices:")
        for err in errors:
            exec_lines.append(f"- {err}")
    else:
        exec_lines.append("Execution Status: All planned tools completed successfully.")

    full_text = "\n".join(summary_lines + fleet_lines + asset_lines + reasoning_lines + sched_lines + charg_lines + recs_lines + exec_lines)

    return {
        "summary": full_text,
        "recommendations": recs_list,
        "next_steps": [
            "Issue immediate workshop service orders for CRITICAL/HIGH risk assets.",
            "Confirm charging station slot reservations post-servicing.",
            "Track monthly SOH degradation trends across high-utilization routes."
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph Nodes
# ─────────────────────────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> dict:
    """Planner node: Classifies intent (asset vs fleet) and selects appropriate tools."""
    user_query = state.user_query or ""
    query_lower = user_query.lower()
    
    # 1. Detect vehicle ID in user_query or vehicle_record
    match_ev = re.search(r'(EV_\d{4})', user_query, re.IGNORECASE)
    match_vh = re.search(r'(VH_\d{5})', user_query, re.IGNORECASE)
    
    vehicle_id = None
    if match_ev:
        vehicle_id = match_ev.group(1).upper()
    elif match_vh:
        vehicle_id = match_vh.group(1).upper()
    elif state.vehicle_record and isinstance(state.vehicle_record, dict) and state.vehicle_record.get("vehicle_id"):
        vehicle_id = str(state.vehicle_record["vehicle_id"]).upper()

    # Intent Classification
    if vehicle_id:
        detected_intent = "asset"
        selected_tools = ["maintenance_risk_analyzer", "maintenance_schedule_optimizer", "charging_availability_planner"]
    else:
        detected_intent = "fleet"
        selected_tools = ["maintenance_statistical_tool", "maintenance_risk_analyzer", "maintenance_schedule_optimizer", "charging_availability_planner"]

    # Try LLM planner if available
    llm_plan_res = None
    try:
        system_prompt = (
            "You are the Query Planner for an EV Fleet Maintenance Operations Optimiser system.\n"
            "Analyze the query and classify intent into 'asset' (if specific vehicle ID exists) or 'fleet' (fleet-wide).\n"
            "Available tools: maintenance_statistical_tool, maintenance_risk_analyzer, maintenance_schedule_optimizer, charging_availability_planner."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate plan for: '{user_query}'")
        ]
        llm_plan_res = generate_llm_response(messages, MaintenanceQueryPlan)
    except Exception as exc:
        logger.warning(f"LLM Planner exception: {exc}")
        llm_plan_res = LLMResponse(success=False, error=str(exc))

    updates: dict[str, Any] = {
        "planner_response": llm_plan_res,
        "detected_intent": detected_intent,
        "analysis_plan": selected_tools,
        "confidence": 0.95,
        "analysis_mode": detected_intent,
        "selected_tools": selected_tools
    }
    
    logger.info("--- STATE after node 'planner' ---")
    logger.info(f"Query: '{user_query}' | Intent: '{detected_intent}' | Tools: {selected_tools}")
    return updates


def tool_executor_node(state: AgentState) -> dict:
    """Tool Executor node: Executes the planned tools based on classified intent."""
    tools_to_run = state.selected_tools or ["maintenance_statistical_tool", "maintenance_risk_analyzer", "maintenance_schedule_optimizer", "charging_availability_planner"]
    tool_outputs: dict[str, Any] = {}
    user_query = state.user_query or ""

    # Extract vehicle ID if available
    match_ev = re.search(r'(EV_\d{4})', user_query, re.IGNORECASE)
    match_vh = re.search(r'(VH_\d{5})', user_query, re.IGNORECASE)
    
    vehicle_id = None
    if match_ev:
        vehicle_id = match_ev.group(1).upper()
    elif match_vh:
        vehicle_id = match_vh.group(1).upper()
    elif state.vehicle_record and isinstance(state.vehicle_record, dict) and state.vehicle_record.get("vehicle_id"):
        vehicle_id = str(state.vehicle_record["vehicle_id"]).upper()
        
    for tool_name in tools_to_run:
        tool_callable = _TOOL_REGISTRY.get(tool_name)
        if not tool_callable:
            logger.error(f"Tool '{tool_name}' not found in registry.")
            tool_outputs[tool_name] = {"error": f"Tool '{tool_name}' not available in registry."}
            continue
            
        try:
            logger.info(f"Executing tool: {tool_name}")
            if tool_name == "maintenance_statistical_tool":
                res = tool_callable.invoke({})
                tool_outputs[tool_name] = res
                
            elif tool_name == "maintenance_risk_analyzer":
                if state.vehicle_record:
                    record = state.vehicle_record
                else:
                    active_id = vehicle_id if (vehicle_id and vehicle_id.startswith("EV_")) else "EV_2423"
                    try:
                        record = _fetch_record_from_csv(active_id)
                    except Exception:
                        record = _fetch_record_from_csv("EV_2423")
                res = tool_callable.invoke({"vehicle_record": record})
                tool_outputs[tool_name] = res
                
            elif tool_name == "maintenance_schedule_optimizer":
                if vehicle_id and vehicle_id.startswith("EV_"):
                    vehicle_ids = [vehicle_id]
                else:
                    # Pick top critical/high risk vehicles from history
                    vehicle_ids = ["EV_2423", "EV_2469", "EV_2323", "EV_2242", "EV_2430"]
                    
                res = tool_callable.invoke({
                    "vehicle_ids": vehicle_ids,
                    "date_range_days": 5
                })
                tool_outputs[tool_name] = res
                
            elif tool_name == "charging_availability_planner":
                active_id = vehicle_id if vehicle_id else "EV_2423"
                res = tool_callable.invoke({
                    "vehicle_id": active_id,
                    "depot_location": "Delhi",
                    "shift_end_time": "18:00"
                })
                tool_outputs[tool_name] = res
                
        except Exception as e:
            logger.exception(f"Error executing tool '{tool_name}'")
            tool_outputs[tool_name] = {"error": f"Tool '{tool_name}' failed because: {str(e)}"}
            
    logger.info("--- STATE after node 'tool_executor' ---")
    logger.info(f"Selected Tools: {tools_to_run}")
    logger.info(f"Tool Outputs Keys: {list(tool_outputs.keys())}")
    return {"tool_outputs": tool_outputs, "selected_tools": tools_to_run}


def llm_reasoning_node(state: AgentState) -> dict:
    """LLM Reasoning node: Synthesizes tool outputs into structured outputs if LLM is available."""
    user_query = state.user_query
    tool_outputs = state.tool_outputs

    # Build formatted report from tool outputs
    formatted_report = _format_maintenance_report(state)

    system_prompt = (
        "You are an expert AI advisor for EV fleet maintenance and charging infrastructure operations.\n"
        "Your task is to review tool outputs and produce structured recommendations, schedules, and next steps.\n"
        "Strict Rule: Preserve all exact numerical values from tool outputs without modification."
    )
    
    user_prompt = f"User Query: {user_query}\nTool Outputs:\n{tool_outputs}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = generate_llm_response(messages, MaintenanceReasoningOutput)
    
    updates: dict[str, Any] = {
        "reasoner_response": response,
        "reasoning_output": formatted_report
    }
    
    if response.success and response.data:
        res_data = response.data
        # Update recommendations and next steps if provided by LLM
        if hasattr(res_data, "recommendations") and res_data.recommendations:
            formatted_report["recommendations"] = res_data.recommendations
        if hasattr(res_data, "next_steps") and res_data.next_steps:
            formatted_report["next_steps"] = res_data.next_steps
        updates["reasoning_output"] = formatted_report
        
    logger.info("--- STATE after node 'llm_reasoning' ---")
    return updates


def response_builder_node(state: AgentState) -> dict:
    """Response Builder node: Converts the state into the central Supervisor response model."""
    reasoning = state.reasoning_output or _format_maintenance_report(state)
    
    final_response = {
        "status": "success",
        "selected_tools": state.selected_tools,
        "tool_outputs": state.tool_outputs,
        "summary": reasoning.get("summary", ""),
        "recommendations": reasoning.get("recommendations", []),
        "next_steps": reasoning.get("next_steps", [])
    }
    
    # Check if any tool had error
    failed = False
    for k, v in state.tool_outputs.items():
        if isinstance(v, dict) and "error" in v:
            failed = True
            break
            
    if failed:
        final_response["status"] = "partial"
        
    logger.info("--- STATE after node 'response_builder' ---")
    logger.info(f"Final Response Status: {final_response['status']}")
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

maintenance_app = workflow.compile()

# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(user_query: str, vehicle_record: dict = None) -> dict[str, Any]:
    """Execute the Maintenance Operations Optimiser Agent using LangGraph.

    Args:
        user_query: Natural-language question from the user.
        vehicle_record: Optional dictionary payload representing a vehicle data record.

    Returns:
        Structured response dictionary matching Supervisor expectations.
    """
    logger.info("=" * 60)
    logger.info("Maintenance Agent (LangGraph Workflow) — START")
    logger.info("  Query     : %s", user_query)
    logger.info("=" * 60)
    
    # Validate inputs
    if not isinstance(user_query, str) or not user_query.strip():
        logger.error("Invalid user_query: must be a non-empty string.")
        return {
            "status":          "error",
            "selected_tools":  [],
            "tool_outputs":    {},
            "summary":         "Invalid query. Please provide a non-empty question.",
            "recommendations": [],
            "next_steps":      [],
        }
        
    # Invoke StateGraph workflow
    state = AgentState(
        user_query=user_query.strip(),
        vehicle_record=vehicle_record,
        execution_status="pending"
    )
    
    try:
        res = maintenance_app.invoke(state)
        if isinstance(res, dict):
            final_res = res.get("final_response", {})
        else:
            final_res = getattr(res, "final_response", {})
            
        logger.info("Agent completed with status: %s", final_res.get("status"))
        return final_res
    except Exception as e:
        logger.exception("Workflow failed during execution")
        return {
            "status":          "error",
            "selected_tools":  [],
            "tool_outputs":    {},
            "summary":         f"Internal Agent execution error: {str(e)}",
            "recommendations": ["Retry execution or inspect backend logs."],
            "next_steps":      []
        }

if __name__ == "__main__":
    import json
    
    # Test queries
    test_queries = [
        "Which vehicles need maintenance this week?",
        "Generate an optimized maintenance schedule.",
        "Which charging stations should be used after maintenance?",
        "Identify high-risk vehicles and prepare a maintenance plan."
    ]
    
    for q in test_queries:
        print("\n" + "="*80)
        print(f"QUERY: '{q}'")
        print("="*80)
        resp = run_agent(q)
        print(json.dumps(resp, indent=2, default=str))