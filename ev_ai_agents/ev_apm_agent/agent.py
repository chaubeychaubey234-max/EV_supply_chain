import os
import logging
import re
from typing import List, Optional, Any, Dict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from .state import APMState
from .tools.battery_tools import fetch_battery_health, predict_battery_health, aggregate_apm_statistics
from .tools.thermal_tools import fetch_thermal_events
from .tools.charging_tools import fetch_charging_patterns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ev_apm_agent")

# Define centralized Tool Registry
_TOOL_REGISTRY = {
    "fetch_battery_health": fetch_battery_health,
    "predict_battery_health": predict_battery_health,
    "fetch_thermal_events": fetch_thermal_events,
    "fetch_charging_patterns": fetch_charging_patterns,
    "aggregate_apm_statistics": aggregate_apm_statistics
}

class APMReasoningOutput(BaseModel):
    summary: str = Field(description="Summary of the battery health status.")
    explanation: str = Field(description="Detailed explanation of the analysis or concept.")
    recommendations: List[str] = Field(description="List of actionable recommendations.")
    reasoning: str = Field(description="Reasoning process for the conclusions.")
    maintenance_triggers: List[str] = Field(description="List of predictive maintenance triggers.")

# Isolated LLM execution helper
def generate_llm_response(prompt_messages: list, response_model: Any = None) -> Any:
    """Helper to run the ChatGroq model with optional structured output."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key.startswith("dummy"):
        raise ValueError("GROQ_API_KEY is missing or invalid. LLM execution cannot proceed.")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    if response_model:
        structured_llm = llm.with_structured_output(response_model)
        return structured_llm.invoke(prompt_messages)
    return llm.invoke(prompt_messages)

# ─────────────────────────────────────────────────────────────────────────────
# HEURISTIC PLANNER — no LLM needed to route. LLM is only used for reasoning.
# ─────────────────────────────────────────────────────────────────────────────
def planner_node(state: APMState) -> dict:
    """
    Classifies the query using pure heuristics and builds the execution plan.
    Rules (in priority order):
      1. If raw telemetry params are provided → predict_battery_health
      2. If a specific EV-XXXX id is in the query or state → fetch asset tools
      3. Otherwise (any general/conceptual/statistical query) → aggregate_apm_statistics
         so the LLM always has real fleet data as context to answer with.
    """
    user_query = (state.get("user_query") or state.get("query") or "").strip()

    # ── Rule 1: live telemetry prediction ──────────────────────────────────
    if state.get("avg_temperature_c") is not None:
        return {
            "user_query": user_query,
            "detected_intent": "prediction",
            "analysis_mode": "live_telemetry_prediction",
            "analysis_plan": ["predict_battery_health"],
            "confidence": 1.0,
            "tool_outputs": {}
        }

    # ── Rule 2: specific EV ID in query or state ────────────────────────────
    ev_match = re.search(r"\b(EV-\d+)\b", user_query, re.IGNORECASE)
    ev_id_from_query = ev_match.group(0).upper() if ev_match else None
    ev_id = ev_id_from_query or state.get("ev_id")

    if ev_id:
        return {
            "user_query": user_query,
            "ev_id": ev_id,
            "detected_intent": "asset",
            "analysis_mode": "asset_analysis",
            "analysis_plan": ["fetch_battery_health", "fetch_thermal_events", "fetch_charging_patterns"],
            "confidence": 1.0,
            "tool_outputs": {}
        }

    # ── Rule 3: general / statistical / conceptual — use fleet aggregation ──
    # Determine aggregation scope from keywords
    q_lower = user_query.lower()
    if any(k in q_lower for k in ["temp", "thermal", "heat", "cooling", "runaway"]):
        agg_metric = "temperature"
    elif any(k in q_lower for k in ["charg", "fast charge", "discharge", "cycle"]):
        agg_metric = "charging_cycles"
    elif any(k in q_lower for k in ["health", "soh", "degradat", "rul", "life"]):
        agg_metric = "battery_health"
    else:
        agg_metric = "all"

    return {
        "user_query": user_query,
        "detected_intent": "statistical",
        "analysis_mode": agg_metric,
        "analysis_plan": ["aggregate_apm_statistics"],
        "confidence": 0.9,
        "tool_outputs": {}
    }


def tool_executor_node(state: APMState) -> dict:
    """Executes the plan tools dynamically from the registry."""
    tools_to_run = state.get("analysis_plan") or []
    tool_outputs = {}
    ev_id = state.get("ev_id")
    agg_metric = state.get("analysis_mode") or "all"

    for tool_name in tools_to_run:
        tool_callable = _TOOL_REGISTRY.get(tool_name)
        if not tool_callable:
            tool_outputs[tool_name] = {"error": f"Tool '{tool_name}' not available."}
            continue

        try:
            logger.info(f"Executing tool: {tool_name}")
            if tool_name == "predict_battery_health":
                result = tool_callable.invoke({
                    "avg_temperature_c": state.get("avg_temperature_c", 25.0),
                    "fast_charge_ratio_pct": state.get("fast_charge_ratio_pct", 30.0),
                    "deep_discharge_cycles": state.get("deep_discharge_cycles", 5),
                    "avg_charge_duration_hours": state.get("avg_charge_duration_hours", 4.0),
                    "max_temperature_c": state.get("max_temperature_c", 45.0)
                })
            elif tool_name == "aggregate_apm_statistics":
                result = tool_callable.invoke({"metric": agg_metric})
            else:
                # asset-specific tools — ev_id is guaranteed to exist (Rule 2 above)
                result = tool_callable.invoke({"ev_id": ev_id})
            tool_outputs[tool_name] = result
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}")
            tool_outputs[tool_name] = {"error": str(e)}

    return {"tool_outputs": tool_outputs}


def llm_reasoning_node(state: APMState) -> dict:
    """Uses LLM to reason over tool outputs and answer the user query."""
    user_query = state.get("user_query") or "No query provided."
    detected_intent = state.get("detected_intent") or "statistical"
    tool_outputs = state.get("tool_outputs") or {}
    ev_id = state.get("ev_id")

    system_prompt = (
        "You are an expert AI for Industrial EV Battery Asset Performance Management.\n"
        "You have access to real fleet statistics and individual EV telemetry fetched directly from the company's dataset.\n"
        "Your role: Interpret the tool outputs below and answer the user's question clearly and completely.\n"
        "Rules:\n"
        "- Use the tool output numbers directly — do NOT recalculate them.\n"
        "- If data is aggregated fleet data, frame your answer around fleet-wide insights.\n"
        "- If data is for a specific EV, give a focused per-asset analysis.\n"
        "- For conceptual questions (no tool ran), answer using your EV domain expertise.\n"
        "- Always give actionable recommendations and maintenance triggers.\n"
        "- Be specific, cite the actual numbers from the tool outputs."
    )

    user_prompt = f"User Question: {user_query}\nAnalysis Mode: {detected_intent}\n"
    if ev_id:
        user_prompt += f"EV Analyzed: {ev_id}\n"
    user_prompt += "\nDataset Tool Outputs:\n"
    for tool_name, output in tool_outputs.items():
        user_prompt += f"\n[{tool_name}]\n{output}\n"
    user_prompt += "\nBased on the data above, provide: summary, explanation, recommendations, reasoning, and maintenance_triggers."

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    try:
        reasoning_result = generate_llm_response(messages, APMReasoningOutput)
        return {"reasoning_output": reasoning_result.model_dump()}
    except Exception as e:
        logger.error(f"LLM Reasoning failed: {e}")
        return {"reasoning_output": {}}


def response_builder_node(state: APMState) -> dict:
    """Builds the final response from tool outputs and LLM reasoning."""
    tool_outputs = state.get("tool_outputs") or {}
    reasoning = state.get("reasoning_output") or {}
    query_type = state.get("detected_intent") or "statistical"

    # ── Battery analysis ───────────────────────────────────────────────────
    if "predict_battery_health" in tool_outputs and "error" not in tool_outputs["predict_battery_health"]:
        battery_analysis = tool_outputs["predict_battery_health"]
    elif "fetch_battery_health" in tool_outputs and "error" not in tool_outputs["fetch_battery_health"]:
        battery_analysis = tool_outputs["fetch_battery_health"]
    elif "aggregate_apm_statistics" in tool_outputs and "error" not in tool_outputs["aggregate_apm_statistics"]:
        stats = tool_outputs["aggregate_apm_statistics"]
        battery_analysis = {
            "state_of_health_percentage": stats.get("average_state_of_health_pct", 0.0),
            "degradation_rate_per_month": stats.get("average_degradation_rate_monthly_pct", 0.0),
            "remaining_useful_life_months": 0,
            "status": "Fleet Average Analysis",
            "source": "fleet_dataset_aggregation"
        }
    else:
        battery_analysis = {"status": "No data", "state_of_health_percentage": 0.0, "degradation_rate_per_month": 0.0}

    # ── Safety analysis ────────────────────────────────────────────────────
    if "fetch_thermal_events" in tool_outputs and "error" not in tool_outputs["fetch_thermal_events"]:
        safety_analysis = tool_outputs["fetch_thermal_events"]
    elif "aggregate_apm_statistics" in tool_outputs and "error" not in tool_outputs["aggregate_apm_statistics"]:
        stats = tool_outputs["aggregate_apm_statistics"]
        safety_analysis = {
            "max_recorded_temperature_celsius": stats.get("max_operating_temperature_celsius", 0.0),
            "average_operating_temperature_celsius": stats.get("average_operating_temperature_celsius", 0.0),
            "thermal_runaway_warnings": stats.get("high_thermal_risk_evs_count", 0),
            "cooling_system_status": "Fleet Average Analysis"
        }
    else:
        safety_analysis = {"max_recorded_temperature_celsius": 0.0, "thermal_runaway_warnings": 0, "cooling_system_status": "OK"}

    # ── Telemetry data ─────────────────────────────────────────────────────
    if "fetch_charging_patterns" in tool_outputs and "error" not in tool_outputs["fetch_charging_patterns"]:
        telemetry_data = tool_outputs["fetch_charging_patterns"]
    elif "aggregate_apm_statistics" in tool_outputs and "error" not in tool_outputs["aggregate_apm_statistics"]:
        stats = tool_outputs["aggregate_apm_statistics"]
        telemetry_data = {
            "fast_charging_ratio_percentage": stats.get("average_fast_charge_ratio_pct", 0.0),
            "deep_discharge_cycles_last_month": int(stats.get("average_deep_discharge_cycles_monthly", 0)),
            "average_charge_duration_hours": stats.get("average_charge_duration_hours", 0.0)
        }
    else:
        telemetry_data = {"fast_charging_ratio_percentage": 0.0}

    # ── Recommendations & triggers ─────────────────────────────────────────
    recs = []
    if reasoning.get("summary"):
        recs.append(f"Summary: {reasoning['summary']}")
    if reasoning.get("explanation"):
        recs.append(f"Explanation: {reasoning['explanation']}")
    for r in reasoning.get("recommendations", []):
        recs.append(r)

    triggers = reasoning.get("maintenance_triggers", [])
    messages = [f"Analysis completed for query type: {query_type}"]
    if reasoning.get("summary"):
        messages.append(reasoning["summary"])

    return {
        "telemetry_data": telemetry_data,
        "battery_analysis": battery_analysis,
        "safety_analysis": safety_analysis,
        "recommendations": recs,
        "maintenance_triggers": triggers,
        "messages": messages
    }


# Build the graph
workflow = StateGraph(APMState)
workflow.add_node("planner", planner_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("llm_reasoning", llm_reasoning_node)
workflow.add_node("response_builder", response_builder_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "tool_executor")
workflow.add_edge("tool_executor", "llm_reasoning")
workflow.add_edge("llm_reasoning", "response_builder")
workflow.add_edge("response_builder", END)

apm_app = workflow.compile()
