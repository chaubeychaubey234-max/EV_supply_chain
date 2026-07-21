import os
import logging
import re
import json
from typing import List, Optional, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from .state import APMState
from .tools.battery_tools import fetch_battery_health, predict_battery_health, aggregate_apm_statistics
from .tools.thermal_tools import fetch_thermal_events
from .tools.charging_tools import fetch_charging_patterns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ev_apm_agent")

_TOOL_REGISTRY = {
    "fetch_battery_health": fetch_battery_health,
    "predict_battery_health": predict_battery_health,
    "fetch_thermal_events": fetch_thermal_events,
    "fetch_charging_patterns": fetch_charging_patterns,
    "aggregate_apm_statistics": aggregate_apm_statistics
}

class APMReasoningOutput(BaseModel):
    summary: str = Field(description="Concise answer to the user's question, grounded in fleet data where relevant.")
    explanation: str = Field(description="Detailed explanation with domain insights. Reference specific numbers from the context when relevant.")
    recommendations: List[str] = Field(description="Actionable recommendations for fleet operators or engineers.")
    reasoning: str = Field(description="Step-by-step reasoning that led to the conclusions.")
    maintenance_triggers: List[str] = Field(description="Specific maintenance triggers or warnings to flag.")

def generate_llm_response(prompt_messages: list, response_model: Any = None) -> Any:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key.startswith("dummy"):
        raise ValueError("GROQ_API_KEY is missing or invalid.")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    if response_model:
        return llm.with_structured_output(response_model).invoke(prompt_messages)
    return llm.invoke(prompt_messages)

# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT LOADER — always runs, loads fleet-wide stats as RAG grounding
# ─────────────────────────────────────────────────────────────────────────────
def _load_fleet_context() -> dict:
    """Load full fleet aggregate stats to use as LLM context for any query."""
    try:
        return aggregate_apm_statistics.invoke({"metric": "all"})
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# PLANNER — heuristic routing, never depends on LLM
# ─────────────────────────────────────────────────────────────────────────────
def planner_node(state: APMState) -> dict:
    """
    Route by heuristic priority:
      1. Live telemetry params provided → predict_battery_health
      2. Specific EV-XXXX in query/state → fetch that asset's records
      3. Everything else → no extra tools needed; fleet context always loaded in reasoning
    """
    user_query = (state.get("user_query") or state.get("query") or "").strip()

    # Rule 1: live telemetry prediction
    if state.get("avg_temperature_c") is not None:
        return {
            "user_query": user_query,
            "detected_intent": "prediction",
            "analysis_mode": "live_telemetry_prediction",
            "analysis_plan": ["predict_battery_health"],
            "confidence": 1.0,
            "tool_outputs": {}
        }

    # Rule 2: specific EV ID
    ev_match = re.search(r"\b(EV-\d+)\b", user_query, re.IGNORECASE)
    ev_id = (ev_match.group(0).upper() if ev_match else None) or state.get("ev_id")
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

    # Rule 3: general query — no extra per-asset tools needed; fleet context covers it
    return {
        "user_query": user_query,
        "detected_intent": "general",
        "analysis_mode": "fleet_context_rag",
        "analysis_plan": [],   # no extra tools — fleet context always injected in reasoning
        "confidence": 1.0,
        "tool_outputs": {}
    }


def tool_executor_node(state: APMState) -> dict:
    """Executes any asset-specific or prediction tools from the plan."""
    tools_to_run = state.get("analysis_plan") or []
    tool_outputs = {}
    ev_id = state.get("ev_id")

    for tool_name in tools_to_run:
        tool_callable = _TOOL_REGISTRY.get(tool_name)
        if not tool_callable:
            tool_outputs[tool_name] = {"error": f"Tool '{tool_name}' not found."}
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
            else:
                result = tool_callable.invoke({"ev_id": ev_id})
            tool_outputs[tool_name] = result
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}")
            tool_outputs[tool_name] = {"error": str(e)}

    return {"tool_outputs": tool_outputs}


def llm_reasoning_node(state: APMState) -> dict:
    """
    Always loads fleet-wide context as RAG grounding, then answers the user's
    question using that data + any per-asset tool outputs.
    Works for ANY query — conceptual, statistical, operational, or comparative.
    """
    user_query = state.get("user_query") or "No query provided."
    detected_intent = state.get("detected_intent") or "general"
    tool_outputs = state.get("tool_outputs") or {}
    ev_id = state.get("ev_id")

    # Always load fleet context as background knowledge
    fleet_context = _load_fleet_context()

    system_prompt = """You are VoltGrid's EV Battery Intelligence Agent — an expert in EV battery asset performance management, degradation science, thermal management, and predictive maintenance.

You have access to LIVE DATA from your company's fleet dataset (loaded below as FLEET CONTEXT). You must use this data to ground all your answers — whether the question is conceptual, statistical, operational, or analytical.

**Your behavior rules:**
1. **Always use fleet context** — When answering any question, reference relevant numbers from the Fleet Context to make your answer grounded and specific to this company's fleet.
2. **Conceptual questions** — Explain the concept AND relate it to the actual fleet data (e.g., "Thermal runaway occurs when... In our fleet, we currently have X EVs at high thermal risk with max recorded temp of Y°C").
3. **Statistical questions** — Answer directly from the Fleet Context data. Do not hedge — the numbers are real.
4. **Specific EV questions** — Use the Per-Asset Tool Outputs to give a focused analysis of that EV, and compare to fleet averages from the Fleet Context.
5. **Operational/advisory questions** — Use fleet data to back your recommendations with real numbers.
6. Never say "I don't have access to data" — you always have the Fleet Context.
7. Be specific, cite actual numbers, be concise but complete."""

    # Build fleet context block
    fleet_ctx_str = json.dumps(fleet_context, indent=2) if fleet_context else "Fleet context unavailable."

    # Build per-asset outputs block
    asset_outputs_str = ""
    if tool_outputs:
        asset_outputs_str = "\n\nPER-ASSET TOOL OUTPUTS (specific EV data):\n"
        for tool_name, output in tool_outputs.items():
            asset_outputs_str += f"\n[{tool_name}]\n{json.dumps(output, indent=2)}\n"

    user_prompt = f"""USER QUESTION: {user_query}
QUERY MODE: {detected_intent}
{f"SPECIFIC EV: {ev_id}" if ev_id else ""}

FLEET CONTEXT (live aggregate data from company's fleet dataset):
{fleet_ctx_str}
{asset_outputs_str}

Answer the user's question completely. Ground your answer in the fleet data above. Be specific — use real numbers."""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    try:
        result = generate_llm_response(messages, APMReasoningOutput)
        return {"reasoning_output": result.model_dump()}
    except Exception as e:
        logger.error(f"LLM Reasoning failed: {e}")
        # Fallback: build a data-only summary from fleet context
        fallback_summary = (
            f"Fleet stats: {fleet_context.get('total_evs_inspected', 'N/A')} EVs, "
            f"avg SoH {fleet_context.get('average_state_of_health_pct', 'N/A')}%, "
            f"avg degradation {fleet_context.get('average_degradation_rate_monthly_pct', 'N/A')}%/month, "
            f"{fleet_context.get('critical_evs_count', 'N/A')} critical EVs."
        )
        return {"reasoning_output": {"summary": fallback_summary, "explanation": "", "recommendations": [], "reasoning": "", "maintenance_triggers": []}}


def response_builder_node(state: APMState) -> dict:
    """Builds the final response. Always provides real fleet data for dashboard charts."""
    tool_outputs = state.get("tool_outputs") or {}
    reasoning = state.get("reasoning_output") or {}
    query_type = state.get("detected_intent") or "general"

    # Battery analysis: per-asset → prediction → fleet aggregate
    if "predict_battery_health" in tool_outputs and "error" not in tool_outputs["predict_battery_health"]:
        battery_analysis = tool_outputs["predict_battery_health"]
    elif "fetch_battery_health" in tool_outputs and "error" not in tool_outputs["fetch_battery_health"]:
        battery_analysis = tool_outputs["fetch_battery_health"]
    else:
        # General query: load fleet aggregate for chart population
        fleet = _load_fleet_context()
        battery_analysis = {
            "state_of_health_percentage": fleet.get("average_state_of_health_pct", 0.0),
            "degradation_rate_per_month": fleet.get("average_degradation_rate_monthly_pct", 0.0),
            "remaining_useful_life_months": 0,
            "status": "Fleet Average",
            "source": "fleet_aggregate"
        }

    # Safety analysis: per-asset → fleet aggregate
    if "fetch_thermal_events" in tool_outputs and "error" not in tool_outputs["fetch_thermal_events"]:
        safety_analysis = tool_outputs["fetch_thermal_events"]
    else:
        fleet = _load_fleet_context()
        safety_analysis = {
            "average_operating_temperature_celsius": fleet.get("average_operating_temperature_celsius", 0.0),
            "max_recorded_temperature_celsius": fleet.get("max_operating_temperature_celsius", 0.0),
            "thermal_runaway_warnings": fleet.get("high_thermal_risk_evs_count", 0),
            "cooling_system_status": "Fleet Average"
        }

    # Telemetry: per-asset → fleet aggregate
    if "fetch_charging_patterns" in tool_outputs and "error" not in tool_outputs["fetch_charging_patterns"]:
        telemetry_data = tool_outputs["fetch_charging_patterns"]
    else:
        fleet = _load_fleet_context()
        telemetry_data = {
            "fast_charging_ratio_percentage": fleet.get("average_fast_charge_ratio_pct", 0.0),
            "deep_discharge_cycles_last_month": int(fleet.get("average_deep_discharge_cycles_monthly", 0)),
            "average_charge_duration_hours": fleet.get("average_charge_duration_hours", 0.0)
        }

    # Recommendations & messages
    recs = []
    if reasoning.get("summary"):
        recs.append(reasoning["summary"])
    if reasoning.get("explanation"):
        recs.append(reasoning["explanation"])
    recs.extend(reasoning.get("recommendations", []))

    messages = [f"Analysis completed ({query_type})"]
    if reasoning.get("summary"):
        messages.append(reasoning["summary"])

    return {
        "telemetry_data": telemetry_data,
        "battery_analysis": battery_analysis,
        "safety_analysis": safety_analysis,
        "recommendations": recs,
        "maintenance_triggers": reasoning.get("maintenance_triggers", []),
        "messages": messages
    }


# Build graph
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
