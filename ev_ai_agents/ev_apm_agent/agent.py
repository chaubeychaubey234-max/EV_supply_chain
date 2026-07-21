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

# Module-level cache — computed once at startup, reused for every request
_FLEET_CONTEXT_CACHE: dict = {}

# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT LOADER — always runs, loads fleet-wide stats as RAG grounding
# ─────────────────────────────────────────────────────────────────────────────
def _load_fleet_context() -> dict:
    """Load fleet aggregate stats, caching the result for the lifetime of the process."""
    global _FLEET_CONTEXT_CACHE
    if not _FLEET_CONTEXT_CACHE:
        try:
            _FLEET_CONTEXT_CACHE = aggregate_apm_statistics.invoke({"metric": "all"})
        except Exception:
            pass
    return _FLEET_CONTEXT_CACHE


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


def _compact_context(ctx: dict) -> str:
    """Convert context dict to a compact single-line string to save tokens."""
    parts = []
    for k, v in ctx.items():
        if isinstance(v, dict):
            # Inline nested dicts as key:value pairs
            inner = ", ".join(f"{ik}={iv}" for ik, iv in v.items())
            parts.append(f"{k}=[{inner}]")
        elif isinstance(v, list):
            parts.append(f"{k}=[{', '.join(str(x) for x in v[:5])}]")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)


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
    fleet_ctx_str = _compact_context(fleet_context) if fleet_context else "Fleet context unavailable."

    # Build per-asset outputs (compact)
    asset_lines = []
    for tool_name, output in tool_outputs.items():
        asset_lines.append(f"[{tool_name}]: {_compact_context(output) if isinstance(output, dict) else output}")
    asset_str = "\n".join(asset_lines)

    system_prompt = (
        "You are VoltGrid's EV Battery Intelligence Agent — expert in EV battery health, degradation, thermal management, and predictive maintenance.\n"
        "You have live fleet data below. Use it to ground EVERY answer with real numbers from this fleet.\n"
        "Rules: (1) Always cite actual fleet numbers. (2) For conceptual questions, explain the concept AND reference fleet data. "
        "(3) For specific EVs, compare to fleet averages. (4) Never say you lack data — you have the Fleet Context."
    )

    user_prompt = (
        f"Question: {user_query}\n"
        f"Mode: {detected_intent}{(' | EV: ' + ev_id) if ev_id else ''}\n\n"
        f"FLEET CONTEXT: {fleet_ctx_str}\n"
        + (f"\nASSET DATA:\n{asset_str}" if asset_str else "") +
        "\n\nAnswer fully, cite real numbers, give actionable recommendations."
    )

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

    # Load fleet once (cached)
    fleet = _load_fleet_context()

    # Battery analysis: per-asset → prediction → fleet aggregate
    if "predict_battery_health" in tool_outputs and "error" not in tool_outputs["predict_battery_health"]:
        battery_analysis = tool_outputs["predict_battery_health"]
    elif "fetch_battery_health" in tool_outputs and "error" not in tool_outputs["fetch_battery_health"]:
        battery_analysis = tool_outputs["fetch_battery_health"]
    else:
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
        telemetry_data = {
            "fast_charging_ratio_percentage": fleet.get("average_fast_charge_ratio_pct", 0.0),
            "deep_discharge_cycles_last_month": int(fleet.get("average_deep_discharge_cycles_monthly", 0)),
            "average_charge_duration_hours": fleet.get("average_charge_duration_hours", 0.0)
        }

    # Recommendations: only actual action items, not summary/explanation
    recs = reasoning.get("recommendations", [])

    messages = [f"Analysis completed ({query_type})"]
    if reasoning.get("summary"):
        messages.append(reasoning["summary"])

    return {
        "reasoning_output": reasoning,   # expose directly so server can read summary/explanation
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
