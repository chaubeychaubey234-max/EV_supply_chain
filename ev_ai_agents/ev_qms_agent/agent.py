import os
import logging
import re
from typing import List, Optional, Any, Dict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from .state import QMSState
from .tools.material_tools import fetch_material_data, predict_quality_drift
from .tools.process_tools import fetch_process_data
from .tools.inspection_tools import fetch_inspection_data, aggregate_qms_statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ev_qms_agent")

# Define centralized Tool Registry
_TOOL_REGISTRY = {
    "fetch_material_data": fetch_material_data,
    "predict_quality_drift": predict_quality_drift,
    "fetch_process_data": fetch_process_data,
    "fetch_inspection_data": fetch_inspection_data,
    "aggregate_qms_statistics": aggregate_qms_statistics
}

class QMSReasoningOutput(BaseModel):
    summary: str = Field(description="Summary of the manufacturing quality status.")
    explanation: str = Field(description="Detailed explanation of the quality drift/root cause.")
    recommendations: List[str] = Field(description="List of corrective actions or alerts for operators.")
    reasoning: str = Field(description="Reasoning process for the root cause and drift analysis.")

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
def planner_node(state: QMSState) -> dict:
    """
    Classifies the query using pure heuristics and builds the execution plan.
    Rules (in priority order):
      1. If live process params are provided → predict_quality_drift
      2. If a specific Batch ID (BTH-XXXX) is in the query or state → fetch batch tools
      3. Otherwise (any general/conceptual/statistical query) → aggregate_qms_statistics
         so the LLM always has real factory data as context to answer with.
    """
    user_query = (state.get("user_query") or state.get("query") or "").strip()

    # ── Rule 1: live process parameter prediction ───────────────────────────
    if state.get("ambient_temp_c") is not None:
        return {
            "user_query": user_query,
            "detected_intent": "prediction",
            "analysis_mode": "live_process_prediction",
            "analysis_plan": ["predict_quality_drift"],
            "confidence": 1.0,
            "tool_outputs": {}
        }

    # ── Rule 2: specific Batch ID in query or state ─────────────────────────
    batch_match = re.search(r"\b(BTH-\d+|BATCH-[A-Za-z0-9-]+)\b", user_query, re.IGNORECASE)
    batch_id_from_query = batch_match.group(0).upper() if batch_match else None
    batch_id = batch_id_from_query or state.get("batch_id")

    if batch_id:
        return {
            "user_query": user_query,
            "batch_id": batch_id,
            "detected_intent": "asset",
            "analysis_mode": "batch_analysis",
            "analysis_plan": ["fetch_material_data", "fetch_process_data", "fetch_inspection_data"],
            "confidence": 1.0,
            "tool_outputs": {}
        }

    # ── Rule 3: general / statistical / conceptual — use factory aggregation ─
    q_lower = user_query.lower()
    if any(k in q_lower for k in ["scrap", "defect", "reject", "grade", "shift", "line"]):
        agg_metric = "scrap_rate"
    elif any(k in q_lower for k in ["capacity", "electrolyte", "mah"]):
        agg_metric = "capacity"
    elif any(k in q_lower for k in ["resistance", "temp", "ambient"]):
        agg_metric = "resistance"
    else:
        agg_metric = "all"

    return {
        "user_query": user_query,
        "detected_intent": "statistical",
        "analysis_mode": agg_metric,
        "analysis_plan": ["aggregate_qms_statistics"],
        "confidence": 0.9,
        "tool_outputs": {}
    }


def tool_executor_node(state: QMSState) -> dict:
    """Executes the plan tools dynamically from the registry."""
    tools_to_run = state.get("analysis_plan") or []
    tool_outputs = {}
    batch_id = state.get("batch_id")
    agg_metric = state.get("analysis_mode") or "all"

    for tool_name in tools_to_run:
        tool_callable = _TOOL_REGISTRY.get(tool_name)
        if not tool_callable:
            tool_outputs[tool_name] = {"error": f"Tool '{tool_name}' not available."}
            continue

        try:
            logger.info(f"Executing tool: {tool_name}")
            if tool_name == "predict_quality_drift":
                result = tool_callable.invoke({
                    "ambient_temp_c": state.get("ambient_temp_c", 22.0),
                    "anode_overhang_mm": state.get("anode_overhang_mm", 1.5),
                    "electrolyte_volume_ml": state.get("electrolyte_volume_ml", 12.0),
                    "internal_resistance_mohm": state.get("internal_resistance_mohm", 1.8),
                    "capacity_mah": state.get("capacity_mah", 5000.0),
                    "retention_50cycle_pct": state.get("retention_50cycle_pct", 99.0)
                })
            elif tool_name == "aggregate_qms_statistics":
                result = tool_callable.invoke({"metric": agg_metric})
            else:
                # batch-specific tools — batch_id is guaranteed to exist (Rule 2 above)
                result = tool_callable.invoke({"batch_id": batch_id})
            tool_outputs[tool_name] = result
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}")
            tool_outputs[tool_name] = {"error": str(e)}

    return {"tool_outputs": tool_outputs}


def llm_reasoning_node(state: QMSState) -> dict:
    """Uses LLM to reason over tool outputs and answer the user query."""
    user_query = state.get("user_query") or "No query provided."
    detected_intent = state.get("detected_intent") or "statistical"
    tool_outputs = state.get("tool_outputs") or {}
    batch_id = state.get("batch_id")

    system_prompt = (
        "You are an expert AI Quality Control Agent for EV battery manufacturing.\n"
        "You have access to real factory inspection data, material specs, and process telemetry fetched from the company's QMS dataset.\n"
        "Your role: Interpret the tool outputs below and answer the user's question clearly and completely.\n"
        "Rules:\n"
        "- Use the actual numbers from the tool outputs — do NOT recalculate them.\n"
        "- If data is factory-wide aggregated statistics, frame your answer around factory/fleet insights.\n"
        "- If data is for a specific batch, give a focused per-batch quality analysis.\n"
        "- For conceptual questions (no tool ran), answer using your EV manufacturing domain expertise.\n"
        "- Always give actionable corrective actions and quality improvement recommendations.\n"
        "- Be specific, cite actual numbers, defect types, and production lines from the tool outputs."
    )

    user_prompt = f"User Question: {user_query}\nAnalysis Mode: {detected_intent}\n"
    if batch_id:
        user_prompt += f"Batch Analyzed: {batch_id}\n"
    user_prompt += "\nDataset Tool Outputs:\n"
    for tool_name, output in tool_outputs.items():
        user_prompt += f"\n[{tool_name}]\n{output}\n"
    user_prompt += "\nBased on the data above, provide: summary, explanation, recommendations (corrective actions), and reasoning."

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    try:
        reasoning_result = generate_llm_response(messages, QMSReasoningOutput)
        return {"reasoning_output": reasoning_result.model_dump()}
    except Exception as e:
        logger.error(f"LLM Reasoning failed: {e}")
        return {"reasoning_output": {}}


def response_builder_node(state: QMSState) -> dict:
    """Builds the final response from tool outputs and LLM reasoning."""
    tool_outputs = state.get("tool_outputs") or {}
    reasoning = state.get("reasoning_output") or {}
    query_type = state.get("detected_intent") or "statistical"

    # ── Material data ──────────────────────────────────────────────────────
    if "fetch_material_data" in tool_outputs and "error" not in tool_outputs["fetch_material_data"]:
        material_data = tool_outputs["fetch_material_data"]
    elif "predict_quality_drift" in tool_outputs:
        material_data = {"anode_overhang_mm": state.get("anode_overhang_mm", 0.0), "source": "model_prediction"}
    else:
        material_data = {"source": "conceptual_analysis"}

    # ── Process data ───────────────────────────────────────────────────────
    if "fetch_process_data" in tool_outputs and "error" not in tool_outputs["fetch_process_data"]:
        process_data = tool_outputs["fetch_process_data"]
    elif "predict_quality_drift" in tool_outputs:
        process_data = {"ambient_temperature_celsius": state.get("ambient_temp_c", 0.0), "source": "model_prediction"}
    else:
        process_data = {"source": "conceptual_analysis"}

    # ── Inspection / batch metrics ─────────────────────────────────────────
    batch_metrics = {}
    if "aggregate_qms_statistics" in tool_outputs and "error" not in tool_outputs["aggregate_qms_statistics"]:
        stats = tool_outputs["aggregate_qms_statistics"]
        batch_metrics = {
            "total_inspected": stats.get("total_cells_inspected", 0),
            "defect_rate_pct": stats.get("overall_scrap_defect_rate_pct", 0.0),
            "avg_resistance_mohm": stats.get("average_internal_resistance_mOhm", 0.0),
            "avg_capacity_mah": stats.get("average_cell_capacity_mAh", 0.0),
            "avg_electrolyte_ml": stats.get("average_electrolyte_volume_ml", 0.0)
        }
        inspection_data = tool_outputs["aggregate_qms_statistics"]
    elif "fetch_inspection_data" in tool_outputs and "error" not in tool_outputs["fetch_inspection_data"]:
        insp = tool_outputs["fetch_inspection_data"]
        batch_metrics = {
            "total_inspected": insp.get("total_inspected", 0),
            "defect_rate_pct": insp.get("scrap_rate_pct", 0.0),
            "avg_resistance_mohm": insp.get("avg_resistance_mOhm", 0.0),
            "avg_capacity_mah": insp.get("avg_capacity_mAh", 0.0),
            "avg_electrolyte_ml": insp.get("avg_electrolyte_volume_ml", 0.0)
        }
        inspection_data = insp
    elif "predict_quality_drift" in tool_outputs and "error" not in tool_outputs["predict_quality_drift"]:
        inspection_data = tool_outputs["predict_quality_drift"]
    else:
        inspection_data = {"source": "conceptual_analysis"}

    # ── LLM reasoning text ─────────────────────────────────────────────────
    drift_text = ""
    if reasoning.get("summary") or reasoning.get("explanation"):
        drift_text = f"Summary: {reasoning.get('summary', '')}\n\nExplanation: {reasoning.get('explanation', '')}"
    root_cause_text = reasoning.get("reasoning", "N/A")
    alerts_list = reasoning.get("recommendations", [])

    messages = [f"Analysis completed for query type: {query_type}"]
    if reasoning.get("summary"):
        messages.append(reasoning["summary"])

    return {
        "material_data": material_data,
        "process_data": process_data,
        "inspection_data": inspection_data,
        "quality_drift_analysis": drift_text,
        "process_drift": drift_text,
        "root_cause_analysis": root_cause_text,
        "root_cause": root_cause_text,
        "alerts": alerts_list,
        "corrective_actions": alerts_list,
        "batch_metrics": batch_metrics,
        "messages": messages
    }


# Build the graph
workflow = StateGraph(QMSState)
workflow.add_node("planner", planner_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("llm_reasoning", llm_reasoning_node)
workflow.add_node("response_builder", response_builder_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "tool_executor")
workflow.add_edge("tool_executor", "llm_reasoning")
workflow.add_edge("llm_reasoning", "response_builder")
workflow.add_edge("response_builder", END)

qms_app = workflow.compile()
