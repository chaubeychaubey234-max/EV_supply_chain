import os
import logging
import re
import json
from typing import List, Optional, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from .state import QMSState
from .tools.material_tools import fetch_material_data, predict_quality_drift
from .tools.process_tools import fetch_process_data
from .tools.inspection_tools import fetch_inspection_data, aggregate_qms_statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ev_qms_agent")

_TOOL_REGISTRY = {
    "fetch_material_data": fetch_material_data,
    "predict_quality_drift": predict_quality_drift,
    "fetch_process_data": fetch_process_data,
    "fetch_inspection_data": fetch_inspection_data,
    "aggregate_qms_statistics": aggregate_qms_statistics
}

class QMSReasoningOutput(BaseModel):
    summary: str = Field(description="Concise answer to the user's question, grounded in factory data where relevant.")
    explanation: str = Field(description="Detailed explanation with manufacturing domain insights. Reference actual numbers from context.")
    recommendations: List[str] = Field(description="Actionable corrective actions for quality engineers or production managers.")
    reasoning: str = Field(description="Step-by-step reasoning for the quality analysis conclusions.")

def generate_llm_response(prompt_messages: list, response_model: Any = None) -> Any:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key.startswith("dummy"):
        raise ValueError("GROQ_API_KEY is missing or invalid.")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    if response_model:
        return llm.with_structured_output(response_model).invoke(prompt_messages)
    return llm.invoke(prompt_messages)

# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT LOADER — always runs, loads factory-wide stats as RAG grounding
# ─────────────────────────────────────────────────────────────────────────────
def _load_factory_context() -> dict:
    """Load full factory aggregate stats to use as LLM context for any query."""
    try:
        return aggregate_qms_statistics.invoke({"metric": "all"})
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# PLANNER — heuristic routing, never depends on LLM
# ─────────────────────────────────────────────────────────────────────────────
def planner_node(state: QMSState) -> dict:
    """
    Route by heuristic priority:
      1. Live process params provided → predict_quality_drift
      2. Specific Batch ID in query/state → fetch that batch's records
      3. Everything else → no extra tools; factory context always loaded in reasoning
    """
    user_query = (state.get("user_query") or state.get("query") or "").strip()

    # Rule 1: live parameter prediction
    if state.get("ambient_temp_c") is not None:
        return {
            "user_query": user_query,
            "detected_intent": "prediction",
            "analysis_mode": "live_process_prediction",
            "analysis_plan": ["predict_quality_drift"],
            "confidence": 1.0,
            "tool_outputs": {}
        }

    # Rule 2: specific Batch ID
    batch_match = re.search(r"\b(BTH-\d+|BATCH-[A-Za-z0-9-]+)\b", user_query, re.IGNORECASE)
    batch_id = (batch_match.group(0).upper() if batch_match else None) or state.get("batch_id")
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

    # Rule 3: general query — factory context covers it
    return {
        "user_query": user_query,
        "detected_intent": "general",
        "analysis_mode": "factory_context_rag",
        "analysis_plan": [],
        "confidence": 1.0,
        "tool_outputs": {}
    }


def tool_executor_node(state: QMSState) -> dict:
    """Executes any batch-specific or prediction tools from the plan."""
    tools_to_run = state.get("analysis_plan") or []
    tool_outputs = {}
    batch_id = state.get("batch_id")

    for tool_name in tools_to_run:
        tool_callable = _TOOL_REGISTRY.get(tool_name)
        if not tool_callable:
            tool_outputs[tool_name] = {"error": f"Tool '{tool_name}' not found."}
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
            else:
                result = tool_callable.invoke({"batch_id": batch_id})
            tool_outputs[tool_name] = result
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}")
            tool_outputs[tool_name] = {"error": str(e)}

    return {"tool_outputs": tool_outputs}


def llm_reasoning_node(state: QMSState) -> dict:
    """
    Always loads factory-wide context as RAG grounding, then answers the user's
    question using that data + any per-batch tool outputs.
    Works for ANY query — conceptual, statistical, operational, or analytical.
    """
    user_query = state.get("user_query") or "No query provided."
    detected_intent = state.get("detected_intent") or "general"
    tool_outputs = state.get("tool_outputs") or {}
    batch_id = state.get("batch_id")

    # Always load factory context as background knowledge
    factory_context = _load_factory_context()

    system_prompt = """You are VoltGrid's EV Battery Manufacturing Quality Intelligence Agent — an expert in EV cell manufacturing, quality management systems, defect root cause analysis, process control, and production line optimization.

You have access to LIVE DATA from your company's manufacturing dataset (loaded below as FACTORY CONTEXT). You must use this data to ground all your answers — whether the question is conceptual, statistical, operational, or analytical.

**Your behavior rules:**
1. **Always use factory context** — When answering any question, reference relevant numbers from the Factory Context to make your answer grounded and specific to this company's manufacturing operations.
2. **Conceptual questions** — Explain the concept AND relate it to actual factory data (e.g., "Anode overhang causes short circuits. In our factory, the most common defect is X, appearing in Y% of scrapped cells").
3. **Statistical questions** — Answer directly from the Factory Context data. Be precise with numbers.
4. **Specific batch questions** — Use the Per-Batch Tool Outputs for focused analysis, compare to factory averages.
5. **Operational/advisory questions** — Use factory data to back your recommendations with real numbers.
6. Never say "I don't have access to data" — you always have the Factory Context.
7. Be specific, cite actual defect types, production lines, scrap rates, and cell metrics from the data."""

    # Build factory context block
    factory_ctx_str = json.dumps(factory_context, indent=2) if factory_context else "Factory context unavailable."

    # Build per-batch outputs block
    asset_outputs_str = ""
    if tool_outputs:
        asset_outputs_str = "\n\nPER-BATCH TOOL OUTPUTS (specific batch data):\n"
        for tool_name, output in tool_outputs.items():
            asset_outputs_str += f"\n[{tool_name}]\n{json.dumps(output, indent=2)}\n"

    user_prompt = f"""USER QUESTION: {user_query}
QUERY MODE: {detected_intent}
{f"SPECIFIC BATCH: {batch_id}" if batch_id else ""}

FACTORY CONTEXT (live aggregate data from company's manufacturing dataset):
{factory_ctx_str}
{asset_outputs_str}

Answer the user's question completely. Ground your answer in the factory data above. Be specific — use real numbers, defect types, and production line details."""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    try:
        result = generate_llm_response(messages, QMSReasoningOutput)
        return {"reasoning_output": result.model_dump()}
    except Exception as e:
        logger.error(f"LLM Reasoning failed: {e}")
        fallback_summary = (
            f"Factory stats: {factory_context.get('total_cells_inspected', 'N/A')} cells inspected "
            f"across {factory_context.get('total_batches_inspected', 'N/A')} batches, "
            f"overall scrap rate {factory_context.get('overall_scrap_defect_rate_pct', 'N/A')}%."
        )
        return {"reasoning_output": {"summary": fallback_summary, "explanation": "", "recommendations": [], "reasoning": ""}}


def response_builder_node(state: QMSState) -> dict:
    """Builds the final response. Always provides real factory data for dashboard charts."""
    tool_outputs = state.get("tool_outputs") or {}
    reasoning = state.get("reasoning_output") or {}
    query_type = state.get("detected_intent") or "general"

    # Material data
    if "fetch_material_data" in tool_outputs and "error" not in tool_outputs["fetch_material_data"]:
        material_data = tool_outputs["fetch_material_data"]
    elif "predict_quality_drift" in tool_outputs:
        material_data = {"source": "model_prediction"}
    else:
        material_data = {"source": "factory_aggregate"}

    # Process data
    if "fetch_process_data" in tool_outputs and "error" not in tool_outputs["fetch_process_data"]:
        process_data = tool_outputs["fetch_process_data"]
    else:
        process_data = {"source": "factory_aggregate"}

    # Inspection data & batch_metrics
    factory = _load_factory_context()
    if "fetch_inspection_data" in tool_outputs and "error" not in tool_outputs["fetch_inspection_data"]:
        insp = tool_outputs["fetch_inspection_data"]
        inspection_data = insp
        batch_metrics = {
            "total_inspected": insp.get("total_inspected", 0),
            "defect_rate_pct": insp.get("scrap_rate_pct", 0.0),
            "avg_resistance_mohm": insp.get("avg_resistance_mOhm", 0.0),
            "avg_capacity_mah": insp.get("avg_capacity_mAh", 0.0),
            "avg_electrolyte_ml": insp.get("avg_electrolyte_volume_ml", 0.0)
        }
    elif "predict_quality_drift" in tool_outputs and "error" not in tool_outputs["predict_quality_drift"]:
        inspection_data = tool_outputs["predict_quality_drift"]
        batch_metrics = {"total_inspected": 1, "defect_rate_pct": 0.0, "avg_resistance_mohm": 0.0, "avg_capacity_mah": 0.0, "avg_electrolyte_ml": 0.0}
    else:
        # General query: use factory aggregate for dashboard charts
        inspection_data = factory
        batch_metrics = {
            "total_inspected": factory.get("total_cells_inspected", 0),
            "defect_rate_pct": factory.get("overall_scrap_defect_rate_pct", 0.0),
            "avg_resistance_mohm": factory.get("average_internal_resistance_mOhm", 0.0),
            "avg_capacity_mah": factory.get("average_cell_capacity_mAh", 0.0),
            "avg_electrolyte_ml": factory.get("average_electrolyte_volume_ml", 0.0)
        }

    # Text outputs
    drift_text = ""
    if reasoning.get("summary") or reasoning.get("explanation"):
        drift_text = f"Summary: {reasoning.get('summary', '')}\n\nExplanation: {reasoning.get('explanation', '')}"
    root_cause_text = reasoning.get("reasoning", "")
    alerts_list = reasoning.get("recommendations", [])

    messages = [f"Analysis completed ({query_type})"]
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


# Build graph
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
