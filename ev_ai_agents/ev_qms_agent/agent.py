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

# Structured Output Schemas for LLM Planner and Reasoner
class QMSQueryPlan(BaseModel):
    query_type: str = Field(description="Query type: conceptual, statistical, asset, or hybrid")
    requires_dataset: bool = Field(description="True if dataset access (lookup or aggregation) is required")
    tools: List[str] = Field(description="List of tool keys to run. Choose from: fetch_material_data, predict_quality_drift, fetch_process_data, fetch_inspection_data, aggregate_qms_statistics")
    requires_llm: bool = Field(description="True if LLM reasoning is required to synthesize the final answer")
    analysis_mode: str = Field(description="Brief description of the analysis mode")
    generic_description: str = Field(default="", description="Extracted generic vehicle/batch description if no specific ID is provided")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 representing how confident we are in this plan")
    extracted_batch_id: Optional[str] = Field(description="Extracted Batch ID (e.g. BATCH-002) from the query if mentioned")
    aggregation_metric: Optional[str] = Field(description="Metric parameter to pass to aggregate_qms_statistics if query is statistical or hybrid. One of: scrap_rate, capacity, resistance, all")

class QMSReasoningOutput(BaseModel):
    summary: str = Field(description="Summary of the manufacturing quality status.")
    explanation: str = Field(description="Detailed explanation of the quality drift/root cause.")
    recommendations: List[str] = Field(description="List of corrective actions or alerts for operators.")
    reasoning: str = Field(description="Reasoning process for the root cause and drift analysis.")

# Isolated LLM execution helper with mock fallback
def generate_llm_response(prompt_messages: list, response_model: Any = None) -> Any:
    """Helper to run the ChatGroq model with optional structured output.
    If GROQ_API_KEY is not present or is a dummy key, returns a fallback mock response.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key.startswith("dummy"):
        raise ValueError("GROQ_API_KEY is missing or invalid. LLM execution cannot proceed.")
            
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    if response_model:
        structured_llm = llm.with_structured_output(response_model)
        return structured_llm.invoke(prompt_messages)
    return llm.invoke(prompt_messages)

# Graph Nodes

def planner_node(state: QMSState) -> dict:
    """Classifies the query and generates an execution plan."""
    user_query = state.get("user_query") or state.get("query")
    
    # If user_query is absent or empty, default to standard asset lookup behavior
    if not user_query or not user_query.strip():
        batch_id = state.get("batch_id") or "BATCH-001"
        if state.get("ambient_temp_c") is not None:
            tools = ["predict_quality_drift"]
        else:
            tools = ["fetch_material_data", "fetch_process_data", "fetch_inspection_data"]
            
        return {
            "user_query": user_query or "",
            "detected_intent": "asset",
            "analysis_mode": "asset_analysis",
            "analysis_plan": tools,
            "confidence": 1.0,
            "tool_outputs": {}
        }
        
    system_prompt = (
        "You are the Query Planner for an EV Manufacturing Quality Management System (QMS).\n"
        "Your task is to analyze the user's query and decide on an execution plan.\n"
        "Available tool capabilities to schedule in the plan:\n"
        "- 'fetch_material_data': Retrieve material specifications and suppliers for a batch_id.\n"
        "- 'predict_quality_drift': Predict cell grade (Pass/Scrap) from live process parameters. Only schedule this if parameters (ambient_temp_c, anode_overhang_mm, electrolyte_volume_ml, etc.) are provided in the query or state.\n"
        "- 'fetch_process_data': Retrieve machinery process telemetry for a batch_id.\n"
        "- 'fetch_inspection_data': Retrieve inline inspection results and defects for a batch_id.\n"
        "- 'aggregate_qms_statistics': Calculate aggregated quality metrics and scrap distributions. Run this if the user asks statistical questions.\n"
        "\n"
        "Classify the query into one of:\n"
        "1. 'conceptual': Concept explanations (e.g. what causes process drift, anode overhang defects explanation). Needs NO dataset/tools.\n"
        "2. 'statistical': Aggregate queries (e.g. average capacity, scrap rate by line, distribution of defect categories). Run 'aggregate_qms_statistics'.\n"
        "3. 'asset': Requests to analyze a specific batch (e.g. 'Inspect BATCH-002', 'Analyze Batch-101'). Run 'fetch_material_data', 'fetch_process_data', 'fetch_inspection_data'.\n"
        "4. 'hybrid': Mix of specific batch inspection and statistical/conceptual comparison (e.g. 'Analyze BATCH-002 and compare it to overall production metrics'). Run both specific batch tools and aggregation tools.\n"
        "\n"
        "If it is a statistical or hybrid query, choose the appropriate aggregation_metric:\n"
        "- 'scrap_rate': for scrap rate, defect types, shift performance questions.\n"
        "- 'capacity': for cell capacity, electrolyte volume questions.\n"
        "- 'resistance': for internal resistance, ambient temperature questions.\n"
        "- 'all': if multiple or overall quality metrics are requested.\n"
        "\n"
        "Ensure confidence is a float between 0.0 and 1.0. If the query is highly structured and clearly fits one of the intents, confidence should be >= 0.9."
    )
    
    # Fallback/heuristic Batch ID extraction
    batch_match = re.search(r"\b(BATCH-[A-Za-z0-9-]+|BATCH-\d+)\b", user_query, re.IGNORECASE)
    extracted_batch_id = batch_match.group(0).upper() if batch_match else None
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate a query plan for the user query: '{user_query}'")
    ]
    
    plan = generate_llm_response(messages, QMSQueryPlan)
    
    final_batch_id = plan.extracted_batch_id or extracted_batch_id
    if final_batch_id and final_batch_id not in state.get("user_query", ""):
        final_batch_id = None
    if plan.query_type in ("asset", "hybrid") and not final_batch_id:
        final_batch_id = state.get("batch_id") or "BATCH-001"
        
    state_updates = {
        "detected_intent": plan.query_type,
        "analysis_mode": plan.analysis_mode,
        "analysis_plan": plan.tools,
        "confidence": plan.confidence,
        "tool_outputs": {}
    }
    
    if final_batch_id:
        state_updates["batch_id"] = final_batch_id
        
    if plan.aggregation_metric:
        state_updates["analysis_mode"] = f"{plan.aggregation_metric}"
        
    return state_updates

def tool_executor_node(state: QMSState) -> dict:
    """Executes the plan tools dynamically from the registry and stores results namespaced."""
    tools_to_run = state.get("analysis_plan") or []
    tool_outputs = {}
    batch_id = state.get("batch_id") or "BATCH-001"
    
    agg_metric = state.get("analysis_mode") or "all"
    if " | " in agg_metric:
        agg_metric = agg_metric.split(" | ")[-1]
    if "metric: " in agg_metric:
        agg_metric = agg_metric.split("metric: ")[-1]
        
    for tool_name in tools_to_run:
        tool_callable = _TOOL_REGISTRY.get(tool_name)
        if not tool_callable:
            logger.error(f"Tool '{tool_name}' not found in registry.")
            tool_outputs[tool_name] = {"error": f"Tool '{tool_name}' not available in registry."}
            continue
            
        try:
            logger.info(f"Executing tool: {tool_name}")
            if tool_name == "predict_quality_drift":
                inputs = {
                    "ambient_temp_c": state.get("ambient_temp_c", 22.0),
                    "anode_overhang_mm": state.get("anode_overhang_mm", 1.5),
                    "electrolyte_volume_ml": state.get("electrolyte_volume_ml", 12.0),
                    "internal_resistance_mohm": state.get("internal_resistance_mohm", 1.8),
                    "capacity_mah": state.get("capacity_mah", 5000.0),
                    "retention_50cycle_pct": state.get("retention_50cycle_pct", 99.0)
                }
                result = tool_callable.invoke(inputs)
            elif tool_name == "aggregate_qms_statistics":
                result = tool_callable.invoke({"metric": agg_metric})
            else:
                result = tool_callable.invoke({"batch_id": batch_id})
                
            tool_outputs[tool_name] = result
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}")
            tool_outputs[tool_name] = {"error": f"Tool execution failed: {str(e)}"}
            
    return {"tool_outputs": tool_outputs}

def llm_reasoning_node(state: QMSState) -> dict:
    """Uses LLM to perform domain-specific reasoning over the tools' outputs and queries."""
    user_query = state.get("user_query") or state.get("query") or "No query provided."
    detected_intent = state.get("detected_intent") or "asset"
    tool_outputs = state.get("tool_outputs") or {}
    batch_id = state.get("batch_id") or "BATCH-001"
    
    system_prompt = (
        "You are an AI Quality Agent and EV manufacturing quality domain expert.\n"
        "Your task is to correlate parameters, incoming material data, and inspection results to detect quality drift, explain root causes, and suggest corrective actions.\n"
        "Strict Rule: You must NEVER calculate statistics. All calculations are done by the tools. You should only explain what they mean in the context of cell quality control and degradation analysis.\n"
        "Strict Rule: If the query is conceptual, answer using your expert domain knowledge. Do not reference datasets or statistics if no tool was run."
    )
    
    user_prompt = (
        f"User Query: {user_query}\n"
        f"Intent/Mode: {detected_intent}\n"
        f"Batch ID Analyzed: {batch_id}\n\n"
        f"Executed Tool Outputs (Namespaced):\n"
    )
    for tool_name, output in tool_outputs.items():
        user_prompt += f"--- {tool_name} ---\n{output}\n\n"
        
    user_prompt += "Analyze these outputs and provide summary, explanation, recommendations (operator alerts), and reasoning."
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    reasoning_result = generate_llm_response(messages, QMSReasoningOutput)
    
    return {"reasoning_output": reasoning_result.model_dump()}

def response_builder_node(state: QMSState) -> dict:
    """Builds the final backward-compatible response based on tool outputs and LLM reasoning."""
    tool_outputs = state.get("tool_outputs") or {}
    reasoning = state.get("reasoning_output") or {}
    query_type = state.get("detected_intent") or "asset"
    
    # 1. Resolve material_data
    material_data = {}
    if "fetch_material_data" in tool_outputs and "error" not in tool_outputs["fetch_material_data"]:
        material_data = tool_outputs["fetch_material_data"]
    elif "predict_quality_drift" in tool_outputs:
        material_data = {
            "anode_overhang_mm": state.get("anode_overhang_mm", 0.0),
            "electrolyte_volume_ml": state.get("electrolyte_volume_ml", 0.0),
            "source": "model_prediction"
        }
    else:
        material_data = {"source": "conceptual_analysis"}
        
    # 2. Resolve process_data
    process_data = {}
    if "fetch_process_data" in tool_outputs and "error" not in tool_outputs["fetch_process_data"]:
        process_data = tool_outputs["fetch_process_data"]
    elif "predict_quality_drift" in tool_outputs:
        process_data = {
            "ambient_temperature_celsius": state.get("ambient_temp_c", 0.0),
            "source": "model_prediction"
        }
    else:
        process_data = {"source": "conceptual_analysis"}
        
    # 3. Resolve inspection_data
    inspection_data = {}
    if "predict_quality_drift" in tool_outputs and "error" not in tool_outputs["predict_quality_drift"]:
        inspection_data = tool_outputs["predict_quality_drift"]
    elif "fetch_inspection_data" in tool_outputs and "error" not in tool_outputs["fetch_inspection_data"]:
        inspection_data = tool_outputs["fetch_inspection_data"]
    else:
        inspection_data = {"source": "conceptual_analysis"}
        
    # 4. Resolve quality_drift_analysis / process_drift
    drift_text = ""
    if reasoning.get("summary") or reasoning.get("explanation"):
        drift_text = f"Summary: {reasoning.get('summary', '')}\n\nExplanation: {reasoning.get('explanation', '')}"
        
    # 5. Resolve root_cause_analysis / root_cause
    root_cause_text = reasoning.get("reasoning", "N/A")
    
    # 6. Resolve alerts / corrective_actions
    alerts_list = reasoning.get("recommendations", [])
    
    # 7. Resolve batch_metrics (crucial for dashboard)
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
    elif "fetch_inspection_data" in tool_outputs and "error" not in tool_outputs["fetch_inspection_data"]:
        insp = tool_outputs["fetch_inspection_data"]
        batch_metrics = {
            "total_inspected": insp.get("total_inspected", 1),
            "defect_rate_pct": insp.get("scrap_rate_pct", 0.0),
            "avg_resistance_mohm": insp.get("avg_resistance_mOhm", 0.0),
            "avg_capacity_mah": insp.get("avg_capacity_mAh", 0.0),
            "avg_electrolyte_ml": insp.get("avg_electrolyte_volume_ml", 0.0)
        }
    else:
        batch_metrics = {
            "total_inspected": 0,
            "defect_rate_pct": 0.0,
            "avg_resistance_mohm": 0.0,
            "avg_capacity_mah": 0.0,
            "avg_electrolyte_ml": 0.0
        }
        
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
