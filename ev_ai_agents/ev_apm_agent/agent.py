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

# Structured Output Schemas for LLM Planner and Reasoner
class APMQueryPlan(BaseModel):
    query_type: str = Field(description="Query type: conceptual, statistical, asset, or hybrid")
    requires_dataset: bool = Field(description="True if dataset access (lookup or aggregation) is required")
    tools: List[str] = Field(description="List of tool keys to run. Choose from: fetch_battery_health, predict_battery_health, fetch_thermal_events, fetch_charging_patterns, aggregate_apm_statistics")
    requires_llm: bool = Field(description="True if LLM reasoning is required to synthesize the final answer")
    analysis_mode: str = Field(description="Brief description of the analysis mode")
    generic_description: str = Field(default="", description="Extracted generic vehicle/batch description if no specific ID is provided")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 representing how confident we are in this plan")
    extracted_ev_id: Optional[str] = Field(description="Extracted EV ID from the query. MUST be null if the query does not contain an explicit EV ID like EV-1234.")
    aggregation_metric: Optional[str] = Field(description="Metric parameter to pass to aggregate_apm_statistics if query is statistical or hybrid. One of: battery_health, temperature, charging_cycles, all")

class APMReasoningOutput(BaseModel):
    summary: str = Field(description="Summary of the battery health status.")
    explanation: str = Field(description="Detailed explanation of the analysis or concept.")
    recommendations: List[str] = Field(description="List of actionable recommendations.")
    reasoning: str = Field(description="Reasoning process for the conclusions.")
    maintenance_triggers: List[str] = Field(description="List of predictive maintenance triggers.")

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

def planner_node(state: APMState) -> dict:
    """Classifies the query and generates an execution plan."""
    user_query = state.get("user_query") or state.get("query")
    
    # If user_query is absent or empty, default to standard asset lookup behavior
    if not user_query or not user_query.strip():
        ev_id = state.get("ev_id") 
        if state.get("avg_temperature_c") is not None:
            tools = ["predict_battery_health"]
        else:
            tools = ["fetch_battery_health", "fetch_thermal_events", "fetch_charging_patterns"]
            
        return {
            "user_query": user_query or "",
            "detected_intent": "asset",
            "analysis_mode": "asset_analysis",
            "analysis_plan": tools,
            "confidence": 1.0,
            "tool_outputs": {}
        }
        
    system_prompt = (
        "You are the Query Planner for an EV Battery Asset Performance Management (APM) system.\n"
        "Your task is to analyze the user's query and decide on an execution plan.\n"
        "Available tool capabilities to schedule in the plan:\n"
        "- 'fetch_battery_health': Retrieve battery health history for a specific ev_id.\n"
        "- 'predict_battery_health': Use ML model to predict battery health from live telemetry parameters. Only schedule this if telemetry parameters (like temperature, cycles, duration) are provided in the query or state.\n"
        "- 'fetch_thermal_events': Retrieve thermal history and anomaly data for a specific ev_id.\n"
        "- 'fetch_charging_patterns': Retrieve charging cycles and risks for a specific ev_id.\n"
        "- 'aggregate_apm_statistics': Calculate aggregated fleet metrics. Run this if the user asks statistical questions.\n"
        "\n"
        "Classify the query into one of:\n"
        "1. 'conceptual': Concept explanations (e.g. why battery health decreases, thermal runaway details). Needs NO dataset/tools.\n"
        "2. 'statistical': Aggregate queries (e.g. average battery health, average SOH, percentage of critical batteries). Run 'aggregate_apm_statistics'.\n"
        "3. 'asset': Requests to analyze a specific EV (e.g. 'Analyze EV-9005', 'Check EV-9001'). Run 'fetch_battery_health', 'fetch_thermal_events', 'fetch_charging_patterns'.\n"
        "4. 'hybrid': Mix of specific asset analysis and statistical/conceptual comparison (e.g. 'Analyze EV-9002 and compare to fleet average'). Run both specific asset tools and aggregation tools.\n"
        "\n"
        "If it is a statistical or hybrid query, choose the appropriate aggregation_metric:\n"
        "- 'battery_health': for SOH, degradation rate, critical batteries questions.\n"
        "- 'temperature': for temperature, thermal, cooling questions.\n"
        "- 'charging_cycles': for charging cycles, fast charging, duration questions.\n"
        "- 'all': if multiple or overall metrics are requested.\n"
        "\n"
        "Ensure confidence is a float between 0.0 and 1.0. If the query is highly structured and clearly fits one of the intents, confidence should be >= 0.9."
    )
    
    # Fallback/heuristic EV ID extraction
    ev_match = re.search(r"\b(EV-\d+)\b", user_query, re.IGNORECASE)
    extracted_ev_id = ev_match.group(0).upper() if ev_match else None
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate a query plan for the user query: '{user_query}'")
    ]
    
    plan = generate_llm_response(messages, APMQueryPlan)
    
    final_ev_id = plan.extracted_ev_id or extracted_ev_id
    if final_ev_id and final_ev_id not in state.get("user_query", ""):
        # LLM hallucinated an ID not in the query
        final_ev_id = None
        
    if plan.query_type in ("asset", "hybrid") and not final_ev_id:
        final_ev_id = state.get("ev_id") 
        
    state_updates = {
        "detected_intent": plan.query_type,
        "analysis_mode": plan.analysis_mode,
        "analysis_plan": plan.tools,
        "confidence": plan.confidence,
        "tool_outputs": {}
    }
    
    if final_ev_id:
        state_updates["ev_id"] = final_ev_id
        
    if plan.aggregation_metric:
        state_updates["analysis_mode"] = f"{plan.aggregation_metric}"
        
    return state_updates

def tool_executor_node(state: APMState) -> dict:
    """Executes the plan tools dynamically from the registry and stores results namespaced."""
    tools_to_run = state.get("analysis_plan") or []
    tool_outputs = {}
    ev_id = state.get("ev_id") 
    
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
            if tool_name == "predict_battery_health":
                inputs = {
                    "avg_temperature_c": state.get("avg_temperature_c", 25.0),
                    "fast_charge_ratio_pct": state.get("fast_charge_ratio_pct", 30.0),
                    "deep_discharge_cycles": state.get("deep_discharge_cycles", 5),
                    "avg_charge_duration_hours": state.get("avg_charge_duration_hours", 4.0),
                    "max_temperature_c": state.get("max_temperature_c", 45.0)
                }
                result = tool_callable.invoke(inputs)
            elif tool_name == "aggregate_apm_statistics":
                result = tool_callable.invoke({"metric": agg_metric})
            else:
                if not ev_id:
                    result = {"message": f"Tool {tool_name} skipped. No specific ev_id provided. Answer conceptually based on user query description."}
                else:
                    result = tool_callable.invoke({"ev_id": ev_id})
                
            tool_outputs[tool_name] = result
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}")
            tool_outputs[tool_name] = {"error": f"Tool execution failed: {str(e)}"}
            
    return {"tool_outputs": tool_outputs}

def llm_reasoning_node(state: APMState) -> dict:
    """Uses LLM to perform domain-specific reasoning over the tools' outputs and queries."""
    user_query = state.get("user_query") or state.get("query") or "No query provided."
    detected_intent = state.get("detected_intent") or "asset"
    tool_outputs = state.get("tool_outputs") or {}
    ev_id = state.get("ev_id") 
    
    system_prompt = (
        "You are an expert AI for Industrial EV Supply Chain & Asset Intelligence. Your role is to act as an EV battery domain expert.\n"
        "Your task is to interpret tool outputs and user queries to provide comprehensive battery health analysis, thermal safety warnings, charging pattern risk analysis, and actionable predictive maintenance recommendations.\n"
        "Strict Rule: You must NEVER calculate statistics. All calculations are done by the tools. You should only explain what they mean in the context of EV operation and battery degradation theory.\n"
        "Strict Rule: If the query is conceptual, answer using your expert domain knowledge. Do not reference datasets or statistics if no tool was run.\n"
        "Strict Rule: Do NOT invent or substitute vehicle IDs. If the tool outputs 'PREDICTED_ASSET', refer to it as a 'hypothetical predicted vehicle'. You must not use any specific vehicle ID like 'EV-XXXX'."
    )
    
    user_prompt = (
        f"User Query: {user_query}\n"
        f"Intent/Mode: {detected_intent}\n"
        f"EV ID Analyzed: {ev_id}\n\n"
        f"Executed Tool Outputs (Namespaced):\n"
    )
    for tool_name, output in tool_outputs.items():
        user_prompt += f"--- {tool_name} ---\n{output}\n\n"
        
    user_prompt += "Analyze these outputs and provide summary, explanation, recommendations, reasoning, and maintenance_triggers."
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        reasoning_result = generate_llm_response(messages, APMReasoningOutput)
        return {"reasoning_output": reasoning_result.model_dump()}
    except Exception as e:
        logger.error(f"LLM Reasoning failed: {e}")
        return {"reasoning_output": {}}

def response_builder_node(state: APMState) -> dict:
    """Builds the final backward-compatible response based on tools outputs and LLM reasoning."""
    tool_outputs = state.get("tool_outputs") or {}
    reasoning = state.get("reasoning_output") or {}
    query_type = state.get("detected_intent") or "asset"
    
    # 1. Resolve battery_analysis
    battery_analysis = {}
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
        battery_analysis = {
            "status": "Conceptual Analysis Only",
            "source": "llm_reasoning",
            "state_of_health_percentage": 0.0,
            "degradation_rate_per_month": 0.0
        }
        
    # 2. Resolve safety_analysis
    safety_analysis = {}
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
        max_temp = state.get("max_temperature_c", 0.0)
        safety_analysis = {
            "max_recorded_temperature_celsius": max_temp,
            "thermal_runaway_warnings": 1 if max_temp > 50.0 else 0,
            "cooling_system_status": "OK" if max_temp < 55.0 else "Degraded",
            "source": "conceptual_analysis"
        }
        
    # 3. Resolve telemetry_data
    telemetry_data = {}
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
        fc_ratio = state.get("fast_charge_ratio_pct", 0.0)
        telemetry_data = {
            "fast_charging_ratio_percentage": fc_ratio,
            "source": "conceptual_analysis"
        }

    # 4. Resolve recommendations & triggers
    recs = []
    if query_type in ("conceptual", "statistical", "hybrid"):
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
