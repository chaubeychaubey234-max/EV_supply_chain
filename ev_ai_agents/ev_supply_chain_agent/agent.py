import os
import re
import logging
from typing import List, Optional, Any, Dict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from .state import SupplyChainState
from ev_ai_agents.ev_supply_chain_agent.tools.supplier_tools import (
    get_supplier_profile,
    get_supplier_tier,
    get_supplier_geography
)
from ev_ai_agents.ev_supply_chain_agent.tools.traceability_tools import (
    trace_material_batch,
    map_cell_to_pack,
    map_pack_to_vehicle,
    verify_traceability_completeness
)
from ev_ai_agents.ev_supply_chain_agent.tools.risk_tools import (
    calculate_supplier_risk_score,
    detect_geopolitical_risk,
    detect_supplier_concentration,
    detect_quality_deviation
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("supply_chain_agent")

_TOOL_REGISTRY = {
    "get_supplier_profile": get_supplier_profile,
    "get_supplier_tier": get_supplier_tier,
    "get_supplier_geography": get_supplier_geography,
    "trace_material_batch": trace_material_batch,
    "map_cell_to_pack": map_cell_to_pack,
    "map_pack_to_vehicle": map_pack_to_vehicle,
    "verify_traceability_completeness": verify_traceability_completeness,
    "calculate_supplier_risk_score": calculate_supplier_risk_score,
    "detect_geopolitical_risk": detect_geopolitical_risk,
    "detect_supplier_concentration": detect_supplier_concentration,
    "detect_quality_deviation": detect_quality_deviation
}

class SupplyChainQueryPlan(BaseModel):
    query_type: str = Field(description="Query type: conceptual, statistical, asset, or hybrid")
    requires_dataset: bool = Field(description="True if dataset access is required")
    tools: List[str] = Field(description="List of tool keys to run.")
    requires_llm: bool = Field(description="True if LLM reasoning is required")
    generic_description: str = Field(default="", description="Extracted generic vehicle/batch description if no specific ID is provided")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")

class SupplyChainReasoningOutput(BaseModel):
    unified_report: str = Field(description="Unified report summarizing supplier profile, traceability, and risk.")

def generate_llm_response(prompt_messages: list, response_model: Any = None) -> Any:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key.startswith("dummy"):
        raise ValueError("GROQ_API_KEY is missing or invalid. LLM execution cannot proceed.")
            
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    if response_model:
        structured_llm = llm.with_structured_output(response_model)
        return structured_llm.invoke(prompt_messages)
    return llm.invoke(prompt_messages)

def planner_node(state: SupplyChainState) -> dict:
    user_query = state.get("user_query", "")
    
    system_prompt = (
        "You are the Query Planner for an EV Supply Chain Intelligence system.\n"
        "Classify the query into one of: 'conceptual', 'statistical', 'asset', or 'hybrid'.\n"
        "Select the appropriate tools to answer the user's query from the registry.\n"
        "If it is an asset query, run 'get_supplier_profile', 'trace_material_batch', 'detect_geopolitical_risk'.\n"
        "For statistical or concentration risk, run 'detect_supplier_concentration'.\n"
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate a query plan for the user query: '{user_query}'")
    ]
    
    plan = generate_llm_response(messages, SupplyChainQueryPlan)
    
    sup_match = re.search(r"\b(SUP-\d+)\b", user_query, re.IGNORECASE)
    supplier_id = sup_match.group(0).upper() if sup_match else None
    
    bat_match = re.search(r"\b(BATCH-\d+)\b", user_query, re.IGNORECASE)
    batch_id = bat_match.group(0).upper() if bat_match else None
    
    return {
        "detected_intent": plan.query_type,
        "analysis_mode": f"{plan.query_type}_analysis",
        "analysis_plan": plan.tools,
        "confidence": plan.confidence,
        "supplier_id": supplier_id,
        "batch_id": batch_id,
        "tool_outputs": {}
    }

def tool_executor_node(state: SupplyChainState) -> dict:
    tools_to_run = state.get("analysis_plan", [])
    tool_outputs = {}
    supplier_id = state.get("supplier_id")
    batch_id = state.get("batch_id")
    
    for tool_name in tools_to_run:
        tool_callable = _TOOL_REGISTRY.get(tool_name)
        if not tool_callable:
            tool_outputs[tool_name] = {"error": f"Tool '{tool_name}' not available."}
            continue
            
        try:
            logger.info(f"Executing tool: {tool_name}")
            if "supplier" in tool_name or tool_name == "detect_geopolitical_risk":
                # For detect_geopolitical_risk, mock country if not extracted
                arg = "China" if tool_name == "detect_geopolitical_risk" else supplier_id
                if tool_name == "detect_supplier_concentration":
                    arg = [supplier_id]
                elif tool_name == "calculate_supplier_risk_score":
                    arg = {"supplier_id": supplier_id, "quality_score": 90, "delivery_score": 95, "esg_score": 85}
                
                result = tool_callable.invoke(arg)
            elif "batch" in tool_name or tool_name == "verify_traceability_completeness":
                arg = {"batch_id": batch_id} if tool_name == "verify_traceability_completeness" else batch_id
                result = tool_callable.invoke(arg)
            else:
                result = tool_callable.invoke(batch_id)
                
            tool_outputs[tool_name] = result
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}")
            tool_outputs[tool_name] = {"error": f"Tool execution failed: {str(e)}"}
            
    return {"tool_outputs": tool_outputs}

def llm_reasoning_node(state: SupplyChainState) -> dict:
    user_query = state.get("user_query", "No query provided.")
    tool_outputs = state.get("tool_outputs", {})
    
    system_prompt = (
        "You are an expert EV Supply Chain Agent.\n"
        "Your task is to interpret tool outputs and user queries to provide comprehensive supply chain analysis.\n"
        "Create a unified report."
    )
    
    user_prompt = f"User Query: {user_query}\n\nTool Outputs:\n"
    for tool_name, output in tool_outputs.items():
        user_prompt += f"--- {tool_name} ---\n{output}\n\n"
        
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    reasoning_result = generate_llm_response(messages, SupplyChainReasoningOutput)
    return {"reasoning_output": reasoning_result.model_dump()}

def response_builder_node(state: SupplyChainState) -> dict:
    reasoning = state.get("reasoning_output", {})
    return {
        "unified_report": reasoning.get("unified_report", "No report generated.")
    }

workflow = StateGraph(SupplyChainState)
workflow.add_node("planner", planner_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("llm_reasoning", llm_reasoning_node)
workflow.add_node("response_builder", response_builder_node)
workflow.set_entry_point("planner")
workflow.add_edge("planner", "tool_executor")
workflow.add_edge("tool_executor", "llm_reasoning")
workflow.add_edge("llm_reasoning", "response_builder")
workflow.add_edge("response_builder", END)

supply_chain_app = workflow.compile()

def get_agent_executor():
    # Shim to support legacy entrypoint
    class AppWrapper:
        def invoke(self, inputs):
            res = supply_chain_app.invoke({"user_query": inputs.get("input", "")}); res["output"] = res.get("unified_report", ""); return res # ({"user_query": inputs.get("input", "")})
    return AppWrapper()
