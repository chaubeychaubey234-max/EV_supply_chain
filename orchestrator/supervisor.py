import os
import json
import re
from typing import Literal, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from pydantic import BaseModel, Field

from orchestrator.state import OrchestratorState

# Import sub-agents
from ev_ai_agents.ev_apm_agent.agent import apm_app
from ev_ai_agents.ev_qms_agent.agent import qms_app
from ev_ai_agents.ev_supply_chain_agent.agent import supply_chain_app
from ev_ai_agents.ev_fleet_electrification_agent.agent import run_agent as run_fleet_agent
from ev_ai_agents.ev_maintenance_operations_agent.agent import run_agent as run_maintenance_agent
from ev_ai_agents.carbon_agent.agent import get_agent_executor as get_carbon_agent

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class RouterOutput(BaseModel):
    next_agents: List[Literal["apm", "qms", "supply_chain", "fleet", "maintenance", "carbon"]] = Field(
        description="List of agents to run in parallel to answer the query."
    )
    reasoning: str = Field(description="Why you chose these agents")

class SynthesisOutput(BaseModel):
    final_answer: str = Field(description="The final comprehensive answer synthesizing all agent data.")

def router_node(state: OrchestratorState) -> OrchestratorState:
    """Decides which agents to invoke in PARALLEL."""
    system_prompt = f"""You are the central EV Supply Chain Orchestrator.
Your goal is to parse the user's complex query and determine ALL domains that need to be evaluated.
Available Agents:
- 'apm': Battery health, RUL, thermal events, charging patterns.
- 'qms': Manufacturing quality, defects, batch inspection.
- 'supply_chain': Supplier tracking, traceability, ESG compliance, risk.
- 'fleet': Fleet electrification readiness, route ROI, procurement matching.
- 'maintenance': Optimizing maintenance schedules and charging availability.
- 'carbon': Tracking scope emissions, net zero progress, route carbon impact.

Select ALL agents that are required to answer the query. They will be run simultaneously in parallel.
"""
    structured_llm = llm.with_structured_output(RouterOutput)
    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["user_query"])
    ])
    
    return {"next_agents": result.next_agents}

def aggregate_responses(state: OrchestratorState) -> OrchestratorState:
    """Fan-in node that aggregates parallel responses into a final answer."""
    system_prompt = f"""You are the Aggregation Layer. 
Synthesize the outputs from all the specialized agents into one unified, actionable, and formatted report for the user.
Do not mention the internal agents by name, just present the data holistically.

Agent Data Provided:
{json.dumps(state.get('agent_responses', {}), indent=2, default=str)}
"""
    structured_llm = llm.with_structured_output(SynthesisOutput)
    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["user_query"])
    ])
    
    return {"final_synthesis": result.final_answer}

# --- Adapter Nodes for Sub-Agents ---
# Note: They return a dictionary meant to be merged via `operator.ior` on `agent_responses`

def call_apm_agent(state: OrchestratorState) -> dict:
    ev_id_match = re.search(r"EV-\d+", state["user_query"], re.IGNORECASE)
    ev_id = ev_id_match.group(0).upper() if ev_id_match else "EV-9000"
    result = apm_app.invoke({"ev_id": ev_id})
    return {"agent_responses": {"apm": {"recommendations": result.get("recommendations", "")}}}

def call_qms_agent(state: OrchestratorState) -> dict:
    batch_id_match = re.search(r"BATCH-\d+", state["user_query"], re.IGNORECASE)
    batch_id = batch_id_match.group(0).upper() if batch_id_match else "BATCH-001"
    result = qms_app.invoke({"batch_id": batch_id})
    return {"agent_responses": {"qms": {"quality_drift_analysis": result.get("quality_drift_analysis", "")}}}

def call_supply_chain_agent(state: OrchestratorState) -> dict:
    result = supply_chain_app.invoke({"user_input": state["user_query"]})
    return {"agent_responses": {"supply_chain": result.get("final_response", "")}}

def call_fleet_agent(state: OrchestratorState) -> dict:
    result = run_fleet_agent(state["user_query"])
    return {"agent_responses": {"fleet": result}}

def call_maintenance_agent(state: OrchestratorState) -> dict:
    result = run_maintenance_agent(state["user_query"])
    return {"agent_responses": {"maintenance": result}}

def call_carbon_agent(state: OrchestratorState) -> dict:
    carbon_agent = get_carbon_agent()
    result = carbon_agent.invoke({"input": state["user_query"]})
    return {"agent_responses": {"carbon": result.get("output", result)}}

# --- Build the Map-Reduce Supervisor Graph ---

def route_to_parallel_agents(state: OrchestratorState):
    """The conditional edge that returns multiple Send() objects for parallel execution."""
    sends = []
    for agent in state.get("next_agents", []):
        if agent in ["apm", "qms", "supply_chain", "fleet", "maintenance", "carbon"]:
            sends.append(Send(agent, state))
    
    # If no agents selected or an error occurred, go straight to aggregator
    if not sends:
        return "aggregator"
        
    return sends

workflow = StateGraph(OrchestratorState)

workflow.add_node("router", router_node)
workflow.add_node("apm", call_apm_agent)
workflow.add_node("qms", call_qms_agent)
workflow.add_node("supply_chain", call_supply_chain_agent)
workflow.add_node("fleet", call_fleet_agent)
workflow.add_node("maintenance", call_maintenance_agent)
workflow.add_node("carbon", call_carbon_agent)
workflow.add_node("aggregator", aggregate_responses)

# Map (Fan-out)
workflow.add_conditional_edges(
    "router",
    route_to_parallel_agents,
    ["apm", "qms", "supply_chain", "fleet", "maintenance", "carbon", "aggregator"]
)

# Reduce (Fan-in): All agents route to the aggregator
for agent in ["apm", "qms", "supply_chain", "fleet", "maintenance", "carbon"]:
    workflow.add_edge(agent, "aggregator")

# Finish
workflow.add_edge("aggregator", END)

workflow.set_entry_point("router")
orchestrator_app = workflow.compile()
