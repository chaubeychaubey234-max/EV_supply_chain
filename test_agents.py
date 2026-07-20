import sys
import os

from ev_ai_agents.ev_apm_agent.agent import apm_app
from ev_ai_agents.ev_qms_agent.agent import qms_app
from ev_ai_agents.carbon_agent.agent import carbon_app
from ev_ai_agents.ev_supply_chain_agent.agent import supply_chain_app
from ev_ai_agents.ev_fleet_electrification_agent.agent import run_agent as run_fleet_agent
from ev_ai_agents.ev_maintenance_operations_agent.agent import run_agent as run_maint_agent
from orchestrator.supervisor import orchestrator_app

print("Testing Supervisor (Ambiguous query routing)")
state = {"user_query": "is my fleet ready for the move to electric", "agent_responses": {}, "next_agents": []}
print(orchestrator_app.invoke(state)["agent_responses"].keys())

print("Testing Supervisor (Multi-agent APM + Maintenance)")
state = {"user_query": "check this vehicle EV-9005's battery health and tell me if it needs maintenance", "agent_responses": {}, "next_agents": []}
print(orchestrator_app.invoke(state)["agent_responses"].keys())

print("Testing Supervisor (Single agent QMS)")
state = {"user_query": "check cell quality for BATCH-001", "agent_responses": {}, "next_agents": []}
print(orchestrator_app.invoke(state)["agent_responses"].keys())

print("Testing Supervisor (Out of scope)")
state = {"user_query": "what's the weather today?", "agent_responses": {}, "next_agents": []}
res = orchestrator_app.invoke(state)
print(res["final_synthesis"])

print("\n\nTesting APM Agent")
print("Conceptual:", apm_app.invoke({"user_query": "why does battery degradation accelerate with fast charging"})["reasoning_output"])
print("Statistical:", apm_app.invoke({"user_query": "what is the average battery health across the fleet"})["reasoning_output"])
print("Asset:", apm_app.invoke({"user_query": "analyze battery health for EV-9005"})["reasoning_output"])
print("Hypothetical/Missing:", apm_app.invoke({"user_query": "analyze a truck that fast charges 5 times a day"})["reasoning_output"])

print("\n\nTesting Supply Chain Agent")
print("Conceptual:", supply_chain_app.invoke({"user_query": "what causes lithium supply risk in general"})["reasoning_output"])
print("Asset:", supply_chain_app.invoke({"user_query": "trace batch BATCH-004"})["reasoning_output"])

print("\n\nTesting Carbon Agent")
print("Conceptual:", carbon_app.invoke({"user_query": "how do EVs reduce carbon emissions?"})["reasoning_output"])
print("Statistical:", carbon_app.invoke({"user_query": "what is our overall net zero progress"})["reasoning_output"])

print("\n\nTesting Maintenance Agent")
print("Maintenance Query:", run_maint_agent("identify high risk vehicles and generate schedule"))

print("\n\nTesting Fleet Agent")
print("Fleet Query:", run_fleet_agent("evaluate my fleet for electrification", "VH_15592"))
