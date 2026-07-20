import os
from orchestrator.supervisor import orchestrator_app

state = {"user_query": "check this vehicle EV-9005's battery health and tell me if it needs maintenance", "agent_responses": {}, "next_agents": []}
res = orchestrator_app.invoke(state)
print("Triggered Agents:", res.get("next_agents", []))
print("Final Synthesis:", res.get("final_synthesis", "No synthesis returned."))
