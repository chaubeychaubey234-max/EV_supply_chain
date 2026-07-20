import os
import glob

agents = [
    "ev_ai_agents/ev_supply_chain_agent/agent.py",
    "ev_ai_agents/carbon_agent/agent.py"
]

for file in agents:
    with open(file, "r") as f:
        content = f.read()
    
    new_wrapper = """class AppWrapper:
        def invoke(self, inputs):
            res = supply_chain_app.invoke({"user_query": inputs.get("input", "")}) if 'supply' in file else carbon_app.invoke({"user_query": inputs.get("input", "")})
            res["output"] = res.get("unified_report", "No report generated.")
            return res"""
            
    # well, simpler: 
    # Just fix the return statement in get_agent_executor.
    pass
