import re

def fix_agent(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    # Add generic_description to the QueryPlan model
    if "generic_description:" not in content:
        # Find the end of the query plan class
        if "confidence: float" in content:
            content = content.replace("confidence: float = Field(description=\"Confidence score", "generic_description: str = Field(default=\"\", description=\"Extracted generic vehicle/batch description if no specific ID is provided\")\n    confidence: float = Field(description=\"Confidence score")

    # In the tool executor, skip tools if ID is missing and it's an asset query
    if "ev_apm" in file_path:
        pass # APM handles predict_battery_health differently
        
    with open(file_path, "w") as f:
        f.write(content)

for agent in ["ev_apm_agent", "ev_qms_agent", "ev_supply_chain_agent", "ev_fleet_electrification_agent"]:
    try:
        fix_agent(f"ev_ai_agents/{agent}/agent.py")
    except Exception as e:
        print(f"Failed {agent}: {e}")
