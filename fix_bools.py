import os
import glob

files = glob.glob("ev_ai_agents/*/agent.py")

for file in files:
    with open(file, "r") as f:
        content = f.read()
    
    content = content.replace("requires_dataset: str", "requires_dataset: bool")
    content = content.replace("requires_llm: str", "requires_llm: bool")
    
    with open(file, "w") as f:
        f.write(content)
