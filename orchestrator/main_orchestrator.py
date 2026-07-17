import os
import sys
from dotenv import load_dotenv

# Ensure the root EV_supply_chain directory is in the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.supervisor import orchestrator_app

load_dotenv()

def run_orchestrator(query: str):
    print(f"\n[Orchestrator] Processing Query: {query}")
    print("-" * 50)
    
    # Initialize the orchestrator state
    initial_state = {
        "user_query": query,
        "agent_responses": {},
        "next_agents": [],
        "final_synthesis": ""
    }
    
    # Run the graph
    for event in orchestrator_app.stream(initial_state):
        for node_name, state_update in event.items():
            print(f"\n[Node Execution] Finished: {node_name}")
            
            if node_name == "router":
                agents = state_update.get("next_agents", [])
                print(f"--> Fan-Out Parallel Routing to: {agents}")
                
            if node_name == "aggregator":
                print("\n=== FINAL AGGREGATED SYNTHESIS ===")
                print(state_update.get("final_synthesis", ""))
                print("====================================\n")

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY not found in environment. Please set it or add to a .env file.")
        sys.exit(1)
        
    sample_query = (
        "Check the APM health for EV-9005. If it needs maintenance, optimize the maintenance schedule. "
        "Also, check the QMS batch data for BATCH-002 for potential supplier defects, and trace the supplier risk. "
        "Finally, what is the carbon impact of this fleet?"
    )
    run_orchestrator(sample_query)
