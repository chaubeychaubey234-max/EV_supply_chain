import requests
import json
import urllib.parse
import time

BASE_URL = "http://localhost:8080/api/agents"

queries = {
    "ev_apm": {
        "url": f"{BASE_URL}/ev_apm",
        "params": {
            "user_query": "Analyze the expected degradation rate and thermal runaway risks for an electric transit van operating in a high-temperature desert climate (45°C) that relies almost entirely on daily fast charging."
        }
    },
    "ev_qms": {
        "url": f"{BASE_URL}/ev_qms",
        "params": {
            "user_query": "If we experience a sudden spike in internal resistance during end-of-line testing for our LFP cells, and we suspect the electrolyte filling volume is too low, what process drift could be happening at the mixing or calendering stage?"
        }
    },
    "supply_chain": {
        "url": f"{BASE_URL}/supply_chain",
        "params": {
            "query": "What is the standard geopolitical risk assessment and sourcing concentration risk for importing raw battery-grade Cobalt from suppliers located in the DRC versus Australia?"
        }
    },
    "fleet_electrification": {
        "url": f"{BASE_URL}/fleet_electrification",
        "params": {
            "query": "How much money can we save by switching a diesel delivery truck to an EV equivalent? Provide a conceptual ROI."
        }
    },
    "maintenance_operations": {
        "url": f"{BASE_URL}/maintenance_operations",
        "params": {
            "query": "What are the best predictive maintenance triggers for a commercial EV fleet to prevent unplanned downtime from battery coolant leaks?"
        }
    },
    "carbon_tracker": {
        "url": f"{BASE_URL}/carbon_tracker",
        "params": {
            "query": "Explain how switching from an internal combustion fleet to an EV fleet impacts both Scope 1 direct emissions and Scope 3 supply chain emissions."
        }
    }
}

for agent, config in queries.items():
    print(f"\n{'='*80}")
    print(f"Testing {agent}")
    print(f"{'='*80}")
    try:
        query_string = urllib.parse.urlencode(config["params"])
        full_url = f"{config['url']}?{query_string}"
        response = requests.get(full_url)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            # Try to grab the final report/summary
            if agent == "ev_apm":
                print("Summary:")
                print(data.get("battery_health_summary", "No summary"))
            elif agent == "ev_qms":
                print("Summary:")
                print(data.get("root_cause_analysis", "No summary"))
            elif agent == "supply_chain":
                print("Summary:")
                print(data.get("unified_report", "No report generated."))
            elif agent == "fleet_electrification":
                print("Summary:")
                print(data.get("readiness_summary", "No summary"))
            elif agent == "maintenance_operations":
                print("Summary:")
                print(data.get("predictive_insights", "No summary"))
            elif agent == "carbon_tracker":
                print("Summary:")
                print(data.get("unified_report", "No summary"))
                print(f"Status: {data.get('status')}")
            
            print("\nRaw Tool Output Snippet (first 300 chars):")
            print(str(data)[:300])
        else:
            print(f"Failed: {response.status_code} {response.reason}")
            print(response.text)
            
    except Exception as e:
        print(f"Failed: {e}")
        
    time.sleep(2)
