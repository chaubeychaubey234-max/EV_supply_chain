import requests
import json
import urllib.parse
import time

BASE_URL = "http://localhost:8080/api/agents"

queries = {
    "ev_apm": {
        "url": f"{BASE_URL}/ev_apm",
        "params": {
            "user_query": "Analyze the degradation rate and remaining useful life for EV-9002. Are there any thermal runaway risks?",
            "ev_id": "EV-9002"
        }
    },
    "ev_qms": {
        "url": f"{BASE_URL}/ev_qms",
        "params": {
            "user_query": "Audit the manufacturing quality for BATCH-003. We saw a spike in internal resistance during end-of-line tests.",
            "batch_id": "BATCH-003"
        }
    },
    "supply_chain": {
        "url": f"{BASE_URL}/supply_chain",
        "params": {
            "query": "Trace the origin of the Cobalt used in BATCH-004. Provide a geopolitical risk assessment for SUP-002.",
            "supplier_id": "SUP-002",
            "batch_id": "BATCH-004"
        }
    },
    "fleet_electrification": {
        "url": f"{BASE_URL}/fleet_electrification",
        "params": {
            "query": "Assess electrification readiness for VEH-003. Provide ROI comparison.",
            "vehicle_id": "VEH-003"
        }
    },
    "maintenance_operations": {
        "url": f"{BASE_URL}/maintenance_operations",
        "params": {
            "query": "What are the predictive maintenance triggers and current risk score for EV_2423?",
            "ev_id": "EV_2423"
        }
    },
    "carbon_tracker": {
        "url": f"{BASE_URL}/carbon_tracker",
        "params": {
            "query": "What is our current scope 1 emission progress and net zero status?"
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
