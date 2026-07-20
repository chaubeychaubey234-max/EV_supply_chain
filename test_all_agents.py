import requests
import json
import urllib.parse

BASE_URL = "http://localhost:8080/api/agents"

queries = {
    "ev_apm": ("user_query", "Analyze the expected degradation rate and thermal runaway risks for an electric transit van operating in a high-temperature desert climate (45°C) that relies almost entirely on daily fast charging."),
    "ev_qms": ("user_query", "If we experience a sudden spike in internal resistance during end-of-line testing for our LFP cells, and we suspect the electrolyte filling volume is too low, what process drift could be happening at the mixing or calendering stage?"),
    "supply_chain": ("query", "What is the standard geopolitical risk assessment and sourcing concentration risk for importing raw battery-grade Cobalt from suppliers located in the DRC versus Australia?"),
    "fleet_electrification": ("query", "What is the typical 5-year ROI and payback period for replacing a heavy-duty diesel delivery truck with an EV equivalent, assuming a daily route of 150 miles and overnight depot charging?"),
    "maintenance_operations": ("query", "What are the best predictive maintenance triggers for a commercial electric vehicle fleet to prevent unplanned downtime, specifically focusing on early warning signs for battery coolant leaks?"),
    "carbon_tracker": ("query", "Explain how switching from an internal combustion fleet to an EV fleet impacts both Scope 1 direct emissions and Scope 3 supply chain emissions, especially factoring in the embedded carbon of battery manufacturing.")
}

for endpoint, (param_name, q) in queries.items():
    print(f"\n{'='*80}\nTesting {endpoint}\n{'='*80}")
    url = f"{BASE_URL}/{endpoint}?{param_name}={urllib.parse.quote(q)}"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        print(f"Status Code: {resp.status_code}")
        
        # Check if hallucinated IDs exist
        hallucinated_keys = ['ev_id', 'batch_id', 'supplier_id']
        for k in hallucinated_keys:
            if k in data and data[k] not in [None, "", "RAW_INPUT", "QUERY_RESULT"]:
                print(f"⚠️ WARNING: Possible hallucinated ID: {k} = {data[k]}")
                
        if 'summary' in data:
            print(f"Summary:\n{data['summary']}")
        elif 'messages' in data:
            print(f"Messages:\n{data['messages'][-1] if data['messages'] else 'Empty'}")
        elif 'ai_insight' in data:
            print(f"Insight:\n{data['ai_insight']}")
        else:
            print("Response snippet:", str(data)[:500])
            
    except Exception as e:
        print(f"Failed: {e}")
        try:
            print(resp.text)
        except:
            pass
