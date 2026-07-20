from dashboard_server import get_ev_apm, get_ev_qms
import os

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

print("--- APM API ---")
res_apm = get_ev_apm(user_query="What is the battery health status of EV-9005?", ev_id=None, avg_temp=None, max_temp=None, fc_ratio=None, deep_cycles=None, charge_dur=None)
print(res_apm["summary"])
print("Battery Analysis:", res_apm["battery_analysis"])

print("\n--- QMS API ---")
res_qms = get_ev_qms(user_query="Check batch BATCH-002", batch_id=None, elec_vol=None, int_res=None, cell_cap=None, amb_temp=None)
print(res_qms["summary"])
print("Cell Metrics:", res_qms["cell_metrics"])
