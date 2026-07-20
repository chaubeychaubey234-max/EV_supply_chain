import os
import glob

# 1. Fleet Agent
fleet_path = "ev_ai_agents/ev_fleet_electrification_agent/agent.py"
with open(fleet_path, "r") as f:
    content = f.read()
    
# Replace the tool loop in fleet agent
old_loop = """        try:
            log.info(f"Executing tool: {tool_name} for vehicle: {vehicle_id}")
            if tool_name == "analyze_fleet_csv":
                result = tool_callable.invoke({})
            else:
                result = tool_callable.invoke({"vehicle_id": vehicle_id})
            tool_outputs[tool_name] = result"""

new_loop = """        try:
            log.info(f"Executing tool: {tool_name} for vehicle: {vehicle_id}")
            if tool_name == "analyze_fleet_csv":
                result = tool_callable.invoke({})
            else:
                if not vehicle_id:
                    result = {"message": f"Tool {tool_name} skipped. No specific vehicle_id provided. Answer conceptually based on user query description."}
                else:
                    result = tool_callable.invoke({"vehicle_id": vehicle_id})
            tool_outputs[tool_name] = result"""

content = content.replace(old_loop, new_loop)
with open(fleet_path, "w") as f:
    f.write(content)

# 2. QMS Agent
qms_path = "ev_ai_agents/ev_qms_agent/agent.py"
with open(qms_path, "r") as f:
    content = f.read()

old_loop_qms = """        try:
            logger.info(f"Executing tool: {tool_name}")
            if tool_name == "predict_quality_drift":
                inputs = {
                    "ambient_temp_c": state.get("ambient_temp_c", 22.0),
                    "mixer_speed_rpm": state.get("mixer_speed_rpm", 1500.0),
                    "coating_thickness_um": state.get("coating_thickness_um", 100.0),
                    "calendering_pressure_mpa": state.get("calendering_pressure_mpa", 50.0),
                    "formation_current_a": state.get("formation_current_a", 10.0)
                }
                result = tool_callable.invoke(inputs)
            elif tool_name == "aggregate_qms_statistics":
                result = tool_callable.invoke({"metric": agg_metric})
            else:
                result = tool_callable.invoke({"batch_id": batch_id})"""

new_loop_qms = """        try:
            logger.info(f"Executing tool: {tool_name}")
            if tool_name == "predict_quality_drift":
                inputs = {
                    "ambient_temp_c": state.get("ambient_temp_c", 22.0),
                    "mixer_speed_rpm": state.get("mixer_speed_rpm", 1500.0),
                    "coating_thickness_um": state.get("coating_thickness_um", 100.0),
                    "calendering_pressure_mpa": state.get("calendering_pressure_mpa", 50.0),
                    "formation_current_a": state.get("formation_current_a", 10.0)
                }
                result = tool_callable.invoke(inputs)
            elif tool_name == "aggregate_qms_statistics":
                result = tool_callable.invoke({"metric": agg_metric})
            else:
                if not batch_id:
                    result = {"message": f"Tool {tool_name} skipped. No specific batch_id provided. Answer conceptually based on user query description."}
                else:
                    result = tool_callable.invoke({"batch_id": batch_id})"""
content = content.replace(old_loop_qms, new_loop_qms)
with open(qms_path, "w") as f:
    f.write(content)

# 3. Supply Chain Agent
supply_path = "ev_ai_agents/ev_supply_chain_agent/agent.py"
with open(supply_path, "r") as f:
    content = f.read()

old_loop_supply = """        try:
            logger.info(f"Executing tool: {tool_name}")
            if tool_name == "assess_geopolitical_risk":
                result = tool_callable.invoke({"country_code": state.get("country", "CN")})
            elif tool_name == "trace_material_batch":
                result = tool_callable.invoke({"batch_id": batch_id})
            elif tool_name == "map_battery_passport":
                result = tool_callable.invoke({"vehicle_vin": state.get("country", "")}) # Dummy map
            else:
                result = tool_callable.invoke({"supplier_id": supplier_id})"""

new_loop_supply = """        try:
            logger.info(f"Executing tool: {tool_name}")
            if tool_name == "assess_geopolitical_risk":
                result = tool_callable.invoke({"country_code": state.get("country", "CN")})
            elif tool_name == "trace_material_batch":
                if not batch_id:
                    result = {"message": "No specific batch_id provided. Answer conceptually."}
                else:
                    result = tool_callable.invoke({"batch_id": batch_id})
            elif tool_name == "map_battery_passport":
                result = tool_callable.invoke({"vehicle_vin": state.get("country", "")})
            else:
                if not supplier_id:
                    result = {"message": "No specific supplier_id provided. Answer conceptually."}
                else:
                    result = tool_callable.invoke({"supplier_id": supplier_id})"""

content = content.replace(old_loop_supply, new_loop_supply)
with open(supply_path, "w") as f:
    f.write(content)


# 4. APM Agent
apm_path = "ev_ai_agents/ev_apm_agent/agent.py"
with open(apm_path, "r") as f:
    content = f.read()

old_loop_apm = """        try:
            logger.info(f"Executing tool: {tool_name}")
            if tool_name == "predict_battery_health":
                inputs = {
                    "avg_temperature_c": state.get("avg_temperature_c", 25.0),
                    "fast_charge_ratio_pct": state.get("fast_charge_ratio_pct", 30.0),
                    "deep_discharge_cycles": state.get("deep_discharge_cycles", 5),
                    "avg_charge_duration_hours": state.get("avg_charge_duration_hours", 4.0),
                    "max_temperature_c": state.get("max_temperature_c", 45.0)
                }
                result = tool_callable.invoke(inputs)
            elif tool_name == "aggregate_apm_statistics":
                result = tool_callable.invoke({"metric": agg_metric})
            else:
                result = tool_callable.invoke({"ev_id": ev_id})"""

new_loop_apm = """        try:
            logger.info(f"Executing tool: {tool_name}")
            if tool_name == "predict_battery_health":
                inputs = {
                    "avg_temperature_c": state.get("avg_temperature_c", 25.0),
                    "fast_charge_ratio_pct": state.get("fast_charge_ratio_pct", 30.0),
                    "deep_discharge_cycles": state.get("deep_discharge_cycles", 5),
                    "avg_charge_duration_hours": state.get("avg_charge_duration_hours", 4.0),
                    "max_temperature_c": state.get("max_temperature_c", 45.0)
                }
                result = tool_callable.invoke(inputs)
            elif tool_name == "aggregate_apm_statistics":
                result = tool_callable.invoke({"metric": agg_metric})
            else:
                if not ev_id:
                    result = {"message": f"Tool {tool_name} skipped. No specific ev_id provided. Answer conceptually based on user query description."}
                else:
                    result = tool_callable.invoke({"ev_id": ev_id})"""

content = content.replace(old_loop_apm, new_loop_apm)
with open(apm_path, "w") as f:
    f.write(content)

print("Patched executors successfully")
