import os, glob

base = "/Users/alex-ankush/Desktop/EV/EV_supply_chain/ev_ai_agents"
shared_ds = os.path.join(base, "datasets")

# 1. Fix APM tools
for tool in glob.glob(os.path.join(base, "ev_apm_agent/tools/*.py")):
    with open(tool, 'r') as f: content = f.read()
    content = content.replace('"/Users/alex-ankush/Desktop/EV/datasets/apm_dataset.csv"', f'"{shared_ds}/apm_dataset.csv"')
    with open(tool, 'w') as f: f.write(content)

# 2. Fix QMS tools
for tool in glob.glob(os.path.join(base, "ev_qms_agent/tools/*.py")):
    with open(tool, 'r') as f: content = f.read()
    content = content.replace('"/Users/alex-ankush/Desktop/EV/datasets/qms_dataset.csv"', f'"{shared_ds}/qms_dataset.csv"')
    with open(tool, 'w') as f: f.write(content)

# 3. Fix Maintenance tools
# It expects them in ev_maintenance_operations_agent/dataset/ but they are in ev_maintenance_operations_agent/datasets/
maint_utils = os.path.join(base, "ev_maintenance_operations_agent/tools/utils.py")
if os.path.exists(maint_utils):
    with open(maint_utils, 'r') as f: content = f.read()
    content = content.replace('dataset', 'datasets') # Replace dataset with datasets folder
    with open(maint_utils, 'w') as f: f.write(content)

# 4. Fix Supply Chain
sc_datasets = os.path.join(base, "ev_supply_chain_agent/datasets.py")
if os.path.exists(sc_datasets):
    with open(sc_datasets, 'r') as f: content = f.read()
    # Replace the DATASETS_DIR to point to shared_ds
    content = content.replace('os.path.join(PROJECT_ROOT, "datasets")', f'"{shared_ds}"')
    with open(sc_datasets, 'w') as f: f.write(content)

print("Paths updated.")
