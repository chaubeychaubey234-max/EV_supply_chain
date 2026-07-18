import os
import shutil
import kagglehub

def download_datasets():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.abspath(os.path.join(CURRENT_DIR, "datasets"))
    os.makedirs(datasets_dir, exist_ok=True)
    
    print("Downloading QMS dataset...")
    qms_path = kagglehub.dataset_download("kanchana1990/ev-battery-qc-synthetic-defect-dataset")
    print(f"QMS downloaded to: {qms_path}")
    
    # Copy files to our local datasets folder
    for file in os.listdir(qms_path):
        if file.endswith('.csv'):
            shutil.copy(os.path.join(qms_path, file), os.path.join(datasets_dir, "qms_dataset.csv"))
            print(f"Copied {file} to datasets/qms_dataset.csv")

    print("Downloading APM dataset...")
    # APM dataset choice: Electric Vehicle (EV) Battery Degradation & Charge
    apm_path = kagglehub.dataset_download("bertnardomariouskono/electric-vehicle-ev-battery-degradation-charge")
    print(f"APM downloaded to: {apm_path}")
    
    for file in os.listdir(apm_path):
        if file.endswith('.csv'):
            shutil.copy(os.path.join(apm_path, file), os.path.join(datasets_dir, "apm_dataset.csv"))
            print(f"Copied {file} to datasets/apm_dataset.csv")

if __name__ == "__main__":
    download_datasets()
