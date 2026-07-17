import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_BASE_DIR, "..", "datasets", "apm_dataset.csv")
MODELS_DIR = os.path.join(_BASE_DIR, "models")

def train_and_evaluate():
    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    
    # Features specified by user
    features = [
        "Avg_Temperature_C", 
        "Fast_Charge_Ratio_Pct", 
        "Deep_Discharge_Cycles", 
        "Avg_Charge_Duration_Hours", 
        "Max_Temperature_C"
    ]
    
    X = df[features]
    y_soh = df["State_of_Health_Pct"]
    y_deg = df["Degradation_Rate_Pct"]
    
    # Train/Test Split
    X_train, X_test, y_soh_train, y_soh_test, y_deg_train, y_deg_test = train_test_split(
        X, y_soh, y_deg, test_size=0.2, random_state=42
    )
    
    # 1. Train State of Health (SoH) Model
    print("Training SoH Model...")
    soh_model = XGBRegressor(n_estimators=100, random_state=42)
    soh_model.fit(X_train, y_soh_train)
    
    soh_preds = soh_model.predict(X_test)
    soh_mae = mean_absolute_error(y_soh_test, soh_preds)
    soh_r2 = r2_score(y_soh_test, soh_preds)
    print(f"SoH Model - MAE: {soh_mae:.4f}, R²: {soh_r2:.4f}")
    
    # 2. Train Degradation Rate Model
    print("Training Degradation Rate Model...")
    deg_model = XGBRegressor(n_estimators=100, random_state=42)
    deg_model.fit(X_train, y_deg_train)
    
    deg_preds = deg_model.predict(X_test)
    deg_mae = mean_absolute_error(y_deg_test, deg_preds)
    deg_r2 = r2_score(y_deg_test, deg_preds)
    print(f"Degradation Rate Model - MAE: {deg_mae:.4f}, R²: {deg_r2:.4f}")
    
    # Save models
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(soh_model, os.path.join(MODELS_DIR, "soh_model.pkl"))
    joblib.dump(deg_model, os.path.join(MODELS_DIR, "degradation_model.pkl"))
    print("Models saved successfully to models/ directory.")

if __name__ == "__main__":
    train_and_evaluate()
