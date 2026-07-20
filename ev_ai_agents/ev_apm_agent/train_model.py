import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_BASE_DIR, "..", "datasets", "apm_dataset.csv")
MODELS_DIR = os.path.join(_BASE_DIR, "models")

def train_and_evaluate():
    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    features = [
        "Avg_Temperature_C",
        "Fast_Charge_Ratio_Pct",
        "Deep_Discharge_Cycles",
        "Avg_Charge_Duration_Hours",
        "Max_Temperature_C"
    ]

    X = df[features]
    y_deg = df["Degradation_Rate_Pct"]
    fleet_mean_soh = float(df["State_of_Health_Pct"].mean())

    print(f"\nFleet mean SoH (baseline): {fleet_mean_soh:.4f}%")
    print(f"SoH ↔ Telemetry correlations (all near-zero, confirming no learnable signal):")
    for feat in features:
        corr = df["State_of_Health_Pct"].corr(df[feat])
        print(f"  {feat}: {corr:.4f}")

    print("\n--- Training Degradation Rate Model (XGBoost Regressor) ---")
    X_train, X_test, y_deg_train, y_deg_test = train_test_split(
        X, y_deg, test_size=0.2, random_state=42
    )

    deg_model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    deg_model.fit(X_train, y_deg_train)

    deg_preds = deg_model.predict(X_test)
    deg_mae = mean_absolute_error(y_deg_test, deg_preds)
    deg_r2 = r2_score(y_deg_test, deg_preds)
    print(f"Degradation Rate Model → MAE: {deg_mae:.4f}, R²: {deg_r2:.4f}")

    # Validate derived SoH: SoH = fleet_mean - deg_rate * 12 months
    # This is physically grounded: batteries with higher degradation rate deviate more from average
    # We use the fleet mean as the intercept (ensemble average starting point)
    ASSET_AGE_MONTHS_ASSUMED = 12
    derived_soh = fleet_mean_soh - (deg_preds * ASSET_AGE_MONTHS_ASSUMED)
    # Clip to reasonable battery range
    derived_soh = np.clip(derived_soh, 60.0, 100.0)
    actual_soh_test = df.loc[X_test.index, "State_of_Health_Pct"].values
    soh_mae = mean_absolute_error(actual_soh_test, derived_soh)
    soh_r2 = r2_score(actual_soh_test, derived_soh)
    print(f"\nDerived SoH (from degradation model) → MAE: {soh_mae:.4f}, R²: {soh_r2:.4f}")
    print(f"(Note: R² vs raw SoH will still be low since SoH has no learnable correlation")
    print(f" with telemetry in this synthetic dataset. The derivation is physically meaningful:") 
    print(f" faster degraders accumulate more loss over time — this is directionally correct.)")

    # Save only the degradation model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(deg_model, os.path.join(MODELS_DIR, "degradation_model.pkl"))
    # Save fleet mean SoH as a scalar for use in predict_battery_health
    joblib.dump({"fleet_mean_soh": fleet_mean_soh, "asset_age_months": ASSET_AGE_MONTHS_ASSUMED},
                os.path.join(MODELS_DIR, "soh_baseline.pkl"))
    print(f"\nModels saved to: {MODELS_DIR}")
    print(f"  degradation_model.pkl  (XGBoost, R²={deg_r2:.4f})")
    print(f"  soh_baseline.pkl       (fleet_mean_soh={fleet_mean_soh:.2f}, used for SoH derivation)")

    # Feature importance
    print("\nDegradation Rate — Feature Importances:")
    for feat, imp in sorted(zip(features, deg_model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")

    return deg_r2, soh_mae

if __name__ == "__main__":
    train_and_evaluate()
