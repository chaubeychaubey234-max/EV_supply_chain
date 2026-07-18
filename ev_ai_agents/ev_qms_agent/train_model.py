import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_BASE_DIR, "..", "datasets", "qms_dataset.csv")
MODELS_DIR = os.path.join(_BASE_DIR, "models")

def train_and_evaluate():
    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    
    # Features requested
    features = [
        "Ambient_Temp_C", 
        "Anode_Overhang_mm", 
        "Electrolyte_Volume_ml", 
        "Internal_Resistance_mOhm", 
        "Capacity_mAh", 
        "Retention_50Cycle_Pct"
    ]
    
    X = df[features]
    
    # Encode target to integers for XGBoost
    grade_map = {"Grade A": 0, "Grade B": 1, "Scrap": 2}
    y = df["QC_Grade"].map(grade_map)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training Quality Classification Model...")
    clf = XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss")
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    
    # Note: The perfect classification metrics (Precision, Recall, F1 = 1.00) are expected because
    # the synthetic dataset has been generated using deterministic rules on the selected features
    # (e.g. strict thresholds for Capacity_mAh, Retention_50Cycle_Pct, and Anode_Overhang_mm).
    # These metrics do not reflect expected real-world production performance where noise,
    # measurement errors, and hidden variables prevent perfect separation.
    print("\nClassification Report:")
    target_names = ["Grade A", "Grade B", "Scrap"]
    print(classification_report(y_test, preds, target_names=target_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    
    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODELS_DIR, "qc_classifier.pkl"))
    print("\nModel saved successfully to models/qc_classifier.pkl")

if __name__ == "__main__":
    train_and_evaluate()
