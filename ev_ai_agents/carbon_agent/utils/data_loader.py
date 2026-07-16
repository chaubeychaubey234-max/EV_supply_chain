import os
import pandas as pd

# Define paths relative to this file's directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "datasets"))

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize pandas DataFrame column names."""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    return df

def load_green_logistics() -> pd.DataFrame:
    """Load and clean green_logistics.csv dataset."""
    path = os.path.join(DATASETS_DIR, "green_logistics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Green logistics dataset not found at {path}")
    df = pd.read_csv(path)
    return clean_columns(df)

def load_emission_factors() -> pd.DataFrame:
    """Load and clean supply_chain_emission_factors.csv dataset."""
    path = os.path.join(DATASETS_DIR, "supply_chain_emission_factors.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Supply chain emission factors dataset not found at {path}")
    df = pd.read_csv(path)
    return clean_columns(df)

def load_co2_emissions() -> pd.DataFrame:
    """Load and clean co2_emissions.csv dataset."""
    path = os.path.join(DATASETS_DIR, "co2_emissions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CO2 emissions dataset not found at {path}")
    df = pd.read_csv(path)
    return clean_columns(df)
