import pandas as pd
from src.paths import HEALTH_FILE

NUMERIC = ["age", "height", "weight", "systolic_bp", "cholesterol"]
CAT_COLUMS = ["sex", "smoker"]
REQUIRED = NUMERIC + CAT_COLUMS + ["disease"]

def load_data(path=HEALTH_FILE) -> pd.DataFrame:
    # Reads the CSV file as a dataframe & checks the data has the required columns
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing a required column: {missing}")
    return df

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # Coerces datatypes & adds BMI
    out = df.copy()

    for c in NUMERIC:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in CAT_COLUMS:
        out[c] = out[c].astype("category")
    
    out["disease"] = pd.to_numeric(out["disease"], errors="coerce").astype("Int64")

    h_m = out["height"] / 100.0
    out["bmi"] = (out["weight"] / (h_m**2)).where(h_m > 0)

    return out