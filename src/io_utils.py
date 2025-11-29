import pandas as pd
from src.paths import HEALTH_FILE

NUMERIC = ["age", "height", "weight", "systolic_bp", "cholesterol"]
CAT_COLUMS = ["sex", "smoker"]
REQUIRED = NUMERIC + CAT_COLUMS + ["disease"]

def load_data(path=HEALTH_FILE) -> pd.DataFrame:
    """Loads health study data from a CSV file and checks for required columns.
    Args:
        path (Path | str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded data.
    Raises:
        ValueError: If any required columns are missing.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing a required column: {missing}")
    return df

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """ Converts columns to numeric / categorical types and computes BMI.
    - Numeric columns are converted using pd.to_numeric with errors='coerce' to NaN.
    - Categorical columns are converted to 'category' dtype.
    - 'disease' column is converted to nullable integer type 'Int64'.
    - BMI is calculated as weight (kg) / (height (m))^2 and added as a new column.
    Args:
        df (pd.DataFrame): Input data frame.
    Returns:
        pd.DataFrame: Data frame with coerced types and BMI column.
    """
    out = df.copy()

    for c in NUMERIC:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in CAT_COLUMS:
        out[c] = out[c].astype("category")
    
    out["disease"] = pd.to_numeric(out["disease"], errors="coerce").astype("Int64")

    h_m = out["height"] / 100.0
    out["bmi"] = (out["weight"] / (h_m**2)).where(h_m > 0)

    return out