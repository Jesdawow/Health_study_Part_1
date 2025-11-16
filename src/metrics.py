import numpy as np
import pandas as pd

DESC = ["age", "height", "weight", "systolic_bp", "cholesterol"]

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    # Returns mean/median/min/max from the requested columns listed below
    rows = []
    for col in DESC:
        vals = df[col].dropna().to_numpy()
        rows.append({
            "metric": col,
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals))
        })
    return pd.DataFrame(rows).set_index("metric")

def smoker_share(df: pd.DataFrame) -> float:
    # Number of smokers (YES/NO) in the "smoker" column
    s = df["smoker"].astype(str).str.lower()
    return float((s == "yes").mean())

def disease_share(df: pd.DataFrame) -> float:
    # Number of people with diseases (0/1) in the disease column
    d = df["disease"].dropna().astype(int).to_numpy()
    return float(np.mean(d))

def group_mean(df: pd.DataFrame, value_col: str, by_col: str) -> pd.Series:
    # Mean per group in by_col for the value_col
    return df.groupby(by_col, observed=True)[value_col].mean().sort_values(ascending=False)