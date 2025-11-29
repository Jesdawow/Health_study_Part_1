import numpy as np
import pandas as pd

DESC = ["age", "height", "weight", "systolic_bp", "cholesterol"]

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic desciriptive statistics for selected numeric columns.
    For each column in DESC, compute mean, median, min, and max.
    Args:
        df DataFrame containing the health data
    Returns:
        DataFrame with descriptive statistics for each column in DESC
    """
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
    """
    Proportion of people who are smokers in the smoker column.
    The "smoker" column contains "yes"/"no" values (case insensitive).
    Args:
        df: DataFrame with a "smoker" column
    Returns:
        Proportion of people who are smokers (float between 0 and 1)
    """
    s = df["smoker"].astype(str).str.lower()
    return float((s == "yes").mean())

def disease_share(df: pd.DataFrame) -> float:
    """
    Compute the share of participants with a disease = 1
    Args:
        df: DataFrame with a "disease" column containing 0/1 values
    Returns:
        Fraction of rows with disease = 1 as a float between 0 and 1.
    """
    d = df["disease"].dropna().astype(int).to_numpy()
    return float(np.mean(d))

def group_mean(df: pd.DataFrame, value_col: str, by_col: str) -> pd.Series:
    """
    Compute group-wise mean of a specified column, grouped by by_col
    Args:
        df: DataFrame containing the data
        value_col: Column name for which to compute the mean
        by_col: Column name to group by (eg "smoker" or "sex")
        Returns:
        A pandas Series with the mean of value_col for each group in by_col,
        sorted in descending order.
    """
    return df.groupby(by_col, observed=True)[value_col].mean().sort_values(ascending=False)