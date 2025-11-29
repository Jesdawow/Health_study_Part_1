import numpy as np
import pandas as pd

def make_bp_regression_data(df: pd.DataFrame):
    """
    Prepare data for OLS regression predicting systolic blood pressure from age and weight.
    Args:
        df: DataFrame with "systolic_bp", "age", and "weight" columns
    Returns:
        X: Design matrix (numpy ndarray) with intercept, age, and weight
        y: Target vector (numpy ndarray) with systolic blood pressure
        feature_names: List of feature names corresponding to columns in X
        idx: Index of rows used in regression (for plotting)
    """
    cols = ["age", "weight"]
    clean = df.dropna(subset=cols + ["systolic_bp"]).copy()

    X = clean[cols].to_numpy(dtype=float)
    X = np.c_[np.ones(X.shape[0]), X]

    y = clean["systolic_bp"].to_numpy(dtype=float)
    feature_names = ["intercept"] + cols

    return X, y, feature_names, clean.index

def ols_fit(X: np.ndarray, y: np.ndarray):
    """
    Fit OLS regression using matrix algebra.
    Args:
        X: Design matrix (numpy ndarray)
        y: Target vector (numpy ndarray)
    Returns:
        beta: Coefficient vector (numpy ndarray)
        y_hat: Fitted values (numpy ndarray)
        residuals: Residuals (numpy ndarray)
    """
    XtX = X.T @ X
    Xty = X.T @ y

    beta = np.linalg.solve(XtX, Xty)
    y_hat = X @ beta
    residuals = y - y_hat

    return beta, y_hat, residuals