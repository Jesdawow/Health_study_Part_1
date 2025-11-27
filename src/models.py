import numpy as np
import pandas as pd

def make_bp_regression_data(df: pd.DataFrame):
    # Builds design matrix X and target vector y for predicting systolic blood pressure from age and weight
    cols = ["age", "weight"]
    clean = df.dropna(subset=cols + ["systolic_bp"]).copy()

    X = clean[cols].to_numpy(dtype=float)
    X = np.c_[np.ones(X.shape[0]), X]

    y = clean["systolic_bp"].to_numpy(dtype=float)
    feature_names = ["intercept"] + cols

    return X, y, feature_names, clean.index
    # Returns x: design matrix, y: target vector, feature_names: list of feature names, index: index of rows used (for plotting)

def ols_fit(X: np.ndarray, y: np.ndarray):
    # Ordinary Least Squares (OLS) regression via matrix algebra 
    XtX = X.T @ X
    Xty = X.T @ y

    beta = np.linalg.solve(XtX, Xty)
    y_hat = X @ beta
    residuals = y - y_hat

    return beta, y_hat, residuals
    # Returns beta: coefficient vector, y_hat: fitted values, residuals: y - y_hat