import numpy as np
from pathlib import Path
import pandas as pd

from src.paths import HEALTH_FILE
from src.io_utils import load_data, coerce_types
from src.metrics import descriptive_stats, disease_share
from src.stats import (ci_mean_normal,
                       ci_mean_t,
                       ci_mean_bootstrap,
                       welch_t_test,
                       bootstrap_mean_diff_pvalue
                       )
from src import models
DEFAULT_SEED = 42

class HealthAnalysis:
    """
    Main class for health study analysis.

    Responsibilities:
    - Load and clean the data
    - Provide methods for descriptive statistics
    - Compute disease rates
    - Simulate disease occurrence
    - Compute confidence intervals for systolic blood pressure
    - Perform t-tests comparing smokers and non-smokers
    - Simulate statistical power for detecting differences
    - Fit OLS regression models predicting systolic blood pressure from age and weight
   """
    
    def __init__(self, path: Path | None = None, seed: int = DEFAULT_SEED):
        # Initializes the HealthAnalysis class with data loaded and types coerced
        if path is None:
            path = HEALTH_FILE

        df = load_data(path)
        self.df = coerce_types(df)
        self.rng = np.random.default_rng(seed)
    
    def descriptive(self) -> pd.DataFrame:
        """
        Return means, medians, mins, and maxes for selected numeric columns.
        """
        return descriptive_stats(self.df)
    
    def disease_rate(self) -> float:
        """
        Compute the share of participants with a disease = 1
        """
        return disease_share(self.df)
    
    def simulate_disease(self, n: int = 1000) -> float:
        """
        Simulate n participants with disease/no disease based on observed disease rate.
        """
        p = self.disease_rate()
        sims = self.rng.binomial(1, p, size=n)
        return float(np.mean(sims))
    
    def ci_bp_normal(self, alpha: float = 0.05):
        # Confidence interval for mean systolic blood pressure using normal approximation
        x = self.df["systolic_bp"].dropna().to_numpy()
        return ci_mean_normal(x, alpha=alpha)
    
    def ci_bp_t(self, alpha: float = 0.05):
        # Confidence interval for mean systolic blood pressure using t-distribution
        x = self.df["systolic_bp"].dropna().to_numpy()
        return ci_mean_t(x, alpha=alpha)
    
    def ci_bp_bootstrap(self, alpha: float = 0.05, n_boot: int = 5000):
        # Confidence interval for mean systolic blood pressure using bootstrap
        x = self.df["systolic_bp"].dropna().to_numpy()
        return ci_mean_bootstrap(x, alpha=alpha, n_boot=n_boot, seed=DEFAULT_SEED)
    
    def smoker_bp_ttests(self):
        """
        Perform Welch's t-test and bootstrap test comparing systolic blood pressure
        Returns:
            t_stat: t-statistic from Welch's t-test
            p_two: two-sided p-value from Welch's t-test
            p_boot: p-value from bootstrap test
        """
        mask_smoker = self.df["smoker"].astype(str).str.lower() == "yes"

        x = self.df.loc[mask_smoker, "systolic_bp"].dropna().to_numpy()
        y = self.df.loc[~mask_smoker, "systolic_bp"].dropna().to_numpy()

        t_stat, p_two = welch_t_test(x, y)
        p_boot = bootstrap_mean_diff_pvalue(x, y, n_boot=5000, seed=DEFAULT_SEED)
        return t_stat, p_two, p_boot
    
    def simulate_power(self, delta: float = 5.0, n_sim: int = 300, alpha: float = 0.05) -> float:
        """
        Estimate test power when smokers truly have higher systolic blood pressure by 'delta' mm Hg.
        """
        base = self.df.copy()
        count = 0
        for _ in range(n_sim):
            sim = base.copy()

            mask = sim.df["smoker"].astype(str).str.lower() == "yes"
            sim.loc[mask, "systolic_bp"] += delta
            x = sim.loc[mask, "systolic_bp"].dropna().to_numpy()
            y = sim.loc[~mask, "systolic_bp"].dropna().to_numpy()

            p_boot = bootstrap_mean_diff_pvalue(x, y, n_boot=2000, seed=self.rng.integers(0, 1_000_000))
            if p_boot < alpha:
                count += 1
        return count / n_sim
    
    def fit_bp_regression(self):
        """
        Fit a linear regression model predicting systolic blood pressure from age and weight.
        Uses explicit matrix algebra (X'X)^{-1}X'y.
        """
        X, y, feature_names, idx = models.make_bp_regression_data(self.df)
        beta, y_hat, residuals = models.ols_fit(X, y)

        return {
            "beta": beta,
            "feature_names": feature_names,
            "y_hat": y_hat,
            "residuals": residuals,
            "index": idx,
        }
    def fit_bp_regression_age(self) -> dict:
        """
        Fit a linear regression model predicting systolic blood pressure from age only.
        Returns:
            Dictionary with keys:
            - "beta": Coefficient vector (intercept and slope)
            - "feature_names": List of feature names
            - "y_hat": Fitted values
            - "residuals": Residuals
            - "index": Index of rows used in regression
        """
        df = self.df.dropna(subset=["systolic_bp", "age"]).copy()

        age = df["age"].to_numpy(dtype=float)
        y = df["systolic_bp"].to_numpy(dtype=float)

        X = np.column_stack([np.ones_like(age), age])

        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        intercept, slope = beta

        y_hat = intercept + slope * age
        residuals = y - y_hat
        return {
            "beta": beta,
            "feature_names": ["intercept", "age"],
            "y_hat": y_hat,
            "residuals": residuals,
            "index": df.index.to_numpy(),
        }