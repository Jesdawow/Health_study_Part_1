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
# A class for analyzing health study data
    
    def __init__(self, path: Path | None = None, seed: int = DEFAULT_SEED):
        # Initializes the HealthAnalysis class with data loaded and types coerced
        if path is None:
            path = HEALTH_FILE

        df = load_data(path)
        self.df = coerce_types(df)
        self.rng = np.random.default_rng(seed)
    
    def descriptive(self) -> pd.DataFrame:
        # Descriptive stats for each numeric column (mean, median, min, max)
        return descriptive_stats(self.df)
    
    def disease_rate(self) -> float:
        # Proportion of people with a disease
        return disease_share(self.df)
    
    def simulate_disease(self, n: int = 1000) -> float:
        # Simulates disease/no disease for n number of individuals based on observed disease rate
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
        # Welch's t-test and bootstrap p-value for difference in mean systolic blood pressure between smokers and non-smokers
        mask_smoker = self.df["smoker"].astype(str).str.lower() == "yes"

        x = self.df.loc[mask_smoker, "systolic_bp"].dropna().to_numpy()
        y = self.df.loc[~mask_smoker, "systolic_bp"].dropna().to_numpy()

        t_stat, p_two = welch_t_test(x, y)
        p_boot = bootstrap_mean_diff_pvalue(x, y, n_boot=5000, seed=DEFAULT_SEED)
        return t_stat, p_two, p_boot
    
    def simulate_power(self, delta: float = 5.0, n_sim: int = 300, alpha: float = 0.05) -> float:
        # Simulates power for detecting a difference in mean systolic blood pressure between smokers and non-smokers (The one I forgot in Part 1)
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
        # OLS regression predicting systolic blood pressure from age and weight, using matrix algebra (no statsmodels)
        X, y, feature_names, idx = models.make_bp_regression_data(self.df)
        beta, y_hat, residuals = models.ols_fit(X, y)

        return {
            "beta": beta,
            "feature_names": feature_names,
            "y_hat": y_hat,
            "residuals": residuals,
            "index": idx,
        }