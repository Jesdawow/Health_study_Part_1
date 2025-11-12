import numpy as np
from typing import Tuple, Optional
from scipy import stats

def ci_mean_normal(data: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    x = np.asarray(data, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return (np.nan, np.nan)
    xbar = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    z = float(stats.norm.ppf(1 - alpha/2))
    half = z * s / np.sqrt(n)
    return (xbar-half, xbar + half)