import numpy as np
from scipy import stats

def _clean_array(data):
    x = np.asarray(data, dtype=float)
    return x[~np.isnan(x)]

def ci_mean_normal(data, alpha: float = 0.05):
    x = _clean_array(data)
    n = x.size
    if n == 0:
        return (np.nan, np.nan)
    mean = x.mean()
    s = x.std(ddof=1)
    z = stats.norm.ppf(1 - alpha / 2)
    margin = z * s / np.sqrt(n)
    return (mean - margin, mean + margin)

def ci_mean_t(data, alpha : float = 0.05):
    x = _clean_array(data)
    n = x.size
    if n == 0:
        return (np.nan, np.nan)
    mean = x.mean()
    s = x.std(ddof=1)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin = t_crit * s / np.sqrt(n)
    return (mean - margin, mean + margin)

def ci_mean_boostrap(data, alpha: float = 0.05, n_boot: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    x = _clean_array(data)
    n = x.size
    if n == 0:
        return (np.nan, np.nan)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        boot_means.append(sample.mean())
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return (float(lower), float(upper))

def welch_t_test(x, y):
    x = np.asarray(x, dtype=float) # type: ignore[arg-type]
    x = x[~np.isnan(x)]
    y = np.asarray(y, dtype=float) # type: ignore[arg-type]
    y = y[~np.isnan(y)]
    t_stat, p_val = stats.ttest_ind(x, y, equal_var=False, nan_policy='omit')
    return float(t_stat), float(p_val)

def bootstrap_mean_diff_pvalue(x, y, n_boot: int = 5000, seed: int = 42, alternative: str = "greater") -> float:
    rng = np.random.default_rng(seed)
    x = _clean_array(x)
    y = _clean_array(y)
    d_oobs = x.mean()  - y.mean()
    pooled = np.concatenate([x, y])
    nx, n = len(x), len(y)

    count = 0
    for _ in range(n_boot):
        xb = rng.choice(pooled, size=nx, replace=True)
        yb = rng.choice(pooled, size=n, replace=True)
        d = xb.mean() - yb.mean()

        if alternative == "greater" and d >= d_oobs:
            count += d >= d_oobs
        elif alternative == "less":
            count += d <= d_oobs
        else:
            count += abs(d) >= abs(d_oobs)
    p = count / n_boot
    return float(p)