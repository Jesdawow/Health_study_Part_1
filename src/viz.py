import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.paths import IMAGES_DIR
from src.metrics import group_mean

def _plain_y(ax):
    """
    Set y-axis to plain number format (no scientific notation)
    """
    ax.ticklabel_format(style="plain", axis="y")

def hist_bp(df: pd.DataFrame) -> None:
    """
    Plot histogram of systolic blood pressure
    Args:
        df: DataFrame with "systolic_bp" column
    """
    vals = df["systolic_bp"].dropna().to_numpy()

    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=20, edgecolor="black", alpha=0.7)
    plt.title("Histogram of Systolic Blood Pressure")
    plt.xlabel("Systolic Blood Pressure (mm Hg)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "hist_systolic_bp.png", dpi=200)
    plt.show()

def box_weight_by_sex(df: pd.DataFrame) -> None:
    """
    Plots a boxplot ofweight, split by sex (M/F)
    Args:
        df: DataFrame with "weight" and "sex" columns"
    """
    m = df.loc[df["sex"].astype(str) == "M", "weight"].dropna().to_numpy()
    f = df.loc[df["sex"].astype(str) == "F", "weight"].dropna().to_numpy()

    plt.figure(figsize=(8, 5))
    plt.boxplot([m, f])
    plt.xticks([1, 2], ["M", "F"])
    plt.title("Boxplot: Weight by Sex")
    plt.ylabel("Weight (kg)")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "box_weight_by_sex.png", dpi=200)
    plt.show()

def bar_smoker_share(df: pd.DataFrame) -> None:
    """
    Bar chart of share of smokers in the dataset
    Arg:
        df: DataFrame with "smoker" column
    """
    s = df["smoker"].astype(str).str.title()
    shares = (s.value_counts(normalize=True).sort_index() * 100)

    x = list(shares.index.astype(str))
    y = shares.to_numpy(dtype=float)

    plt.figure(figsize=(6, 4))
    plt.bar(x,y,edgecolor="black")
    plt.title("Share of Smokers")
    plt.ylabel("Percentage (%)")
    _plain_y(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "bar_smoker_share.png", dpi=200)
    plt.show()

def scatter_bp_vs_age(df: pd.DataFrame) -> None:
    """
    Scatter plot of systolic blood pressure vs age
    Args:
        df: DataFrame with "systolic_bp" and "age" columns
    """
    x = df["age"].to_numpy()
    y = df["systolic_bp"].to_numpy()

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.5, s=15, edgecolors="w")
    plt.title("Systolic Blood Pressure vs Age")
    plt.xlabel("Age (years)")
    plt.ylabel("Systolic Blood Pressure (mm Hg)")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "scatter_bp_vs_age.png", dpi=200)
    plt.show()

def bar_mean_bp_smoker(df: pd.DataFrame) -> None:
    """
    Bar chart of mean systolic blood pressure by smoker status
    Args:
        df: DataFrame with "systolic_bp" and "smoker" columns
    """
    means = group_mean(df, "systolic_bp", "smoker")

    x = list(means.index.astype(str))
    y = means.to_numpy(dtype=float)

    plt.figure(figsize=(6, 4))
    plt.bar(x,y,edgecolor="black")
    plt.title("Mean Systolic BP by Smoker Status")
    plt.ylabel("mm Hg")
    _plain_y(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "bar_mean_bp_smoker.png", dpi=200)
    plt.show()

def scatter_bp_vs_age_with_line(df: pd.DataFrame, y_hat: np.ndarray, idx) -> None:
    """
    Scatter plot of systolic blood pressure vs age with regression line
    Args:
        df: DataFrame with "systolic_bp" and "age" columns
        y_hat: Fitted systolic blood pressure values from regression
        idx: Index of rows used in regression (to align y_hat with df)
    """
    x = df.loc[idx,"age"].to_numpy(dtype=float)
    y = df.loc[idx,"systolic_bp"].to_numpy(dtype=float)

    order = np.argsort(x)
    x_sorted = x[order]
    y_hat_sorted = y_hat[order]

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.4, s=15, edgecolors="w", label="Data")
    plt.plot(x_sorted, y_hat_sorted, color="red", linewidth=2, label="Regression Line")
    plt.title("Systolic BP vs Age with Regression Line")
    plt.xlabel("Age (years)")
    plt.ylabel("Systolic BP (mm Hg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "scatter_bp_vs_age_with_line.png", dpi=200)
    plt.show()