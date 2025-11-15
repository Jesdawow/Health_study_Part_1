import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.paths import IMAGES_DIR
from src.metrics import group_mean

def _plain_y(ax):
    ax.ticklabel_format(style="plain", axis="y")

def hist_bp(df: pd.DataFrame) -> None:
    # Histogram of systolic blood pressure
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
    # Box plot of weight by sex
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
    # Bar chart of smoker share
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
    # Scatter plot of systolic blood pressure vs age
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
    # Bar chart of mean systolic blood pressure by smoker status
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