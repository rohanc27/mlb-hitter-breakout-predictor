"""Generate charts for the README: model comparison, feature importance,
calibration/probability distribution by outcome.

Usage:
    python scripts/make_readme_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

FEATURES = [
    "Age", "WAR", "PA", "BA", "OBP", "SLG", "OPS",
    "bb_rate", "k_rate", "hr_rate", "iso", "babip",
    "war_trend", "ops_trend",
]


def fig_model_comparison():
    """Bar chart comparing logreg vs xgb on AUC, PR-AUC, Brier."""
    logreg_metrics = json.loads((MODELS_DIR / "logreg.metrics.json").read_text())
    xgb_metrics = json.loads((MODELS_DIR / "xgb.metrics.json").read_text())

    metrics_to_plot = ["auc", "pr_auc"]
    labels = ["AUC", "PR-AUC"]
    logreg_vals = [logreg_metrics[m] for m in metrics_to_plot]
    xgb_vals = [xgb_metrics[m] for m in metrics_to_plot]

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars1 = ax.bar(x - width/2, logreg_vals, width, label="Logistic Regression", color="#3b82f6")
    bars2 = ax.bar(x + width/2, xgb_vals, width, label="XGBoost (tuned)", color="#94a3b8")

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                     ha="center", fontsize=10, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, max(logreg_vals + xgb_vals) * 1.25)
    ax.set_title("Model comparison (2022-2023 test set)", fontsize=13, weight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=200)
    plt.close(fig)
    print("Saved model_comparison.png")


def fig_feature_coefficients():
    """Horizontal bar chart of logreg standardized coefficients."""
    model = joblib.load(MODELS_DIR / "logreg.joblib")
    classifier = model.named_steps["classifier"]
    coefs = classifier.coef_[0]

    order = np.argsort(np.abs(coefs))
    sorted_features = [FEATURES[i] for i in order]
    sorted_coefs = coefs[order]
    colors = ["#21c55d" if c > 0 else "#ef4444" for c in sorted_coefs]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(sorted_features, sorted_coefs, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardized coefficient (impact on breakout log-odds)", fontsize=11)
    ax.set_title("What drives the breakout prediction?", fontsize=13, weight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_coefficients.png", dpi=200)
    plt.close(fig)
    print("Saved feature_coefficients.png")


def fig_probability_by_outcome():
    """Distribution of predicted probabilities, split by actual breakout outcome."""
    model = joblib.load(MODELS_DIR / "logreg.joblib")
    df = pd.read_parquet(PROCESSED_DIR / "breakout_features.parquet")
    df = df[df["has_next_year"] & df["has_prior_year"]].copy()
    test_df = df[df["year_ID"].isin([2022, 2023])].copy()

    X_test = test_df[FEATURES]
    y_test = test_df["breakout"].astype(int)
    proba = model.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(proba[y_test == 0], bins=20, alpha=0.6, label="No breakout (actual)", color="#94a3b8", density=True)
    ax.hist(proba[y_test == 1], bins=20, alpha=0.7, label="Breakout (actual)", color="#3b82f6", density=True)
    ax.set_xlabel("Predicted breakout probability", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Predicted probability separates real outcomes", fontsize=13, weight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "probability_by_outcome.png", dpi=200)
    plt.close(fig)
    print("Saved probability_by_outcome.png")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_model_comparison()
    fig_feature_coefficients()
    fig_probability_by_outcome()
    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
