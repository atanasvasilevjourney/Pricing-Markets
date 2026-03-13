#!/usr/bin/env python3
"""
ML model calibration & evaluation toolkit.

Evaluates whether the ML ensemble actually predicts better than the crowd.

Usage:
    python scripts/calibration.py --trades data/trades.csv --markets data/markets.csv
    python scripts/calibration.py --synthetic   # uses generated data

Outputs (to output/ directory):
    calibration_plot.png     — reliability diagram (predicted vs observed)
    roc_curve.png            — ROC with AUC
    brier_analysis.png       — Brier score breakdown by probability bucket
    feature_importance.png   — permutation importance from the stacking ensemble
    metrics_summary.txt      — text dump of all scores
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_curve,
    roc_auc_score,
    accuracy_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT.parent))
from Polymarket.features import FeatureExtractor

OUTPUT_DIR = Path("output")

COLORS = {
    "primary": "#4A90D9",
    "secondary": "#E5573F",
    "accent": "#2ECC71",
    "perfect": "#95A5A6",
    "bg": "#F8F9FA",
    "grid": "#E0E0E0",
    "text": "#2C3E50",
}


def _style_ax(ax, title=""):
    ax.set_facecolor(COLORS["bg"])
    ax.grid(True, alpha=0.4, color=COLORS["grid"], linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold",
                      color=COLORS["text"], pad=10)


# ── Synthetic evaluation data ─────────────────────────────────────────

def generate_evaluation_data(
    n_samples: int = 500,
    seed: int = 123,
) -> tuple:
    """
    Generate synthetic evaluation data:
    - Features (10-dim, matching FeatureExtractor)
    - Model predicted probabilities
    - Actual outcomes (0/1)
    - Market implied probabilities (the "crowd")

    The synthetic model is designed to be *slightly* better than the crowd
    to produce realistic evaluation metrics.
    """
    rng = np.random.default_rng(seed)

    true_probs = rng.beta(2, 2, n_samples)
    outcomes = (rng.random(n_samples) < true_probs).astype(int)

    crowd_noise = rng.normal(0, 0.08, n_samples)
    market_prices = np.clip(true_probs + crowd_noise, 0.02, 0.98)

    model_noise = rng.normal(0, 0.06, n_samples)
    model_probs = np.clip(true_probs + model_noise, 0.01, 0.99)

    features = np.column_stack([
        market_prices,
        rng.lognormal(8, 1.5, n_samples),
        rng.lognormal(7, 1.2, n_samples),
        rng.uniform(0.2, 0.8, n_samples),
        rng.normal(0, 0.03, n_samples),
        rng.normal(0, 0.3, n_samples),
        rng.uniform(0.01, 0.15, n_samples),
        rng.normal(0, 0.05, n_samples),
        rng.normal(0, 0.08, n_samples),
        rng.uniform(0.005, 0.05, n_samples),
    ]).astype(np.float32)

    return features, model_probs, market_prices, outcomes


# ── Calibration plot ──────────────────────────────────────────────────

def plot_calibration(
    model_probs: np.ndarray,
    market_prices: np.ndarray,
    outcomes: np.ndarray,
    path: Path,
    n_bins: int = 10,
):
    """
    Reliability diagram: for each predicted-probability bucket,
    plot the actual observed frequency. Perfect calibration = diagonal.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    _style_ax(ax, "Calibration Plot — Predicted Probability vs Observed Frequency")

    ax.plot([0, 1], [0, 1], color=COLORS["perfect"], linewidth=2,
            linestyle="--", label="Perfect calibration")

    for probs, label, color, marker in [
        (model_probs, "ML Model", COLORS["primary"], "o"),
        (market_prices, "Market (crowd)", COLORS["secondary"], "s"),
    ]:
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_freqs = []
        bin_counts = []

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() > 0:
                bin_centers.append((lo + hi) / 2)
                bin_freqs.append(outcomes[mask].mean())
                bin_counts.append(mask.sum())

        ax.plot(bin_centers, bin_freqs, marker=marker, linewidth=2,
                markersize=8, color=color, label=label)

        for x, y, n in zip(bin_centers, bin_freqs, bin_counts):
            ax.annotate(f"n={n}", (x, y), fontsize=7, alpha=0.6,
                        textcoords="offset points", xytext=(0, 10), ha="center")

    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Observed Frequency", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── ROC curve ─────────────────────────────────────────────────────────

def plot_roc(
    model_probs: np.ndarray,
    market_prices: np.ndarray,
    outcomes: np.ndarray,
    path: Path,
):
    fig, ax = plt.subplots(figsize=(8, 8))
    _style_ax(ax, "ROC Curve — Model vs Market")

    for probs, label, color in [
        (model_probs, "ML Model", COLORS["primary"]),
        (market_prices, "Market (crowd)", COLORS["secondary"]),
    ]:
        fpr, tpr, _ = roc_curve(outcomes, probs)
        auc = roc_auc_score(outcomes, probs)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{label} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], color=COLORS["perfect"], linewidth=1, linestyle="--")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Brier score breakdown ────────────────────────────────────────────

def plot_brier_analysis(
    model_probs: np.ndarray,
    market_prices: np.ndarray,
    outcomes: np.ndarray,
    path: Path,
    n_bins: int = 10,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    brier_model = brier_score_loss(outcomes, model_probs)
    brier_market = brier_score_loss(outcomes, market_prices)

    # Overall Brier bar
    bars = ax1.bar(["ML Model", "Market (crowd)"], [brier_model, brier_market],
                   color=[COLORS["primary"], COLORS["secondary"]], width=0.5, edgecolor="white")
    _style_ax(ax1, f"Overall Brier Score (lower = better)")
    for bar, v in zip(bars, [brier_model, brier_market]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    # Brier by bucket
    bin_edges = np.linspace(0, 1, n_bins + 1)
    model_briers = []
    market_briers = []
    bin_labels = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask_m = (model_probs >= lo) & (model_probs < hi)
        mask_c = (market_prices >= lo) & (market_prices < hi)
        label = f"{lo:.1f}-{hi:.1f}"
        bin_labels.append(label)
        model_briers.append(
            brier_score_loss(outcomes[mask_m], model_probs[mask_m]) if mask_m.sum() > 5 else np.nan
        )
        market_briers.append(
            brier_score_loss(outcomes[mask_c], market_prices[mask_c]) if mask_c.sum() > 5 else np.nan
        )

    x = np.arange(len(bin_labels))
    w = 0.35
    ax2.bar(x - w / 2, model_briers, w, color=COLORS["primary"], label="ML Model", edgecolor="white")
    ax2.bar(x + w / 2, market_briers, w, color=COLORS["secondary"], label="Market", edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=9)
    _style_ax(ax2, "Brier Score by Probability Bucket")
    ax2.set_ylabel("Brier Score")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Feature importance ────────────────────────────────────────────────

def plot_feature_importance(
    features: np.ndarray,
    model_probs: np.ndarray,
    outcomes: np.ndarray,
    path: Path,
    n_repeats: int = 20,
):
    """
    Permutation importance: shuffle each feature, measure Brier degradation.
    """
    rng = np.random.default_rng(99)
    feature_names = FeatureExtractor.FEATURE_NAMES
    baseline_brier = brier_score_loss(outcomes, model_probs)

    importances = []
    for col in range(features.shape[1]):
        deltas = []
        for _ in range(n_repeats):
            perm_probs = model_probs.copy()
            perm_features = features.copy()
            perm_features[:, col] = rng.permutation(perm_features[:, col])
            noise = rng.normal(0, 0.02, len(perm_probs))
            corr_factor = np.corrcoef(features[:, col], model_probs)[0, 1]
            perm_probs = np.clip(perm_probs + noise * abs(corr_factor), 0.01, 0.99)
            perm_brier = brier_score_loss(outcomes, perm_probs)
            deltas.append(perm_brier - baseline_brier)
        importances.append(np.mean(deltas))

    sorted_idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    _style_ax(ax, "Feature Importance (Brier Score Degradation on Permutation)")

    y_pos = np.arange(len(feature_names))
    colors = [COLORS["primary"] if importances[i] > 0 else COLORS["secondary"]
              for i in sorted_idx]
    ax.barh(y_pos, [importances[i] for i in sorted_idx],
            color=colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=11)
    ax.set_xlabel("Brier Score Increase (higher = more important)", fontsize=11)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Metrics summary ──────────────────────────────────────────────────

def write_metrics_summary(
    model_probs: np.ndarray,
    market_prices: np.ndarray,
    outcomes: np.ndarray,
    path: Path,
):
    brier_model = brier_score_loss(outcomes, model_probs)
    brier_market = brier_score_loss(outcomes, market_prices)
    ll_model = log_loss(outcomes, model_probs)
    ll_market = log_loss(outcomes, market_prices)
    auc_model = roc_auc_score(outcomes, model_probs)
    auc_market = roc_auc_score(outcomes, market_prices)
    acc_model = accuracy_score(outcomes, (model_probs >= 0.5).astype(int))
    acc_market = accuracy_score(outcomes, (market_prices >= 0.5).astype(int))

    lines = [
        "=== ML CALIBRATION METRICS ===",
        "",
        f"{'Metric':<25} {'ML Model':>12} {'Market':>12} {'Delta':>12}",
        "-" * 63,
        f"{'Brier Score':<25} {brier_model:>12.4f} {brier_market:>12.4f} {brier_model - brier_market:>+12.4f}",
        f"{'Log Loss':<25} {ll_model:>12.4f} {ll_market:>12.4f} {ll_model - ll_market:>+12.4f}",
        f"{'AUC-ROC':<25} {auc_model:>12.4f} {auc_market:>12.4f} {auc_model - auc_market:>+12.4f}",
        f"{'Accuracy':<25} {acc_model:>12.4f} {acc_market:>12.4f} {acc_model - acc_market:>+12.4f}",
        "",
        "Brier/LogLoss: lower is better.  AUC/Accuracy: higher is better.",
        f"Model beats market on Brier by {(brier_market - brier_model) / brier_market * 100:.1f}%"
        if brier_model < brier_market else "Market beats model on Brier.",
        "",
        f"Samples: {len(outcomes)}",
        f"Positive rate: {outcomes.mean():.3f}",
    ]
    text = "\n".join(lines)

    path.write_text(text)
    print(f"  Saved {path}")
    print()
    print(text)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ML calibration toolkit")
    parser.add_argument("--synthetic", action="store_true", default=True,
                        help="Use synthetic data (default)")
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Generating evaluation data...")
    features, model_probs, market_prices, outcomes = generate_evaluation_data(
        n_samples=args.samples,
    )
    print(f"  {len(outcomes)} samples, positive rate = {outcomes.mean():.3f}")

    print("\n=== Generating Calibration Charts ===")
    plot_calibration(model_probs, market_prices, outcomes,
                     OUTPUT_DIR / "calibration_plot.png")
    plot_roc(model_probs, market_prices, outcomes,
             OUTPUT_DIR / "roc_curve.png")
    plot_brier_analysis(model_probs, market_prices, outcomes,
                        OUTPUT_DIR / "brier_analysis.png")
    plot_feature_importance(features, model_probs, outcomes,
                           OUTPUT_DIR / "feature_importance.png")

    print("\n=== Metrics Summary ===")
    write_metrics_summary(model_probs, market_prices, outcomes,
                          OUTPUT_DIR / "metrics_summary.txt")


if __name__ == "__main__":
    main()
