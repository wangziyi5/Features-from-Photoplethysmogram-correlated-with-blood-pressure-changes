import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd

# Global font configuration for Arial
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = True  


def plot_combined_importance_and_stability(
    df_summary: pd.DataFrame,
    target: str,
    top_k: int,
    out_path: Path,
):
    """
    Plot side-by-side panels on a single figure:
      - Left: Top K features ranked by Mean Absolute SHAP values (Mean ± SD).
      - Right: Top K features ranked independently by Median Ranking Coefficients ([Min, Max] stability whiskers).
    """
    # 1. Left dataset: Top K sorted by Mean Absolute SHAP descending
    df_top_shap = (
        df_summary.sort_values(by="mean_abs_shap_5fold", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    # 2. Right dataset: Top K sorted by Median Ranking Coefficient descending
    df_top_rank = (
        df_summary.sort_values(
            by=["median_rank_coefficient", "mean_rank_coefficient", "mean_abs_shap_5fold"],
            ascending=False,
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    # Independent Y-axes to allow separate feature name labels and ordering
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), sharey=False)

    
    color_shap = "#6BAED6"  
    color_rank = "#B0BEC5"  
    edge_color = "#222222"

    # -------------------------------------------------------------------------
    # Panel 1: Top K Features Ranked by Mean Absolute SHAP Value
    # -------------------------------------------------------------------------
    y_pos1 = np.arange(len(df_top_shap))[::-1]  # Rank 1 at the top
    feat_names1 = df_top_shap["feature_name"].tolist()
    mean_shap = df_top_shap["mean_abs_shap_5fold"].to_numpy(dtype=float)
    std_shap = df_top_shap["std_abs_shap_5fold"].to_numpy(dtype=float)

    ax1.barh(
        y_pos1,
        mean_shap,
        xerr=std_shap,
        color=color_shap,
        edgecolor=edge_color,
        height=0.68,
        linewidth=1.2,
        capsize=4.5,
        error_kw=dict(ecolor="black", lw=1.4, capthick=1.4),
    )
    ax1.set_yticks(y_pos1)
    ax1.set_yticklabels(feat_names1, fontsize=16)
    ax1.set_xlabel(f"Mean |SHAP value| (contribution to ∆{target}, mmHg)", fontsize=18, fontweight="bold", labelpad=12)
    ax1.grid(axis="x", linestyle="--", alpha=0.5)

    
    max_val1 = np.max(mean_shap + std_shap)
    step1 = 0.2 if max_val1 <= 1.5 else (0.5 if max_val1 <= 3.0 else 1.0)
    x_upper1 = np.ceil((max_val1 * 1.08) / step1) * step1
    if target in ["DBP", "MAP"]:
        ax1.xaxis.set_major_locator(MultipleLocator(0.5))
        x_upper1 = max(x_upper1, 2.0)
        ax1.axvline(x=2, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, zorder=1)
    # ==========================
    ax1.set_xlim(0, x_upper1)

    # -------------------------------------------------------------------------
    # Panel 2: Top K Features Ranked by Median Ranking Coefficient
    # -------------------------------------------------------------------------
    y_pos2 = np.arange(len(df_top_rank))[::-1]  # Rank 1 at the top
    feat_names2 = df_top_rank["feature_name"].tolist()
    median_rank_coeff = df_top_rank["median_rank_coefficient"].to_numpy(dtype=float)
    min_rank_coeff = df_top_rank["min_rank_coefficient"].to_numpy(dtype=float)
    max_rank_coeff = df_top_rank["max_rank_coefficient"].to_numpy(dtype=float)

    xerr_low = median_rank_coeff - min_rank_coeff
    xerr_high = max_rank_coeff - median_rank_coeff
    xerr_stability = [xerr_low, xerr_high]

    ax2.barh(
        y_pos2,
        median_rank_coeff,
        color=color_rank,
        edgecolor=edge_color,
        height=0.68,
        linewidth=1.2,
    )
    ax2.errorbar(
        median_rank_coeff,
        y_pos2,
        xerr=xerr_stability,
        fmt="o",
        color="black",
        ecolor="black",
        elinewidth=1.4,
        capsize=4.5,
        capthick=1.4,
        markersize=5.5,
    )
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(feat_names2, fontsize=16)
    ax2.set_xlabel("Median ranking coefficient", fontsize=18, fontweight="bold", labelpad=12)
    ax2.grid(axis="x", linestyle="--", alpha=0.5)
    ax2.set_xlim(0, 1.0)  

    
    for ax in (ax1, ax2):
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: "0" if np.isclose(x, 0) else f"{x:g}".replace("-", "\u2212"))
        )
        ax.tick_params(axis="x", direction="in", labelsize=15)
        ax.tick_params(axis="y", direction="in", labelsize=18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.3)
        ax.spines["bottom"].set_linewidth(1.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Dual-panel plot saved: {out_path.resolve()} (300 DPI)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Dual-Panel SHAP Importance and Ranking Stability Plot."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="SBP",
        choices=["SBP", "DBP", "MAP"],
        help="Target blood pressure: SBP, DBP, or MAP (default: SBP)",
    )
    parser.add_argument(
        "--summary_csv",
        "-s",
        default=None,
        help="Path to the _SHAP_feature_summary.csv file",
    )
    parser.add_argument(
        "--top_k",
        "-k",
        type=int,
        default=15,
        help="Number of top features to visualize per panel (default: 15)",
    )
    args = parser.parse_args()

    target = args.target.upper()
    n_features = 161

    # Locate input summary CSV
    summary_csv_path = (
        Path(args.summary_csv)
        if args.summary_csv
        else Path(
            f"{target}_{n_features}_features_SHAP_analysis_5fold_CV_SHAP_feature_summary.csv"
        )
    )

    if not summary_csv_path.exists():
        raise FileNotFoundError(
            f"SHAP summary CSV not found: {summary_csv_path.resolve()}"
        )

    print(f"Loading summary data: {summary_csv_path.resolve()}")
    df_summary = pd.read_csv(summary_csv_path)

    out_combined_path = Path(f"{target}_SHAP_top{args.top_k}_importance_and_stability.png")
    plot_combined_importance_and_stability(
        df_summary=df_summary,
        target=target,
        top_k=args.top_k,
        out_path=out_combined_path,
    )

    print("\nRanking stability plot generated successfully.")


if __name__ == "__main__":
    main()