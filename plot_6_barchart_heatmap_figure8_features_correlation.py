import argparse
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

# ---------------------------------------------------------
# Global Font Configuration: Arial (Publication Quality)
# ---------------------------------------------------------
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "sans-serif"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

DEMOGRAPHIC_FEATURES = [
    "Height",
    "Weight",
    "BMI",
    "Gender",
    "Age",
    "height",
    "weight",
    "bmi",
    "gender",
    "age",
]

CATEGORY_ORDER = [
    "Time",
    "Amplitude",
    "VPG_APG",
    "VPG_APG_Ratio",
    "Area",
    "Power",
    "Ratio",
    "Slope",
    "Width",
    "Perfusion",
    "Contour",
    "Demographics",
]


def load_feature_matrix(matrix_csv_path: Path) -> pd.DataFrame:
    """Load feature matrix.

    Supports both raw all_trimmed.csv (extracts 156 + 5 = 161 features)
    and preprocessed *_feature_and_SHAP_matrix.csv.
    """
    if not matrix_csv_path.exists():
        raise FileNotFoundError(f"File not found: {matrix_csv_path.resolve()}")

    df_raw = pd.read_csv(matrix_csv_path)

    # 1. Direct handling for all_trimmed.csv format (>= 166 columns)
    if "all_trimmed" in matrix_csv_path.name.lower() or df_raw.shape[1] >= 166:
        df_feat_156 = df_raw.iloc[:, 10:166].copy()
        df_demo_5 = df_raw.iloc[:, 5:10].copy()

        if any("unnamed" in str(c).lower() for c in df_demo_5.columns):
            df_demo_5.columns = ["Height", "Weight", "BMI", "Gender", "Age"]

        if df_demo_5.isna().any().any():
            df_demo_5 = df_demo_5.ffill()

        df_feat = pd.concat([df_feat_156, df_demo_5], axis=1)
        df_feat = df_feat.reset_index(drop=True).astype(float)
        return df_feat

    # 2. Handling for pre-merged SHAP/Feature matrix
    meta_cols_set = {
        "sample_index",
        "subject",
        "subject_id",
        "fold",
        "label",
        "target_delta",
        "delta_sbp",
        "delta_dbp",
        "delta_map",
        "delta_SBP",
        "delta_DBP",
        "delta_MAP",
        "target",
        "y_true",
        "y_pred",
        "y_true_delta_bp",
        "y_pred_delta_bp",
        "unnamed: 0",
    }

    meta_cols_found = [
        c for c in df_raw.columns if str(c).strip().lower() in meta_cols_set
    ]
    feature_cols = [c for c in df_raw.columns if c not in meta_cols_found]

    if "label" in df_raw.columns:
        df_feat = df_raw[df_raw["label"] == "feature_delta"][feature_cols]
        if df_feat.empty:
            df_feat = df_raw[df_raw["label"] != "shap_value"][feature_cols]
    else:
        df_feat = df_raw[feature_cols]

    df_feat = df_feat.reset_index(drop=True).astype(float)
    return df_feat


def load_union_features(summary_csv_path: Path) -> list[str]:
    """Extract selected union features from summary CSV."""
    if not summary_csv_path.exists():
        raise FileNotFoundError(
            f"Summary CSV file not found: {summary_csv_path.resolve()}"
        )

    df_summary = pd.read_csv(summary_csv_path, comment="#")
    if "feature_name" not in df_summary.columns:
        raise ValueError("The summary CSV must contain a 'feature_name' column.")

    return df_summary["feature_name"].dropna().astype(str).tolist()


# --- [NEW] Load Target-Specific Features ---
def load_target_specific_features(csv_path: Path) -> list[str]:
    """
    Load target-specific features from the specified CSV file.
    Extracts features listed below 'STAGE 2 FINAL REPRESENTATIVE FEATURES'.
    """
    features = []
    start_reading = False
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "STAGE 2 FINAL REPRESENTATIVE FEATURES" in line.upper():
                    start_reading = True
                    continue
                
                if start_reading and not line.startswith('#'):
                    # Skip standard CSV header rows if they exist
                    if "cluster_id" in line.lower() or "feature_name" in line.lower():
                        continue
                    parts = line.split(',')
                    # Assuming CSV format: cluster_id, feature_name, ...
                    if len(parts) >= 2 and parts[1].strip():
                        feat_name = parts[1].strip().replace('"', '').replace("'", "")
                        features.append(feat_name)
                    # Fallback if it's a single column
                    elif len(parts) == 1 and parts[0].strip():
                        feat_name = parts[0].strip().replace('"', '').replace("'", "")
                        features.append(feat_name)
    except Exception as e:
        print(f"[Error] Failed to read {csv_path}: {e}")
        
    return features
# -------------------------------------------


def get_feature_category(feat_name: str, feat_idx_1based: int) -> str:
    """Classify feature category based on 1-based index and demographics."""
    if str(feat_name).strip().lower() in [
        d.lower() for d in DEMOGRAPHIC_FEATURES
    ]:
        return "Demographics"

    if 1 <= feat_idx_1based <= 24:
        return "Time"
    elif 25 <= feat_idx_1based <= 39:
        return "Amplitude"
    elif 40 <= feat_idx_1based <= 49:
        return "VPG_APG"
    elif 50 <= feat_idx_1based <= 62:
        return "VPG_APG_Ratio"
    elif 63 <= feat_idx_1based <= 71:
        return "Area"
    elif 72 <= feat_idx_1based <= 86:
        return "Power"
    elif 87 <= feat_idx_1based <= 119:
        return "Ratio"
    elif 120 <= feat_idx_1based <= 135:
        return "Slope"
    elif 136 <= feat_idx_1based <= 153:
        return "Width"
    elif feat_idx_1based == 154:
        return "Perfusion"
    elif 155 <= feat_idx_1based <= 156:
        return "Contour"
    elif 157 <= feat_idx_1based <= 161:
        return "Demographics"
    else:
        return "Demographics"


def compute_pairwise_pearson(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise Absolute Pearson Correlation |rho_p| and p-values."""
    cols = df.columns.tolist()
    n = len(cols)
    X = df.values

    corr_mat = np.zeros((n, n), dtype=float)
    pval_mat = np.zeros((n, n), dtype=float)

    for i in range(n):
        corr_mat[i, i] = 1.0
        pval_mat[i, i] = 0.0
        for j in range(i + 1, n):
            r, p = pearsonr(X[:, i], X[:, j])
            if np.isnan(r):
                r, p = 0.0, 1.0
            abs_r = abs(r)
            corr_mat[i, j] = corr_mat[j, i] = abs_r
            pval_mat[i, j] = pval_mat[j, i] = p

    df_corr = pd.DataFrame(corr_mat, index=cols, columns=cols)
    df_pval = pd.DataFrame(pval_mat, index=cols, columns=cols)
    return df_corr, df_pval


def calculate_collinearity_metrics(
    df_corr: pd.DataFrame, threshold: float = 0.8
) -> dict:
    """Calculate collinearity statistics (|rho| > threshold)."""
    n = df_corr.shape[0]
    mat = df_corr.values
    total_pairs = n * (n - 1) // 2

    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    strong_pairs = int(np.sum((mat > threshold) & upper_tri))
    p_strong = (strong_pairs / total_pairs) * 100.0 if total_pairs > 0 else 0.0

    mat_offdiag = mat.copy()
    np.fill_diagonal(mat_offdiag, 0.0)
    has_strong_feature = np.any(mat_offdiag > threshold, axis=1)
    n_features_with_strong = int(np.sum(has_strong_feature))
    p_features_with_strong = (n_features_with_strong / n) * 100.0

    return {
        "n_features": n,
        "total_pairs": total_pairs,
        "n_strong": strong_pairs,
        "p_strong": p_strong,
        "n_strong_features": n_features_with_strong,
        "p_strong_features": p_features_with_strong,
    }


def compute_pairwise_mutual_information(
    df: pd.DataFrame,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute pairwise raw Mutual Information (MI) using KSG estimator."""
    n_cols = df.shape[1]
    cols = df.columns.tolist()
    X_vals = df.values
    mi_matrix = np.zeros((n_cols, n_cols), dtype=float)

    for i in range(n_cols):
        y_i = X_vals[:, i]
        mi_vals = mutual_info_regression(
            X_vals,
            y_i,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        mi_matrix[i, :] = mi_vals

    mi_matrix_sym = (mi_matrix + mi_matrix.T) / 2.0
    return pd.DataFrame(mi_matrix_sym, index=cols, columns=cols)


def calculate_information_coverage(
    df_mi: pd.DataFrame,
    union_features: list[str],
    mi_threshold: float = 0.5,
) -> dict:
    """Calculate Mutual Information Coverage for removed features."""
    all_features = df_mi.columns.tolist()
    removed_features = [f for f in all_features if f not in union_features]

    if not removed_features:
        return {
            "n_removed": 0,
            "n_covered": 0,
            "coverage": 100.0,
            "mean_max_mi": 0.0,
            "median_max_mi": 0.0,
            "max_mi_per_removed": pd.Series([], dtype=float),
        }

    sub_mi = df_mi.loc[removed_features, union_features]
    max_mi_per_removed = sub_mi.max(axis=1)

    covered_mask = max_mi_per_removed > mi_threshold
    n_covered = int(np.sum(covered_mask))
    n_removed = len(removed_features)
    coverage_pct = (n_covered / n_removed) * 100.0

    return {
        "n_removed": n_removed,
        "n_covered": n_covered,
        "coverage": coverage_pct,
        "mean_max_mi": float(max_mi_per_removed.mean()),
        "median_max_mi": float(max_mi_per_removed.median()),
        "max_mi_per_removed": max_mi_per_removed,
    }


def plot_full_161_features_heatmap(
    df_corr: pd.DataFrame,
    df_pval: pd.DataFrame,
    out_png: Path,
) -> None:
    """Plot Full 161 Features Pearson correlation matrix."""
    features = df_corr.columns.tolist()
    feat_to_cat = {
        f: get_feature_category(f, idx + 1) for idx, f in enumerate(features)
    }

    sorted_features = []
    cat_boundaries = [0]
    cat_names = []

    for cat in CATEGORY_ORDER:
        cat_feats = [f for f in features if feat_to_cat[f] == cat]
        if len(cat_feats) > 0:
            sorted_features.extend(cat_feats)
            cat_boundaries.append(len(sorted_features))
            cat_names.append(cat)

    n = len(sorted_features)
    rev_sorted = sorted_features[::-1]
    plot_corr = df_corr.loc[sorted_features, rev_sorted].copy()
    plot_pval = df_pval.loc[sorted_features, rev_sorted].copy()

    mask = np.zeros((n, n), dtype=bool)
    for r in range(n):
        for c in range(n):
            if c > (n - 1 - r):
                mask[r, c] = True
            elif plot_corr.iloc[r, c] < 0.2 or plot_pval.iloc[r, c] > 0.05:
                mask[r, c] = True

    cat_midpoints = [
        (cat_boundaries[i] + cat_boundaries[i + 1]) / 2.0
        for i in range(len(cat_boundaries) - 1)
    ]

    fig, ax = plt.subplots(figsize=(14, 12), dpi=300)

    sns.heatmap(
        plot_corr,
        mask=mask,
        cmap="YlOrRd",
        vmin=0.1,
        vmax=1.0,
        square=True,
        cbar=True,
        cbar_kws={"shrink": 0.75, "pad": 0.03},
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )

    for k in range(n + 1):
        ax.axhline(k, color="gray", linewidth=0.35, alpha=0.45)
        ax.axvline(k, color="gray", linewidth=0.35, alpha=0.45)

    for b in cat_boundaries[1:-1]:
        ax.hlines(y=b, xmin=0, xmax=n - b, color="black", linewidth=1.0)
        ax.vlines(x=n - b, ymin=0, ymax=b, color="black", linewidth=1.0)

    ax.set_yticks(cat_midpoints)
    ax.set_yticklabels(
        cat_names, fontsize=11, fontweight="bold", fontname="Arial"
    )
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xticks([])

    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(
            r"$|\rho_p|$", fontsize=13, fontname="Arial", fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_filtered_41_features_heatmap(
    df_corr: pd.DataFrame,
    df_pval: pd.DataFrame,
    out_png: Path,
) -> None:
    """Plot 41 features Pearson correlation matrix."""
    features = df_corr.columns.tolist()
    n = len(features)

    rev_features = features[::-1]
    plot_corr = df_corr.loc[features, rev_features].copy()
    plot_pval = df_pval.loc[features, rev_features].copy()

    mask = np.zeros((n, n), dtype=bool)
    for r in range(n):
        for c in range(n):
            if c > (n - 1 - r):
                mask[r, c] = True
            elif plot_corr.iloc[r, c] < 0.2 or plot_pval.iloc[r, c] > 0.05:
                mask[r, c] = True

    fig, ax = plt.subplots(figsize=(15, 13), dpi=300)

    sns.heatmap(
        plot_corr,
        mask=mask,
        cmap="YlOrRd",
        vmin=0.1,
        vmax=1.0,
        square=True,
        cbar=True,
        cbar_kws={"shrink": 0.75, "pad": 0.03},
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )

    for k in range(n + 1):
        ax.axhline(k, color="gray", linewidth=0.35, alpha=0.45)
        ax.axvline(k, color="gray", linewidth=0.35, alpha=0.45)

    y_ticks = [i + 0.5 for i in range(n) if i % 2 == 0]
    y_labels = [features[i] for i in range(n) if i % 2 == 0]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8, fontname="Arial")

    x_ticks = [j + 0.5 for j in range(n) if (n - 1 - j) % 2 != 0]
    x_labels = [rev_features[j] for j in range(n) if (n - 1 - j) % 2 != 0]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=8, fontname="Arial", rotation=90)

    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(
            r"$|\rho_p|$", fontsize=13, fontname="Arial", fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_mi_coverage_distribution(
    max_mi_series: pd.Series,
    mi_threshold: float,
    coverage_pct: float,
    out_png: Path,
    target_name: str = "",
) -> None:
    """Plot mutual information coverage histogram (Target-specific)"""
    if max_mi_series.empty:
        return

    mean_val = float(max_mi_series.mean())
    median_val = float(max_mi_series.median())

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 1. Histogram without KDE, count stat
    sns.histplot(
        max_mi_series,
        kde=False,
        stat="count",
        color="#1f77b4",
        binwidth=0.25,
        binrange=(0, 5),
        alpha=0.55,
        edgecolor="white",
        linewidth=1.2,
        ax=ax,
    )

    # 2. Vertical reference lines with 2 decimal places and formatted labels
    ax.axvline(
        mi_threshold,
        color="#d62728",
        linestyle="--",
        linewidth=2.5,
        label=f"Threshold = {mi_threshold:.2f} nats",
    )
    ax.axvline(
        mean_val,
        color="#2ca02c",
        linestyle="-.",
        linewidth=2.3,
        label=f"Mean Max-MI = {mean_val:.2f} nats",
    )
    ax.axvline(
        median_val,
        color="#ff7f0e",
        linestyle=":",
        linewidth=2.3,
        label=f"Median Max-MI = {median_val:.2f} nats",
    )

    # 3. Axis labels with increased font sizes
    ax.set_xlabel(
        "Maximum Mutual Information (nats)",
        fontsize=20,
        fontweight="bold",
        fontname="Arial",
    )
    ax.set_ylabel(
        "Feature number",
        fontsize=20,
        fontweight="bold",
        fontname="Arial",
    )

    # 4. Inward ticks on both axes with larger font size
    ax.tick_params(axis="both", direction="in", labelsize=20)

    # 5. Dynamic y-ticks based on actual distribution to handle varying removed feature counts
    ax.set_xlim(0, 5)
    ax.set_xticks(np.arange(0, 6, 1))
    
    # Dynamically scale Y-axis based on histogram max bin count
    if target_name == "SBP":
        ax.set_ylim(0, 80)
        ax.set_yticks(np.arange(0, 81, 10))
    else:
        max_count = int(np.histogram(max_mi_series, bins=np.arange(0, 5.25, 0.25))[0].max())
        y_upper = ((max_count // 5) + 2) * 5
        ax.set_ylim(0, y_upper)
        ax.set_yticks(np.arange(0, y_upper + 1, 5))
    
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: "" if np.isclose(y, 0) else f"{int(y)}")
    )

    ax.grid(True, linestyle="--", alpha=0.5)

    # 6. Legend with increased font size
    ax.legend(
        frameon=True,
        facecolor="none",
        edgecolor="none",
        fontsize=20,
        loc="upper right",
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_mi_coverage_curve(
    max_mi_series: pd.Series,
    out_png: Path,
    threshold_min: float = 0.2,
    threshold_max: float = 0.8,
    num_points: int = 13,
    highlight_threshold: float = 0.5,
) -> None:
    """Plot Stage2_mutual_information_coverage_2.png:

    Line plot showing Information Coverage (%) across thresholds [0.2, 0.8].
    """
    if max_mi_series.empty:
        return

    thresholds = np.linspace(threshold_min, threshold_max, num_points)
    n_removed = len(max_mi_series)
    coverages = [
        float(np.sum(max_mi_series > t) / n_removed * 100.0) for t in thresholds
    ]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    ax.plot(
        thresholds,
        coverages,
        color="#1f77b4",
        marker="o",
        markersize=7,
        linewidth=2.2,
        markeredgecolor="white",
        markeredgewidth=1.2,
        label="Information Coverage Curve",
    )

    highlight_cov = float(
        np.sum(max_mi_series > highlight_threshold) / n_removed * 100.0
    )
    ax.scatter(
        [highlight_threshold],
        [highlight_cov],
        color="#d62728",
        s=100,
        zorder=5,
        label=f"Selected Threshold ({highlight_threshold:.2f} nats: {highlight_cov:.2f}%)",
    )
    ax.annotate(
        f"{highlight_cov:.1f}% @ {highlight_threshold:.2f} nats",
        xy=(highlight_threshold, highlight_cov),
        xytext=(
            highlight_threshold + 0.03,
            (
                highlight_cov - 6
                if highlight_cov > 80
                else highlight_cov + 6
            ),
        ),
        arrowprops=dict(
            facecolor="#d62728", shrink=0.08, width=1.2, headwidth=6
        ),
        fontname="Arial",
        fontweight="bold",
        fontsize=18,
        color="#d62728",
    )

    ax.set_title(
        "Information Coverage vs. Mutual Information Threshold",
        fontsize=18,
        fontweight="bold",
        fontname="Arial",
        pad=12,
    )
    ax.set_xlabel(
        "Mutual Information Threshold (nats)",
        fontsize=18,
        fontweight="bold",
        fontname="Arial",
    )
    ax.set_ylabel(
        "Information Coverage (%)",
        fontsize=18,
        fontweight="bold",
        fontname="Arial",
    )
    ax.set_xlim(threshold_min - 0.03, threshold_max + 0.03)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 110, 10))
    ax.set_yticklabels(
        [f"{y}%" for y in np.arange(0, 110, 10)], fontname="Arial", fontsize=18
    )
    ax.set_xticks(np.arange(threshold_min, threshold_max + 0.05, 0.1))
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(
        frameon=True,
        facecolor="none",
        edgecolor="none",
        fontsize=18,
        loc="lower left",
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Pairwise Pearson Correlation Heatmaps, Collinearity"
        " Statistics, and MI Coverage."
    )
    parser.add_argument(
        "--matrix_csv",
        "-i",
        default=None,
        help="Path to all_trimmed.csv or *_feature_and_SHAP_matrix.csv.",
    )
    parser.add_argument(
        "--summary_csv",
        "-s",
        default="Stage2_overlapping_features_summary.csv",
        help="Path to Stage2_overlapping_features_summary.csv.",
    )
    parser.add_argument(
        "--output_prefix",
        "-o",
        default="Stage2_41_features",
        help="Prefix for output files (default: Stage2_41_features).",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.8,
        help="Threshold for strong Pearson correlation (default: 0.8).",
    )
    parser.add_argument(
        "--mi_threshold",
        type=float,
        default=0.5,
        help=(
            "Threshold for strong Mutual Information coverage in nats (default:"
            " 0.5)."
        ),
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=3,
        help=(
            "Number of nearest neighbors for mutual information estimation"
            " (default: 3)."
        ),
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for KSG estimation (default: 42).",
    )
    args = parser.parse_args()

    # 1. Locate matrix CSV
    if args.matrix_csv:
        matrix_path = Path(args.matrix_csv)
    else:
        default_candidates = [
            Path("all_trimmed.csv"),
            Path(
                "SBP_161_features_SHAP_analysis_5fold_CV_Finnegan_feature_and_SHAP_matrix.csv"
            ),
            Path(
                "DBP_161_features_SHAP_analysis_5fold_CV_Finnegan_feature_and_SHAP_matrix.csv"
            ),
            Path(
                "MAP_161_features_SHAP_analysis_5fold_CV_Finnegan_feature_and_SHAP_matrix.csv"
            ),
        ]
        matrix_path = next((p for p in default_candidates if p.exists()), None)
        if matrix_path is None:
            found_files = glob.glob("*feature_and_SHAP_matrix.csv") + glob.glob(
                "*all_trimmed.csv"
            )
            if found_files:
                matrix_path = Path(found_files[0])
            else:
                raise FileNotFoundError("No input feature CSV file found.")

    # 2. Load feature matrices (N = 161)
    df_feat_all = load_feature_matrix(matrix_path)
    union_features = load_union_features(Path(args.summary_csv))

    missing_feats = [f for f in union_features if f not in df_feat_all.columns]
    if missing_feats:
        print(
            "[Warning] Exact feature names not matched:"
            f" {missing_feats}. Proceeding with intersection."
        )
        valid_union = [f for f in union_features if f in df_feat_all.columns]
        if not valid_union:
            raise KeyError(
                "None of the union features were found in data columns:"
                f" {df_feat_all.columns.tolist()[:10]}..."
            )
        df_feat_41 = df_feat_all[valid_union].copy()
    else:
        df_feat_41 = df_feat_all[union_features].copy()

    # 3. Compute Pearson correlation and p-values
    corr_all, pval_all = compute_pairwise_pearson(df_feat_all)
    corr_41, pval_41 = compute_pairwise_pearson(df_feat_41)

    # 4. Compute Collinearity Metrics
    m_all = calculate_collinearity_metrics(corr_all, threshold=args.threshold)
    m_41 = calculate_collinearity_metrics(corr_41, threshold=args.threshold)

    # 5. Compute Mutual Information Base Matrix
    print("\nComputing pairwise Mutual Information matrix (KSG Estimator)...")
    df_mi_all = compute_pairwise_mutual_information(
        df_feat_all,
        n_neighbors=args.n_neighbors,
        random_state=args.random_state,
    )

    # 6. Print Base Summary
    print("\n" + "=" * 72)
    print(
        "PEARSON CORRELATION COLLINEARITY ANALYSIS (|rho| >"
        f" {args.threshold:.2f})"
    )
    print("=" * 72)
    print(
        f"[Full Features] (N = {m_all['n_features']}, Total Pairs ="
        f" {m_all['total_pairs']})"
    )
    print(f"  - N_strong     = {m_all['n_strong']}")
    print(f"  - P_strong     = {m_all['p_strong']:.2f}%")
    print(
        "  - Features with >= 1 Strong Collinearity:"
        f" {m_all['n_strong_features']} / {m_all['n_features']}"
        f" ({m_all['p_strong_features']:.2f}%)"
    )
    print("-" * 72)
    print(
        f"[Selected 41 Features] (N = {m_41['n_features']}, Total Pairs ="
        f" {m_41['total_pairs']})"
    )
    print(f"  - N_strong,41  = {m_41['n_strong']}")
    print(f"  - P_strong,41  = {m_41['p_strong']:.2f}%")
    print(
        "  - Features with >= 1 Strong Collinearity:"
        f" {m_41['n_strong_features']} / {m_41['n_features']}"
        f" ({m_41['p_strong_features']:.2f}%)"
    )
    print("-" * 72)
    print("[Redundancy Reduction Summary]")
    pair_red = m_all["p_strong"] - m_41["p_strong"]
    feat_red = m_all["p_strong_features"] - m_41["p_strong_features"]
    print(
        f"  - Collinear Pairs Ratio (P_strong): {m_all['p_strong']:.2f}% ->"
        f" {m_41['p_strong']:.2f}% (Delta: -{pair_red:.2f}%)"
    )
    print(
        "  - Collinear Features Ratio:       "
        f" {m_all['p_strong_features']:.2f}% -> {m_41['p_strong_features']:.2f}%"
        f" (Delta: -{feat_red:.2f}%)"
    )

    # 7. Generate and save PNG Heatmaps
    out_all_png = Path(
        f"Stage2_{m_all['n_features']}_features_Pearson_correlation_heatmap.png"
    )
    plot_full_161_features_heatmap(corr_all, pval_all, out_all_png)

    out_41_png = Path(f"{args.output_prefix}_correlation_heatmap.png")
    plot_filtered_41_features_heatmap(corr_41, pval_41, out_41_png)

    # 8. [NEW] Target-Specific Mutual Information Coverage Plots
    print("\n" + "=" * 72)
    print(
        "TARGET-SPECIFIC MUTUAL INFORMATION COVERAGE ANALYSIS (Threshold >"
        f" {args.mi_threshold:.2f} nats)"
    )
    print("=" * 72)
    
    data_dir = Path("output_data_used")
    targets = ["SBP", "DBP", "MAP"]
    
    for target in targets:
        target_csv_path = data_dir / f"{target}_SHAP_stage2_clusters_features.csv"
        
        if not target_csv_path.exists():
            print(f"[Warning] Target specific CSV not found for {target}: {target_csv_path}")
            continue
            
        target_features = load_target_specific_features(target_csv_path)
        valid_target_features = [f for f in target_features if f in df_feat_all.columns]
        
        if not valid_target_features:
            print(f"[Warning] No valid features matched for {target}. Skipping MI plot.")
            continue
            
        cov_stats_target = calculate_information_coverage(
            df_mi=df_mi_all,
            union_features=valid_target_features,
            mi_threshold=args.mi_threshold,
        )
        
        print(f"[{target} Analysis]")
        print(f"  - Target-Specific Features Retained:  {len(valid_target_features)}")
        print(f"  - Removed Features Count:             {cov_stats_target['n_removed']}")
        print(
            "  - Covered by >= 1 Selected Feature:  "
            f" {cov_stats_target['n_covered']} / {cov_stats_target['n_removed']}"
        )
        print(
            "  - Information Coverage:              "
            f" coverage = {cov_stats_target['coverage']:.2f}%"
        )
        print(
            "  - Mean / Median Max-MI of Removed:   "
            f" {cov_stats_target['mean_max_mi']:.4f} nats /"
            f" {cov_stats_target['median_max_mi']:.4f} nats"
        )
        print("-" * 72)
        
        # Plot corresponding histogram
        out_target_png = Path(f"figure8_histogram_mutual_information_coverage_{target}.png")
        plot_mi_coverage_distribution(
            max_mi_series=cov_stats_target["max_mi_per_removed"],
            mi_threshold=args.mi_threshold,
            coverage_pct=cov_stats_target["coverage"],
            out_png=out_target_png,
            target_name=target,
        )

    # 9. (Optional) Keep the overall MI Curve based on the 41 union features
    cov_stats_union = calculate_information_coverage(
        df_mi=df_mi_all,
        union_features=(
            union_features if not missing_feats else df_feat_41.columns.tolist()
        ),
        mi_threshold=args.mi_threshold,
    )
    out_mi_curve_png = Path("Stage2_mutual_information_coverage_2.png")
    plot_mi_coverage_curve(
        max_mi_series=cov_stats_union["max_mi_per_removed"],
        out_png=out_mi_curve_png,
        threshold_min=0.2,
        threshold_max=0.8,
        num_points=13,
        highlight_threshold=args.mi_threshold,
    )


if __name__ == "__main__":
    main()