import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


def main():
    parser = argparse.ArgumentParser(
        description="Two-Stage SHAP Feature Selection: Mean |SHAP| Screening + Silhouette-Optimized Clustering."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="SBP",
        choices=["SBP", "DBP", "MAP"],
        help="Target blood pressure: SBP, DBP, or MAP (default: SBP)",
    )
    parser.add_argument(
        "--matrix_csv",
        "-i",
        default=None,
        help="Path to the _feature_and_SHAP_matrix.csv file",
    )
    parser.add_argument(
        "--summary_csv",
        "-s",
        default=None,
        help="Path to the _SHAP_feature_summary.csv file",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Step size for threshold r search in [0.20, 0.80] (default: 0.05)",
    )
    args = parser.parse_args()

    target = args.target.upper()
    n_features = 161

    
    matrix_csv_path = (
        Path(args.matrix_csv)
        if args.matrix_csv
        else Path(
            f"{target}_{n_features}_features_SHAP_analysis_5fold_CV_feature_and_SHAP_matrix.csv"
        )
    )
    summary_csv_path = (
        Path(args.summary_csv)
        if args.summary_csv
        else Path(
            f"{target}_{n_features}_features_SHAP_analysis_5fold_CV_SHAP_feature_summary.csv"
        )
    )

    if not matrix_csv_path.exists():
        raise FileNotFoundError(
            f"SHAP matrix CSV not found: {matrix_csv_path.resolve()}"
        )

    print(f"Loading SHAP matrix: {matrix_csv_path.resolve()}")
    df_merged = pd.read_csv(matrix_csv_path)

    
    meta_cols_set = {
        "sample_index",
        "subject",
        "fold",
        "label",
        "target_delta",
        f"delta_{target.lower()}",
        f"delta_{target.upper()}",
        "y",
        "target",
        "y_true",
    }
    feature_cols = [c for c in df_merged.columns if c.lower() not in meta_cols_set]

    df_shap = (
        df_merged[df_merged["label"] == "shap_value"][feature_cols]
        .reset_index(drop=True)
        .astype(float)
    )
    print(f"Extracted SHAP matrix: {df_shap.shape[0]} samples x {df_shap.shape[1]} features.")


    corr_shap_full = df_shap.corr(method="pearson")
    dist_mat_full = np.clip(1.0 - corr_shap_full.values, 0.0, 2.0)
    np.fill_diagonal(dist_mat_full, 0.0)
    condensed_dist_full = squareform(dist_mat_full, checks=False)
    linkage_matrix_full = linkage(condensed_dist_full, method="average")

    out_clustermap_path = Path(f"{target}_SHAP_correlation_clustermap.png")
    g = sns.clustermap(
        corr_shap_full,
        row_linkage=linkage_matrix_full,
        col_linkage=linkage_matrix_full,
        cmap="vlag",
        vmin=-1.0,
        vmax=1.0,
        figsize=(16, 16),
        cbar_kws={"label": "SHAP Pearson Correlation (r)", "shrink": 0.8},
        dendrogram_ratio=(0.12, 0.12),
        xticklabels=True,
        yticklabels=True,
    )

    g.ax_heatmap.set_title(
        f"SHAP Value Correlation & Hierarchical Clustering for Δ{target} (161 Features)",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    g.ax_heatmap.tick_params(axis="x", labelsize=6.5, rotation=90)
    g.ax_heatmap.tick_params(axis="y", labelsize=6.5, rotation=0)

    plt.savefig(out_clustermap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Clustermap saved (Full 161 features): {out_clustermap_path.resolve()} (300 DPI)")

  
    if summary_csv_path.exists():
        df_summary = pd.read_csv(summary_csv_path)
        feat_info_dict = df_summary.set_index("feature_name").to_dict(orient="index")
        mean_abs_all = np.array(
            [
                feat_info_dict.get(c, {}).get("mean_abs_shap_5fold", np.mean(np.abs(df_shap[c].values)))
                for c in feature_cols
            ]
        )
    else:
        feat_info_dict = {}
        mean_abs_all = np.mean(np.abs(df_shap.values), axis=0)

    # =========================================================================
    # 【Stage 1】Eliminate features with Mean |SHAP| greater than or equal to the overall mean
    # =========================================================================
    mean_shap_threshold = np.mean(mean_abs_all)
    stage1_mask = mean_abs_all >= mean_shap_threshold
    stage1_features = [feature_cols[i] for i in range(len(feature_cols)) if stage1_mask[i]]
    stage1_shap_means = mean_abs_all[stage1_mask]

    df_stage1 = pd.DataFrame({
        "feature_name": stage1_features,
        "category": [feat_info_dict.get(c, {}).get("category", "Unknown") for c in stage1_features],
        "mean_abs_shap": stage1_shap_means,
        "median_rank_coefficient": [
            feat_info_dict.get(c, {}).get("median_rank_coefficient", np.nan) for c in stage1_features
        ],
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    out_stage1_csv = Path(f"{target}_SHAP_stage1_screened_features.csv")
    df_stage1.to_csv(out_stage1_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(f"STAGE 1: IMPORTANCE SCREENING COMPLETED (Target: Δ{target})")
    print("=" * 80)
    print(f"Mean |SHAP| Average Threshold : {mean_shap_threshold:.6f}")
    print(f"Retained Features in Stage 1  : {len(stage1_features)} / {len(feature_cols)} features")
    print(f"Stage 1 Screened CSV Saved    : {out_stage1_csv.resolve()}")

    # =========================================================================
    # 【Stage 2】Silhouette Coefficient Maximization Adaptive Optimization (0.20 ~ 0.80, step=0.05)
    # =========================================================================
    df_shap_s1 = df_shap[stage1_features]
    corr_shap_s1 = df_shap_s1.corr(method="pearson")
    dist_mat_s1 = np.clip(1.0 - corr_shap_s1.values, 0.0, 2.0)
    np.fill_diagonal(dist_mat_s1, 0.0)
    condensed_dist_s1 = squareform(dist_mat_s1, checks=False)
    linkage_matrix_s1 = linkage(condensed_dist_s1, method="average")

    r_thresholds = np.round(np.arange(0.20, 0.80 + 1e-5, args.step), 4)
    eval_records = []

    for r in r_thresholds:
        dist_cut = 1.0 - r
        clusters = fcluster(linkage_matrix_s1, t=dist_cut, criterion="distance")
        n_clusters = len(np.unique(clusters))

        # 计算轮廓系数 (需满足 2 <= 簇数 <= 样本数-1)
        if 1 < n_clusters < len(stage1_features):
            score = silhouette_score(dist_mat_s1, clusters, metric="precomputed")
        else:
            score = np.nan

        eval_records.append({
            "threshold_r": r,
            "distance_cutoff": dist_cut,
            "n_clusters": n_clusters,
            "silhouette_score": score,
        })

    df_eval = pd.DataFrame(eval_records).dropna().reset_index(drop=True)

    
    best_idx = df_eval["silhouette_score"].idxmax()
    best_r = float(df_eval.loc[best_idx, "threshold_r"])
    best_score = float(df_eval.loc[best_idx, "silhouette_score"])
    best_k = int(df_eval.loc[best_idx, "n_clusters"])

    print("\n" + "=" * 80)
    print(f"STAGE 2: SILHOUETTE OPTIMIZATION COMPLETED (Target: Δ{target})")
    print("=" * 80)
    print(f"Optimal Correlation Threshold (r*) : {best_r:.4f}")
    print(f"Max Mean Silhouette Score           : {best_score:.4f}")
    print(f"Optimal Number of Feature Clusters  : {best_k} features")

    # =========================================================================
    # 【Step 2】output stage 2 full feature clustering and representative features CSV
    # =========================================================================
    best_dist_cut = 1.0 - best_r
    final_clusters = fcluster(linkage_matrix_s1, t=best_dist_cut, criterion="distance")
    df_stage1["cluster_id"] = final_clusters


    df_stage2_rep = (
        df_stage1.sort_values(by=["cluster_id", "mean_abs_shap"], ascending=[True, False])
        .groupby("cluster_id")
        .first()
        .reset_index()
        .sort_values(by="mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


    rep_feature_set = set(df_stage2_rep["feature_name"])
    cluster_mapping = dict(zip(df_stage1["feature_name"], df_stage1["cluster_id"]))

    df_all_161_features = pd.DataFrame({
        "feature_name": feature_cols,
        "category": [feat_info_dict.get(c, {}).get("category", "Unknown") for c in feature_cols],
        "mean_abs_shap": mean_abs_all,
        "stage1_passed": [c in stage1_features for c in feature_cols],
        "cluster_id": [cluster_mapping.get(c, np.nan) for c in feature_cols],
        "is_stage2_representative": [c in rep_feature_set for c in feature_cols],
    }).sort_values(
        by=["stage1_passed", "cluster_id", "mean_abs_shap"],
        ascending=[False, True, False]
    ).reset_index(drop=True)


    out_stage2_csv = Path(f"{target}_SHAP_stage2_clusters_features.csv")
    with open(out_stage2_csv, "w", encoding="utf-8-sig", newline="") as f:
        
        df_all_161_features.to_csv(f, index=False)
        
        
        f.write("\n")
        f.write("# " + "=" * 78 + "\n")
        f.write(f"# STAGE 2 FINAL REPRESENTATIVE FEATURES ({len(df_stage2_rep)} SELECTED FEATURES)\n")
        f.write("# " + "=" * 78 + "\n")
        
        
        df_stage2_rep.to_csv(f, index=False)

    print(f"Stage 2 Full Clustering & Representative Features CSV : {out_stage2_csv.resolve()}")

    
    print("\nFinal Representative Features - Category Breakdown:")
    print(df_stage2_rep["category"].value_counts().to_string())

    # =========================================================================
    # 【Step 3】Plotting Silhouette Score vs Threshold r and Number of Features k
    # =========================================================================
    out_sil_png_path = Path(f"{target}_Clustering_threshold_Silhouette_Score.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

   
    ax1.plot(
        df_eval["threshold_r"],
        df_eval["silhouette_score"],
        marker="o",
        color="#1f77b4",
        linewidth=2.2,
        markersize=6,
        label="Mean Silhouette Score",
    )
    ax1.axvline(
        best_r,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Optimal r = {best_r:.2f} ({best_k} feats)",
    )
    ax1.scatter(
        [best_r],
        [best_score],
        color="#d62728",
        s=140,
        zorder=5,
        marker="*",
    )
    ax1.set_title(
        f"(a) Correlation Threshold (r) vs Mean Silhouette Score (Δ{target})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xlabel("Correlation Threshold (r)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Silhouette Score", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="best", frameon=True)

    
    for idx in range(len(df_eval)):
        row = df_eval.iloc[idx]
        ax1.annotate(
            f"{int(row['n_clusters'])}f",
            (row["threshold_r"], row["silhouette_score"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7.5,
            color="#333333",
        )

    
    df_sorted_k = df_eval.sort_values(by="n_clusters").reset_index(drop=True)
    ax2.plot(
        df_sorted_k["n_clusters"],
        df_sorted_k["silhouette_score"],
        marker="s",
        color="#2ca02c",
        linewidth=2.2,
        markersize=6,
        label="Cluster Validity Curve",
    )
    ax2.axvline(
        best_k,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Optimal k = {best_k} (Score: {best_score:.3f})",
    )
    ax2.scatter(
        [best_k],
        [best_score],
        color="#d62728",
        s=140,
        zorder=5,
        marker="*",
    )
    ax2.set_title(
        f"(b) Retained Feature Count (k) vs Mean Silhouette Score (Δ{target})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax2.set_xlabel("Number of Feature Clusters (k)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Silhouette Score", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="best", frameon=True)

    plt.suptitle(
        f"Internal Cluster Validity: Silhouette Score Maximization for Δ{target} Feature Selection",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_sil_png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Silhouette Curve Saved        : {out_sil_png_path.resolve()} (300 DPI)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()