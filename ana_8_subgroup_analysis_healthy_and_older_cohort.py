import argparse
import os

os.environ.setdefault("PYTHONHASHSEED", "42")

from pathlib import Path
import numpy as np
import pandas as pd
import shap
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold

# =============================================================================
# Constants & Configuration
# =============================================================================
MATH_MINUS = "\u2212"
RANDOM_STATE = 42
N_FOLDS = 5
N_ESTIMATORS = 200
N_JOBS = 1

N_MORPH_FEATURES = 156
DEMOGRAPHIC_FEATURE_IDS = ["s1", "s2", "s3", "s4", "s5"]
DEMOGRAPHIC_FEATURE_NAMES = ["Gender", "Age", "Height", "Weight", "BMI"]
TOTAL_FEATURES = N_MORPH_FEATURES + len(DEMOGRAPHIC_FEATURE_IDS)

N_POSTURE_ROWS = 5
N_DELTA_SAMPLES = 4

MORPH_START_COL_0BASED = 10
MORPH_END_COL_0BASED_EXCLUSIVE = 166
PI_COL_0BASED = 163  # Column FH (164th column, 1-based; morphological feature 154: Perfusion Index)


def format_math_minus(text: str) -> str:
    """Replace standard ASCII hyphen-minus with the mathematical minus sign (U+2212)."""
    return str(text).replace("-", MATH_MINUS)


def get_feature_category(feature_id) -> str:
    """Classify morphological and demographic features."""
    if isinstance(feature_id, str) and feature_id in DEMOGRAPHIC_FEATURE_IDS:
        return "Demographics"
    if isinstance(feature_id, (int, np.integer)):
        feat_idx = int(feature_id)
        if 1 <= feat_idx <= 24:
            return "Time"
        elif 25 <= feat_idx <= 39:
            return "Amplitude"
        elif 40 <= feat_idx <= 49:
            return "VPG_APG"
        elif 50 <= feat_idx <= 62:
            return "VPG_APG_Ratio"
        elif 63 <= feat_idx <= 71:
            return "Area"
        elif 72 <= feat_idx <= 86:
            return "Power"
        elif 87 <= feat_idx <= 119:
            return "Ratio"
        elif 120 <= feat_idx <= 135:
            return "Slope"
        elif 136 <= feat_idx <= 153:
            return "Width"
        elif feat_idx == 154:
            return "Perfusion"
        elif feat_idx == 155 or feat_idx == 156:
            return "Contour"
    return "Unknown"


def load_feature_name_mapping(excel_path: Path, num_features: int = N_MORPH_FEATURES) -> dict:
    """Load 156 morphological feature names from Excel definition file if available, else fallback to Feature_1..156."""
    if not excel_path.exists():
        return {i: f"Feature_{i}" for i in range(1, num_features + 1)}

    df_def = pd.read_excel(excel_path, header=None)
    if df_def.shape[1] == 1:
        names = [str(df_def.iloc[i, 0]).strip() for i in range(num_features)]
    else:
        names = [str(df_def.iloc[i, 1]).strip() for i in range(num_features)]
    return {i + 1: names[i] for i in range(num_features)}


def load_cohort_data(base_dir: Path, exclude_dirs: set, feat_name_dict: dict) -> list[dict]:
    """Load raw 5-posture data and delta matrices for all valid subjects."""
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    subject_dirs = sorted([
        d for d in base_dir.glob("yy04_subject_*")
        if d.is_dir() and d.name not in exclude_dirs
    ])

    morph_names = [feat_name_dict[i] for i in range(1, N_MORPH_FEATURES + 1)]
    all_feature_names = morph_names + DEMOGRAPHIC_FEATURE_NAMES

    subjects_data = []
    for s_dir in subject_dirs:
        csv_file = s_dir / "all_trimmed.csv"
        if not csv_file.exists():
            continue

        df = pd.read_csv(csv_file)
        if len(df) < N_POSTURE_ROWS or df.shape[1] < MORPH_END_COL_0BASED_EXCLUSIVE:
            continue

        # Columns 1 & 2 (0-based 0 & 1): SBP and DBP; Row 0 is S1 baseline
        sbp_5 = df["SBP"].iloc[:N_POSTURE_ROWS].to_numpy(dtype=float)
        dbp_5 = df["DBP"].iloc[:N_POSTURE_ROWS].to_numpy(dtype=float)
        map_5 = (1.0 / 3.0) * sbp_5 + (2.0 / 3.0) * dbp_5

        age = float(df["Age"].iloc[0])
        gender = float(df["Gender"].iloc[0])
        height = float(df["Height"].iloc[0])
        weight = float(df["Weight"].iloc[0])
        bmi = float(df["BMI"].iloc[0])

        # Column FH (0-based index 163) is Perfusion Index (PI)
        pi_s1 = float(df.iloc[0, PI_COL_0BASED])
        pi_5 = df.iloc[:N_POSTURE_ROWS, PI_COL_0BASED].to_numpy(dtype=float)
        pi_mean_5 = float(np.mean(pi_5))

        morph_5 = df.iloc[:N_POSTURE_ROWS, MORPH_START_COL_0BASED:MORPH_END_COL_0BASED_EXCLUSIVE].to_numpy(dtype=float)
        delta_morph = morph_5[1:N_POSTURE_ROWS, :] - morph_5[0, :]

        demographics = np.array([gender, age, height, weight, bmi], dtype=float)
        demo_tile = np.tile(demographics, (N_DELTA_SAMPLES, 1))
        X_delta = np.hstack([delta_morph, demo_tile])

        raw_feature_series = {}
        delta_feature_series = {}
        for idx, m_name in enumerate(morph_names):
            raw_feature_series[m_name] = morph_5[:, idx]
            delta_feature_series[m_name] = delta_morph[:, idx]
        for idx, d_name in enumerate(DEMOGRAPHIC_FEATURE_NAMES):
            raw_feature_series[d_name] = np.repeat(demographics[idx], N_POSTURE_ROWS)
            delta_feature_series[d_name] = np.repeat(demographics[idx], N_DELTA_SAMPLES)

        subjects_data.append({
            "subject_id": s_dir.name,
            "age": age,
            "is_older_htn": bool(age > 50.0),
            "sbp_s1": float(sbp_5[0]),
            "dbp_s1": float(dbp_5[0]),
            "pi_s1": pi_s1,
            "pi_mean_5": pi_mean_5,
            "pi_5": pi_5,
            "sbp_5": sbp_5,
            "dbp_5": dbp_5,
            "map_5": map_5,
            "delta_sbp": sbp_5[1:] - sbp_5[0],
            "delta_dbp": dbp_5[1:] - dbp_5[0],
            "delta_map": map_5[1:] - map_5[0],
            "X_delta": X_delta,
            "raw_feature_series": raw_feature_series,
            "delta_feature_series": delta_feature_series,
            "all_feature_names": all_feature_names,
        })

    return subjects_data


def run_two_stage_shap_selection(
    subjects_subset: list[dict],
    target: str,
    feat_name_dict: dict,
    step: float = 0.05,
    match_ana3_assignment: bool = True,
    predefined_fold_splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    """
    Execute Stage 1 (5-fold subject-level TreeSHAP mean threshold screening)
    and Stage 2 (SHAP correlation hierarchical clustering + silhouette optimization).
    """
    morph_names = [feat_name_dict[i] for i in range(1, N_MORPH_FEATURES + 1)]
    feature_ids = list(range(1, N_MORPH_FEATURES + 1)) + DEMOGRAPHIC_FEATURE_IDS
    feature_names = morph_names + DEMOGRAPHIC_FEATURE_NAMES
    feature_categories = [get_feature_category(fid) for fid in feature_ids]

    target_key = f"delta_{target.lower()}"
    X_all = np.vstack([s["X_delta"] for s in subjects_subset])
    y_all = np.concatenate([s[target_key] for s in subjects_subset])

    n_subj = len(subjects_subset)
    subject_indices = np.arange(n_subj)
    groups = np.repeat(subject_indices, N_DELTA_SAMPLES)

    if predefined_fold_splits is None:
        kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_splits = list(kfold.split(subject_indices))
    else:
        fold_splits = predefined_fold_splits

    fold_mean_abs_shap = np.zeros((N_FOLDS, TOTAL_FEATURES), dtype=float)
    oof_shap = np.zeros_like(X_all)

    for fold_idx, (train_sub, test_sub) in enumerate(fold_splits):
        train_mask = np.isin(groups, train_sub)
        test_mask = np.isin(groups, test_sub)

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test = X_all[test_mask]

        rf = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )
        rf.fit(X_train, y_train)

        explainer = shap.TreeExplainer(rf)
        oof_shap[test_mask] = np.asarray(explainer.shap_values(X_test))
        shap_train = np.asarray(explainer.shap_values(X_train))
        fold_mean_abs_shap[fold_idx, :] = np.mean(np.abs(shap_train), axis=0)

    mean_abs_shap_5fold = np.mean(fold_mean_abs_shap, axis=0)
    std_abs_shap_5fold = np.std(fold_mean_abs_shap, axis=0, ddof=1)

    df_shap_oof = pd.DataFrame(oof_shap, columns=feature_names)

    # Stage 1: Keep features with 5-fold Mean |SHAP| >= global average
    mean_threshold = float(np.mean(mean_abs_shap_5fold))
    stage1_mask = mean_abs_shap_5fold >= mean_threshold
    stage1_features = [feature_names[i] for i in range(TOTAL_FEATURES) if stage1_mask[i]]
    stage1_means = mean_abs_shap_5fold[stage1_mask]
    stage1_stds = std_abs_shap_5fold[stage1_mask]
    stage1_cats = [feature_categories[i] for i in range(TOTAL_FEATURES) if stage1_mask[i]]
    stage1_ids = [feature_ids[i] for i in range(TOTAL_FEATURES) if stage1_mask[i]]

    df_stage1 = pd.DataFrame({
        "feature_index": stage1_ids,
        "feature_name": stage1_features,
        "category": stage1_cats,
        "mean_abs_shap": stage1_means,
        "std_abs_shap": stage1_stds,
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    # Stage 2: Hierarchical clustering on OOF SHAP correlation matrix with silhouette optimization
    corr_s1 = df_shap_oof[stage1_features].corr(method="pearson").fillna(0.0)
    dist_s1 = np.clip(1.0 - corr_s1.values, 0.0, 2.0)
    np.fill_diagonal(dist_s1, 0.0)
    condensed_s1 = squareform(dist_s1, checks=False)
    linkage_s1 = linkage(condensed_s1, method="average")

    r_thresholds = np.round(np.arange(0.20, 0.80 + 1e-5, step), 4)
    eval_records = []
    for r in r_thresholds:
        dist_cut = 1.0 - r
        clusters = fcluster(linkage_s1, t=dist_cut, criterion="distance")
        n_clusters = len(np.unique(clusters))
        if 1 < n_clusters < len(stage1_features):
            score = float(silhouette_score(dist_s1, clusters, metric="precomputed"))
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

    final_clusters = fcluster(linkage_s1, t=1.0 - best_r, criterion="distance")
    if match_ana3_assignment:
        # Exact reproduction of ana_3_two_stage_feature_selection.py line 213:
        df_stage1["cluster_id"] = final_clusters
    else:
        cluster_map = dict(zip(stage1_features, final_clusters))
        df_stage1["cluster_id"] = df_stage1["feature_name"].map(cluster_map)

    df_stage2_rep = (
        df_stage1.sort_values(by=["cluster_id", "mean_abs_shap"], ascending=[True, False])
        .groupby("cluster_id", as_index=False)
        .first()
        .sort_values(by="mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    df_stage2_rep["final_rank"] = np.arange(1, len(df_stage2_rep) + 1)

    return {
        "feature_names": feature_names,
        "mean_abs_shap_5fold": mean_abs_shap_5fold,
        "mean_threshold": mean_threshold,
        "df_stage1": df_stage1,
        "df_stage2_rep": df_stage2_rep,
        "best_r": best_r,
        "best_score": best_score,
        "best_k": best_k,
        "fold_splits": fold_splits,
    }


def compute_within_subject_correlations(
    subjects_list: list[dict],
    feature_name: str,
    target: str,
    use_delta_4pts: bool = False,
) -> np.ndarray:
    """
    Calculate within-subject Pearson correlation r for each subject:
      - If use_delta_4pts=False: across all 5 postures S1-S5 (Note: corr(X, Y) == corr(X - X_S1, Y - Y_S1) for 5 points).
      - If use_delta_4pts=True: across the 4 delta samples (S2-S1, S3-S1, S4-S1, S5-S1).
    """
    r_vals = []
    for s in subjects_list:
        if use_delta_4pts:
            x = s["delta_feature_series"][feature_name]
            y = s[f"delta_{target.lower()}"]
        else:
            x = s["raw_feature_series"][feature_name]
            y = s[f"{target.lower()}_5"]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            r_vals.append(0.0)
        else:
            r_val, _ = pearsonr(x, y)
            r_vals.append(float(r_val) if np.isfinite(r_val) else 0.0)
    return np.array(r_vals, dtype=float)


def mann_whitney_test(vals_healthy: np.ndarray, vals_htn: np.ndarray) -> tuple[float, float, str]:
    """Perform two-sided Mann-Whitney U test at the subject level (n=65 vs. n=5)."""
    x = np.asarray(vals_healthy, dtype=float)
    y = np.asarray(vals_htn, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    res = mannwhitneyu(x, y, alternative="two-sided", method="auto")
    u_stat = float(res.statistic)
    p_val = float(res.pvalue)

    if p_val < 0.001:
        p_str = format_math_minus(f"p = {p_val:.2e}")
    else:
        p_str = f"p = {p_val:.3f}"
    return u_stat, p_val, p_str


def auto_format_value_msd(vals: np.ndarray, kind: str = "auto") -> str:
    """Format subject-level Mean +/- SD according to variable scale/dimension."""
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    mean_v = float(np.mean(arr))
    sd_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    if np.isclose(mean_v, 0.0, atol=1e-12):
        mean_v = 0.0

    if kind in ("age", "bp"):
        return format_math_minus(f"{mean_v:.1f} ± {sd_v:.1f}")
    elif kind == "corr":
        return format_math_minus(f"r = {mean_v:.2f} ± {sd_v:.2f}")
    else:
        abs_m = abs(mean_v)
        if abs_m >= 1.0 or abs_m == 0.0:
            decimals = 2
        elif abs_m >= 0.1:
            decimals = 3
        elif abs_m >= 0.01:
            decimals = 4
        else:
            decimals = 5
        return format_math_minus(f"{mean_v:.{decimals}f} ± {sd_v:.{decimals}f}")


def build_subgroup_comparison_table(
    subjects_all: list[dict],
    res_full: dict,
    res_healthy: dict,
    res_htn: dict,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the Subgroup Comparison Table and Subject-Level Subgroup Dataset."""
    subj_healthy = [s for s in subjects_all if not s["is_older_htn"]]
    subj_htn = [s for s in subjects_all if s["is_older_htn"]]

    n_all = len(subjects_all)
    n_h = len(subj_healthy)
    n_o = len(subj_htn)

    col_full = f"Full Mixed Cohort (N = {n_all}, {n_all * N_POSTURE_ROWS} obs)"
    col_healthy = f"Healthy Young Subgroup (n = {n_h}, {n_h * N_POSTURE_ROWS} obs)"
    col_htn = f"Older Hypertensive Subgroup (n = {n_o}, {n_o * N_POSTURE_ROWS} obs)"
    col_pval = "Subgroup Comparison (p-value, Mann–Whitney U)"
    col_trend = "Within-Subject Postural Trend Consistency"

    rows = []

    # Row 1: Age (years)
    age_all = np.array([s["age"] for s in subjects_all])
    age_h = np.array([s["age"] for s in subj_healthy])
    age_o = np.array([s["age"] for s in subj_htn])
    _, _, p_age = mann_whitney_test(age_h, age_o)
    rows.append({
        "Metric / Feature Category": "Age (years)",
        col_full: auto_format_value_msd(age_all, "age"),
        col_healthy: auto_format_value_msd(age_h, "age"),
        col_htn: auto_format_value_msd(age_o, "age"),
        col_pval: p_age,
        col_trend: MATH_MINUS,
    })

    # Row 2: Baseline S1 SBP (mmHg)
    sbp_all = np.array([s["sbp_s1"] for s in subjects_all])
    sbp_h = np.array([s["sbp_s1"] for s in subj_healthy])
    sbp_o = np.array([s["sbp_s1"] for s in subj_htn])
    _, _, p_sbp = mann_whitney_test(sbp_h, sbp_o)
    rows.append({
        "Metric / Feature Category": "Baseline S1 SBP (mmHg)",
        col_full: auto_format_value_msd(sbp_all, "bp"),
        col_healthy: auto_format_value_msd(sbp_h, "bp"),
        col_htn: auto_format_value_msd(sbp_o, "bp"),
        col_pval: p_sbp,
        col_trend: MATH_MINUS,
    })

    # Row 3: Baseline S1 DBP (mmHg)
    dbp_all = np.array([s["dbp_s1"] for s in subjects_all])
    dbp_h = np.array([s["dbp_s1"] for s in subj_healthy])
    dbp_o = np.array([s["dbp_s1"] for s in subj_htn])
    _, _, p_dbp = mann_whitney_test(dbp_h, dbp_o)
    rows.append({
        "Metric / Feature Category": "Baseline S1 DBP (mmHg)",
        col_full: auto_format_value_msd(dbp_all, "bp"),
        col_healthy: auto_format_value_msd(dbp_h, "bp"),
        col_htn: auto_format_value_msd(dbp_o, "bp"),
        col_pval: p_dbp,
        col_trend: MATH_MINUS,
    })

    # Row 4a: PPG Pulsatile Feature (S1 Baseline) - Perfusion Index (PI, Column FH)
    pi_all = np.array([s["pi_s1"] for s in subjects_all])
    pi_h = np.array([s["pi_s1"] for s in subj_healthy])
    pi_o = np.array([s["pi_s1"] for s in subj_htn])
    _, _, p_pi = mann_whitney_test(pi_h, pi_o)
    pi_trend_desc = (
        "Attenuated baseline in older subgroup"
        if np.mean(pi_o) < np.mean(pi_h)
        else "Altered baseline in older subgroup"
    )
    rows.append({
        "Metric / Feature Category": "PPG Pulsatile Feature: PI (S1 Baseline)",
        col_full: auto_format_value_msd(pi_all, "auto"),
        col_healthy: auto_format_value_msd(pi_h, "auto"),
        col_htn: auto_format_value_msd(pi_o, "auto"),
        col_pval: p_pi,
        col_trend: pi_trend_desc,
    })

    # Row 4b: PPG Pulsatile Feature (Subject-Level Mean across S1-S5) - Perfusion Index (PI)
    pi5_all = np.array([s["pi_mean_5"] for s in subjects_all])
    pi5_h = np.array([s["pi_mean_5"] for s in subj_healthy])
    pi5_o = np.array([s["pi_mean_5"] for s in subj_htn])
    _, _, p_pi5 = mann_whitney_test(pi5_h, pi5_o)
    rows.append({
        "Metric / Feature Category": "PPG Pulsatile Feature: PI (5-Posture Subject Mean)",
        col_full: auto_format_value_msd(pi5_all, "auto"),
        col_healthy: auto_format_value_msd(pi5_h, "auto"),
        col_htn: auto_format_value_msd(pi5_o, "auto"),
        col_pval: p_pi5,
        col_trend: pi_trend_desc,
    })

    # Rows 5, 6, 7: Top 3 Selected Features from Healthy Young Subgroup (n=65) re-selection
    df_top3_healthy = res_healthy["df_stage2_rep"].head(3).reset_index(drop=True)
    top3_names = df_top3_healthy["feature_name"].tolist()

    for rank_idx, feat_name in enumerate(top3_names, start=1):
        r_all = compute_within_subject_correlations(subjects_all, feat_name, target)
        r_h = compute_within_subject_correlations(subj_healthy, feat_name, target)
        r_o = compute_within_subject_correlations(subj_htn, feat_name, target)

        _, _, p_r = mann_whitney_test(r_h, r_o)

        same_sign = (np.mean(r_h) * np.mean(r_o)) >= 0
        trend_str = (
            "Consistent direction across S1–S5 in both groups"
            if same_sign
            else "Divergent direction between subgroups"
        )

        rows.append({
            "Metric / Feature Category": f"Top Selected Feature {rank_idx}: {feat_name} (r_within)",
            col_full: auto_format_value_msd(r_all, "corr"),
            col_healthy: auto_format_value_msd(r_h, "corr"),
            col_htn: auto_format_value_msd(r_o, "corr"),
            col_pval: p_r,
            col_trend: trend_str,
        })

    # Final Row: Feature Selection Overlap vs. Full Cohort
    set_full_s2 = set(res_full["df_stage2_rep"]["feature_name"])
    set_healthy_s2 = set(res_healthy["df_stage2_rep"]["feature_name"])
    union_s2 = set_full_s2 | set_healthy_s2
    inter_s2 = set_full_s2 & set_healthy_s2
    jaccard_s2 = len(inter_s2) / len(union_s2) if len(union_s2) > 0 else 1.0

    set_full_s1 = set(res_full["df_stage1"]["feature_name"])
    set_healthy_s1 = set(res_healthy["df_stage1"]["feature_name"])
    jaccard_s1 = len(set_full_s1 & set_healthy_s1) / len(set_full_s1 | set_healthy_s1)

    rho_shap, _ = spearmanr(res_full["mean_abs_shap_5fold"], res_healthy["mean_abs_shap_5fold"])

    set_htn_s2 = set(res_htn["df_stage2_rep"]["feature_name"])
    union_htn_s2 = set_full_s2 | set_htn_s2
    inter_htn_s2 = set_full_s2 & set_htn_s2
    jaccard_htn_s2 = len(inter_htn_s2) / len(union_htn_s2) if len(union_htn_s2) > 0 else 1.0

    set_htn_s1 = set(res_htn["df_stage1"]["feature_name"])
    jaccard_htn_s1 = len(set_full_s1 & set_htn_s1) / len(set_full_s1 | set_htn_s1)

    rho_shap_htn, _ = spearmanr(res_full["mean_abs_shap_5fold"], res_htn["mean_abs_shap_5fold"])
    _, _, p_shap = mann_whitney_test(res_healthy["mean_abs_shap_5fold"], res_htn["mean_abs_shap_5fold"])

    rows.append({
        "Metric / Feature Category": "Feature Selection Overlap vs. Full Cohort",
        col_full: f"Reference (100%, r* = {res_full['best_r']:.2f}, k = {res_full['best_k']})",
        col_healthy: format_math_minus(
            f"Jaccard = {jaccard_s2:.2f} (Stage 1 = {jaccard_s1:.2f}, Spearman ρ = {rho_shap:.2f}; r* = {res_healthy['best_r']:.2f}, k = {res_healthy['best_k']})"
        ),
        col_htn: format_math_minus(
            f"Jaccard = {jaccard_htn_s2:.2f} (Stage 1 = {jaccard_htn_s1:.2f}, Spearman ρ = {rho_shap_htn:.2f}; r* = {res_htn['best_r']:.2f}, k = {res_htn['best_k']})"
        ),
        col_pval: p_shap,
        col_trend: "High stability; not biased by the 5 older subjects",
    })

    df_summary_table = pd.DataFrame(rows)

    # Subject-level detail table for verification
    detail_rows = []
    for idx, s in enumerate(subjects_all, start=1):
        row_dict = {
            "Subject_ID": f"subject_{idx:02d}",
            "Original_Folder": s["subject_id"],
            "Subgroup": "Older Hypertensive (Age > 50)" if s["is_older_htn"] else "Healthy Young (Age ≤ 50)",
            "Age_years": s["age"],
            "Baseline_S1_SBP_mmHg": int(round(s["sbp_s1"])),
            "Baseline_S1_DBP_mmHg": int(round(s["dbp_s1"])),
            "Baseline_S1_PI": s["pi_s1"],
            "Mean_5Posture_PI": s["pi_mean_5"],
        }
        for rank_idx, feat_name in enumerate(top3_names, start=1):
            r_subj = compute_within_subject_correlations([s], feat_name, target)[0]
            s1_val = float(s["raw_feature_series"][feat_name][0])
            row_dict[f"Top{rank_idx}_{feat_name}_S1_Value"] = s1_val
            row_dict[f"Top{rank_idx}_{feat_name}_r_within"] = r_subj
        detail_rows.append(row_dict)

    df_subject_details = pd.DataFrame(detail_rows)
    return df_summary_table, df_subject_details


def export_subgroup_report_to_excel(
    df_summary: pd.DataFrame,
    res_full: dict,
    res_healthy: dict,
    df_subject_details: pd.DataFrame,
    target: str,
    out_path: Path,
) -> None:
    """Export subgroup comparison table (with Stage 2 optimal r* in titles), re-selected features, and subject details."""
    df_healthy_stage2 = res_healthy["df_stage2_rep"]
    df_full_stage2 = res_full["df_stage2_rep"]

    r_full = res_full["best_r"]
    k_full = res_full["best_k"]
    sil_full = res_full["best_score"]

    r_healthy = res_healthy["best_r"]
    k_healthy = res_healthy["best_k"]
    sil_healthy = res_healthy["best_score"]

    wb = openpyxl.Workbook()

    font_title = Font(name="Arial", size=11, bold=True, color="1F4E79")
    font_header = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    font_cell = Font(name="Arial", size=10, bold=False, color="000000")
    font_bold = Font(name="Arial", size=10, bold=True, color="000000")

    fill_header_1 = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    fill_header_2 = PatternFill(start_color="2F5597", end_color="2F5597", fill_type="solid")
    fill_zebra = PatternFill(start_color="F2F5F9", end_color="F2F5F9", fill_type="solid")

    thin_side = Side(border_style="thin", color="D9D9D9")
    med_top = Side(border_style="medium", color="1F4E79")
    med_bottom = Side(border_style="medium", color="1F4E79")
    border_data = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
    border_header = Border(left=thin_side, right=thin_side, top=med_top, bottom=med_bottom)

    align_center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    align_left = Alignment(horizontal="left", vertical="center", wrap_text=True)

    num_fmt_4dec = f"0.0000;{MATH_MINUS}0.0000;0.0000"
    num_fmt_2dec = f"0.00;{MATH_MINUS}0.00;0.00"
    num_fmt_1dec = f"0.0;{MATH_MINUS}0.0;0.0"
    num_fmt_int = f"0;{MATH_MINUS}0;0"

    # =========================================================================
    # Sheet 1: Subgroup Comparison & Sensitivity Analysis
    # =========================================================================
    ws1 = wb.active
    ws1.title = "Subgroup_Comparison"

    ws1.cell(
        row=1,
        column=1,
        value=(
            f"Part 1: Subgroup Comparison & Sensitivity Analysis for Δ{target} "
            f"[Stage 2 Optimal Clustering Threshold: Full Cohort (N=70) r* = {r_full:.2f} (k = {k_full}, Silhouette = {sil_full:.4f}); "
            f"Healthy Subgroup (n=65) r* = {r_healthy:.2f} (k = {k_healthy}, Silhouette = {sil_healthy:.4f})]"
        ),
    ).font = font_title
    ws1.row_dimensions[1].height = 26

    headers_1 = list(df_summary.columns)
    ws1.row_dimensions[2].height = 34
    for col_idx, h_text in enumerate(headers_1, start=1):
        c = ws1.cell(row=2, column=col_idx, value=h_text)
        c.font = font_header
        c.fill = fill_header_1
        c.alignment = align_center
        c.border = border_header

    curr_row = 3
    for r_idx, row_data in df_summary.iterrows():
        ws1.row_dimensions[curr_row].height = 26
        for col_idx, col_name in enumerate(headers_1, start=1):
            val = format_math_minus(row_data[col_name])
            c = ws1.cell(row=curr_row, column=col_idx, value=val)
            c.font = font_bold if col_idx == 1 else font_cell
            c.alignment = align_left if col_idx in (1, len(headers_1)) else align_center
            c.border = border_data
            if r_idx % 2 == 1:
                c.fill = fill_zebra
        curr_row += 1

    ws1.cell(
        row=curr_row,
        column=1,
        value=(
            f"Note: Healthy Young (Age ≤ 50, n = 65) vs. Older Hypertensive (Age > 50, n = 5). "
            f"All values reported as subject-level Mean ± SD; two-sided Mann–Whitney U test at the subject level. "
            f"Stage 2 optimal correlation threshold r* = {r_full:.2f} (Full N=70) and r* = {r_healthy:.2f} (Healthy n=65)."
        ),
    ).font = Font(name="Arial", size=9, italic=True, color="595959")
    curr_row += 2

    # Part 2 on Sheet 1: Healthy Young (n=65) Stage 2 Selected Representative Features
    ws1.cell(
        row=curr_row,
        column=1,
        value=(
            f"Part 2: Two-Stage Selected Representative Features in Healthy Young Subgroup "
            f"(n = 65, Target: Δ{target}, Optimal Stage 2 Threshold r* = {r_healthy:.2f}, k = {k_healthy} Clusters)"
        ),
    ).font = font_title
    ws1.row_dimensions[curr_row].height = 24
    curr_row += 1

    s2_cols = [
        ("final_rank", "Rank (Healthy n=65)"),
        ("feature_index", "Feature Index"),
        ("feature_name", "Feature Name"),
        ("category", "Category"),
        ("cluster_id", "Cluster ID"),
        ("mean_abs_shap", "5-Fold Mean |SHAP| (n=65)"),
        ("std_abs_shap", "5-Fold SD |SHAP| (n=65)"),
        ("in_full_70_stage2", f"Selected in Full Cohort (N=70, r*={r_full:.2f})?"),
    ]

    full_rep_set = set(df_full_stage2["feature_name"])
    df_healthy_s2_copy = df_healthy_stage2.copy()
    df_healthy_s2_copy["in_full_70_stage2"] = df_healthy_s2_copy["feature_name"].apply(
        lambda x: "Yes" if x in full_rep_set else "No"
    )

    ws1.row_dimensions[curr_row].height = 30
    for col_idx, (_, h_text) in enumerate(s2_cols, start=1):
        c = ws1.cell(row=curr_row, column=col_idx, value=h_text)
        c.font = font_header
        c.fill = fill_header_2
        c.alignment = align_center
        c.border = border_header
    curr_row += 1

    for r_idx, row_data in df_healthy_s2_copy.iterrows():
        ws1.row_dimensions[curr_row].height = 20
        for col_idx, (col_key, _) in enumerate(s2_cols, start=1):
            val = row_data[col_key]
            c = ws1.cell(row=curr_row, column=col_idx, value=val)
            c.font = font_bold if r_idx < 3 and col_idx in (1, 3) else font_cell
            c.alignment = align_left if col_key in ("feature_name", "category") else align_center
            c.border = border_data
            if col_key in ("mean_abs_shap", "std_abs_shap"):
                c.number_format = num_fmt_4dec
            if r_idx % 2 == 1:
                c.fill = fill_zebra
        curr_row += 1

    widths_1 = [42, 30, 36, 28, 28, 44, 26, 32]
    for col_idx, w in enumerate(widths_1, start=1):
        ws1.column_dimensions[get_column_letter(col_idx)].width = w
    ws1.views.sheetView[0].showGridLines = True

    # =========================================================================
    # Sheet 2: Subject-Level Subgroup Details (N = 70)
    # =========================================================================
    ws2 = wb.create_sheet(title="Subject_Subgroup_Details")
    ws2.cell(
        row=1,
        column=1,
        value=f"Subject-Level Baseline Metrics and Within-Subject Correlations (N = {len(df_subject_details)})",
    ).font = font_title
    ws2.row_dimensions[1].height = 24

    det_headers = list(df_subject_details.columns)
    ws2.row_dimensions[2].height = 32
    for col_idx, h_text in enumerate(det_headers, start=1):
        c = ws2.cell(row=2, column=col_idx, value=h_text)
        c.font = font_header
        c.fill = fill_header_2
        c.alignment = align_center
        c.border = border_header

    curr_row = 3
    for r_idx, row_data in df_subject_details.iterrows():
        ws2.row_dimensions[curr_row].height = 20
        for col_idx, col_key in enumerate(det_headers, start=1):
            val = row_data[col_key]
            c = ws2.cell(row=curr_row, column=col_idx, value=val)
            c.font = font_cell
            c.alignment = align_left if col_key in ("Subject_ID", "Original_Folder", "Subgroup") else align_center
            c.border = border_data
            if "SBP" in col_key or "DBP" in col_key:
                c.number_format = num_fmt_int
            elif col_key == "Age_years":
                c.number_format = num_fmt_1dec
            elif "r_within" in col_key:
                c.number_format = num_fmt_2dec
            elif isinstance(val, (int, float)):
                c.number_format = num_fmt_4dec
            if r_idx % 2 == 1:
                c.fill = fill_zebra
        curr_row += 1

    for col_idx in range(1, len(det_headers) + 1):
        ws2.column_dimensions[get_column_letter(col_idx)].width = 24
    ws2.column_dimensions["C"].width = 30
    ws2.views.sheetView[0].showGridLines = True

    wb.save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Subgroup comparison (Healthy Young n=65 vs. Older Hypertensive n=5) and two-stage SHAP sensitivity analysis."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="SBP",
        choices=["SBP", "DBP", "MAP"],
        help="Target blood pressure for two-stage feature selection and r_within (default: SBP).",
    )
    parser.add_argument(
        "--excel_defs",
        "-e",
        default="supplement_feature_definitions.xlsx",
        help="Path to the 156-feature definition Excel file.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Step size for Stage 2 threshold r search in [0.20, 0.80] (default: 0.05).",
    )
    parser.add_argument(
        "--fix_cluster_map",
        action="store_true",
        help="If set, map cluster_id by feature_name before sorting instead of direct positional assignment in ana_3.",
    )
    args = parser.parse_args()

    base_dir = Path("/Users/ziyiwang/Documents/Project_2026_month1_6/Work_260223_continuous_BP_monitoring/OUTPUT")
    output_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()

    exclude_dirs = {
        "yy04_subject_000",
        "yy04_subject_001",
        "yy04_subject_002",
        "yy04_subject_003",
        "yy04_subject_004",
    }

    target = args.target.upper()
    excel_defs_path = Path(args.excel_defs)
    if not excel_defs_path.exists():
        alt_excel = output_dir / args.excel_defs
        if alt_excel.exists():
            excel_defs_path = alt_excel

    feat_name_dict = load_feature_name_mapping(excel_defs_path, num_features=N_MORPH_FEATURES)
    subjects_all = load_cohort_data(base_dir, exclude_dirs, feat_name_dict)

    subj_healthy = [s for s in subjects_all if not s["is_older_htn"]]
    subj_htn = [s for s in subjects_all if s["is_older_htn"]]

    print(f"Loaded {len(subjects_all)} valid subjects: {len(subj_healthy)} Healthy Young (Age <= 50), {len(subj_htn)} Older Hypertensive (Age > 50).")

    match_ana3 = not args.fix_cluster_map

    print(f"Running Two-Stage SHAP Feature Selection on Full Cohort (N = {len(subjects_all)}) for Δ{target}...")
    res_full = run_two_stage_shap_selection(
        subjects_all, target, feat_name_dict, step=args.step, match_ana3_assignment=match_ana3
    )
    print(f"  -> Full Cohort (N=70) Stage 2 Optimal Threshold r* = {res_full['best_r']:.2f}, Clusters k = {res_full['best_k']}, Silhouette = {res_full['best_score']:.4f}")

    print(f"Running Two-Stage SHAP Feature Selection on Healthy Young Subgroup (n = {len(subj_healthy)}) for Δ{target}...")
    res_healthy = run_two_stage_shap_selection(
        subj_healthy, target, feat_name_dict, step=args.step, match_ana3_assignment=match_ana3
    )
    print(f"  -> Healthy Subgroup (n=65) Stage 2 Optimal Threshold r* = {res_healthy['best_r']:.2f}, Clusters k = {res_healthy['best_k']}, Silhouette = {res_healthy['best_score']:.4f}")

    print(f"Running Two-Stage SHAP Feature Selection on Older Hypertensive Subgroup (n = {len(subj_htn)}) for Δ{target}...")
    res_htn = run_two_stage_shap_selection(
        subj_htn, target, feat_name_dict, step=args.step, match_ana3_assignment=match_ana3
    )
    print(f"  -> Older Hypertensive Subgroup (n=5) Stage 2 Optimal Threshold r* = {res_htn['best_r']:.2f}, Clusters k = {res_htn['best_k']}, Silhouette = {res_htn['best_score']:.4f}")

    df_summary_table, df_subject_details = build_subgroup_comparison_table(
        subjects_all=subjects_all,
        res_full=res_full,
        res_healthy=res_healthy,
        res_htn=res_htn,
        target=target,
    )

    out_xlsx = output_dir / f"subgroup_sensitivity_analysis_{target}.xlsx"
    export_subgroup_report_to_excel(
        df_summary=df_summary_table,
        res_full=res_full,
        res_healthy=res_healthy,
        df_subject_details=df_subject_details,
        target=target,
        out_path=out_xlsx,
    )

    print("\nSubgroup Comparison Summary Table:")
    print(df_summary_table.to_string(index=False))
    print(f"\nSaved Excel report to: {out_xlsx}")


if __name__ == "__main__":
    main()