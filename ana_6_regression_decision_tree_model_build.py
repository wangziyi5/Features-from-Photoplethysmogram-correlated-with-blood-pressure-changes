import io
import os
import random
import re
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon, t as student_t
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 20
plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"

# Fix all random seeds to guarantee exact reproducibility
RANDOM_STATE = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

BASE_DIR = "/Users/ziyiwang/Documents/Project_2026_month1_6/Work_260223_continuous_BP_monitoring/OUTPUT"
ANALYSIS_DIR = "/Users/ziyiwang/Documents/Project_2026_month7_9/Work_260827_Multi_Feature_Relation"
EXCLUDE_SUBJECTS = {
    "yy04_subject_000",
    "yy04_subject_001",
    "yy04_subject_002",
    "yy04_subject_003",
    "yy04_subject_004",
}

FEATURE_FILES = {
    "SBP": "SBP_SHAP_stage2_clusters_features.csv",
    "DBP": "DBP_SHAP_stage2_clusters_features.csv",
    "MAP": "MAP_SHAP_stage2_clusters_features.csv",
}
DEFINITION_FILE = "supplement_feature_definitions.xlsx"

DEMO_COL_MAP_0BASED = {
    "HEIGHT": 5,
    "WEIGHT": 6,
    "BMI": 7,
    "GENDER": 8,
    "SEX": 8,
    "AGE": 9,
}


def find_file(filename, search_dirs):
    for directory in search_dirs:
        if not directory or not os.path.exists(directory):
            continue
        candidate = os.path.join(directory, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def normalise_feature_name(value):
    return str(value).strip().casefold()


def relaxed_clean_name(value):
    """Normalize feature name by stripping non-alphanumeric characters for robust matching."""
    return re.sub(r"[^a-zA-Z0-9]", "", str(value)).casefold()


def load_feature_definitions(excel_path):
    """Map feature name to serial number (ID) from supplement_feature_definitions.xlsx."""
    df_def = pd.read_excel(excel_path, header=None, usecols=[0, 1])
    name_to_id = {}
    relaxed_to_id = {}
    serial_numbers = []

    for serial_number, feature_name in df_def.itertuples(index=False, name=None):
        if pd.isna(serial_number) or pd.isna(feature_name):
            continue
        try:
            serial_number = int(float(serial_number))
        except (TypeError, ValueError):
            continue

        feat_str = str(feature_name).strip()
        if not feat_str:
            continue

        norm_key = normalise_feature_name(feat_str)
        relax_key = relaxed_clean_name(feat_str)

        name_to_id[norm_key] = serial_number
        if relax_key:
            relaxed_to_id[relax_key] = serial_number
        serial_numbers.append(serial_number)

    min_serial = min(serial_numbers) if serial_numbers else 1
    serial_offset = 9 if min_serial >= 1 else 10

    return name_to_id, relaxed_to_id, serial_offset


def load_stage2_selected_features(file_name, search_dirs):
    """Extract feature names under 'STAGE 2 FINAL REPRESENTATIVE FEATURES' section."""
    file_path = find_file(file_name, search_dirs)
    if file_path is None:
        raise FileNotFoundError(f"Could not locate feature file: {file_name}")

    content = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "latin1"]:
        try:
            with open(file_path, "r", encoding=enc) as f:
                content = f.readlines()
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        raise ValueError(f"Could not decode feature file: {file_path}")

    stage2_start_idx = -1
    for idx, line in enumerate(content):
        line_clean = line.strip().upper()
        if "STAGE 2" in line_clean and "REPRESENTATIVE" in line_clean:
            stage2_start_idx = idx
            break

    if stage2_start_idx != -1:
        table_lines = []
        for line in content[stage2_start_idx + 1 :]:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if set(stripped.replace("#", "").strip()) <= {"=", "-", "_"}:
                    continue
                if "STAGE" in stripped.upper() or "SUMMARY" in stripped.upper():
                    break
                continue
            table_lines.append(line)

        if table_lines:
            df_stage2 = pd.read_csv(io.StringIO("".join(table_lines)))
            feat_col = None
            for col in df_stage2.columns:
                c_clean = str(col).strip().lower()
                if c_clean in ["feature_name", "feature", "features", "feat_name"]:
                    feat_col = col
                    break
            if feat_col is None:
                for col in df_stage2.columns:
                    if "feat" in str(col).strip().lower():
                        feat_col = col
                        break

            if feat_col is not None:
                features = (
                    df_stage2[feat_col]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .tolist()
                )
                features = [f for f in features if f and f.lower() not in ["none", "nan", ""]]
                if features:
                    return features

    for enc in ["utf-8-sig", "utf-8", "gbk", "latin1"]:
        try:
            df_fallback = pd.read_csv(file_path, encoding=enc)
            break
        except Exception:
            continue

    if "feature_name" in df_fallback.columns:
        if "is_stage2_representative" in df_fallback.columns:
            mask = (
                df_fallback["is_stage2_representative"]
                .astype(str)
                .str.strip()
                .str.casefold()
                .isin(["true", "1", "yes"])
            )
            features = df_fallback.loc[mask, "feature_name"].dropna().astype(str).str.strip().tolist()
        else:
            features = df_fallback["feature_name"].dropna().astype(str).str.strip().tolist()
        if features:
            return features

    raise ValueError(f"Could not extract Stage 2 representative features from: {file_path}")


def identify_bp_columns(df):
    """Identify SBP and DBP reference columns within the first 10 columns."""
    sbp_col, dbp_col = None, None
    for col in df.columns[:10]:
        name = str(col).strip().upper()
        if ("SBP" in name or "SYS" in name) and "DBP" not in name and "DIA" not in name:
            sbp_col = col
        elif ("DBP" in name or "DIA" in name) and "SBP" not in name and "SYS" not in name:
            dbp_col = col

    if sbp_col is None or dbp_col is None:
        raise ValueError(
            f"Could not identify both SBP and DBP columns within first 10 columns of all_trimmed.csv. "
            f"Columns inspected: {list(df.columns[:10])}"
        )
    return sbp_col, dbp_col


def encode_gender(val):
    if pd.isna(val):
        return 0.0
    val_str = str(val).strip().lower()
    if val_str in ["m", "male", "1", "1.0", "男"]:
        return 1.0
    return 0.0


def map_feature_to_col_idx(feat_name, feat_mapping, relaxed_mapping, serial_offset, df_columns):
    """Map feature name to 0-based column index in all_trimmed.csv."""
    feature_key = normalise_feature_name(feat_name)
    relaxed_key = relaxed_clean_name(feat_name)

    if feature_key.upper() in DEMO_COL_MAP_0BASED:
        return DEMO_COL_MAP_0BASED[feature_key.upper()], True

    if feature_key in feat_mapping:
        serial_id = feat_mapping[feature_key]
        return serial_id + serial_offset, False

    if relaxed_key in relaxed_mapping:
        serial_id = relaxed_mapping[relaxed_key]
        return serial_id + serial_offset, False

    column_lookup = {normalise_feature_name(col): idx for idx, col in enumerate(df_columns)}
    if feature_key in column_lookup:
        col_idx = column_lookup[feature_key]
        return col_idx, 5 <= col_idx <= 9

    relaxed_col_lookup = {relaxed_clean_name(col): idx for idx, col in enumerate(df_columns)}
    if relaxed_key in relaxed_col_lookup:
        col_idx = relaxed_col_lookup[relaxed_key]
        return col_idx, 5 <= col_idx <= 9

    return None, False


def resolve_selected_feature_indices(
    selected_features_dict, feat_mapping, relaxed_mapping, serial_offset, df_columns
):
    """Resolve and validate selected-feature positions before model training."""
    selected_indices = {}
    for target_key, feature_names in selected_features_dict.items():
        indices = []
        missing_features = []
        for feature_name in feature_names:
            col_idx, is_demo = map_feature_to_col_idx(
                feature_name, feat_mapping, relaxed_mapping, serial_offset, df_columns
            )
            if col_idx is None or col_idx >= len(df_columns):
                missing_features.append(feature_name)
            else:
                indices.append((col_idx, is_demo))

        if missing_features:
            raise ValueError(
                f"[{target_key}] Features cannot be mapped to all_trimmed.csv: {missing_features}"
            )
        selected_indices[target_key] = indices
    return selected_indices


def extract_subject_records(csv_path, selected_features_dict, feat_mapping, relaxed_mapping, serial_offset):
    """Process subject data: row 0 is baseline; rows 1-4 generate 4 delta records."""
    df = pd.read_csv(csv_path)
    if len(df) < 5:
        return None

    if len(df.columns) < 166:
        raise ValueError(
            f"Columns insufficient: expected at least 166, got {len(df.columns)} in {csv_path}"
        )

    demo_indices = [5, 6, 7, 8, 9]
    signal_indices = list(range(10, 166))
    df_clean = df.copy()

    demo_cols = [df_clean.columns[i] for i in demo_indices]
    signal_cols = [df_clean.columns[i] for i in signal_indices]
    sbp_col, dbp_col = identify_bp_columns(df_clean)

    gender_col = df_clean.columns[8]
    df_clean[gender_col] = (
        df_clean[gender_col].apply(encode_gender).astype(float)
    )

    target_cols = list(
        dict.fromkeys(demo_cols + signal_cols + [sbp_col, dbp_col])
    )
    for col in target_cols:
      df_clean[col] = (
          pd.to_numeric(df_clean[col], errors="coerce")
          .astype(float)
          .ffill()
          .bfill()
          .fillna(0.0)
      )

    static_demo = df_clean.iloc[0, demo_indices].values.astype(float)

    base_signal = df_clean.iloc[0, signal_indices].values.astype(float)
    base_sbp = float(df_clean.iloc[0][sbp_col])
    base_dbp = float(df_clean.iloc[0][dbp_col])
    base_map = (base_sbp + 2.0 * base_dbp) / 3.0

    col_names = list(df_clean.columns)
    target_feat_indices = resolve_selected_feature_indices(
        selected_features_dict, feat_mapping, relaxed_mapping, serial_offset, col_names
    )

    records = []
    for idx in range(1, 5):
        cur_row = df_clean.iloc[idx]
        cur_sbp = float(cur_row[sbp_col])
        cur_dbp = float(cur_row[dbp_col])
        cur_map = (cur_sbp + 2.0 * cur_dbp) / 3.0

        delta_sbp = cur_sbp - base_sbp
        delta_dbp = cur_dbp - base_dbp
        delta_map = cur_map - base_map

        cur_signal = cur_row.iloc[signal_indices].values.astype(float)
        delta_signal = cur_signal - base_signal
        full_feat = np.concatenate([delta_signal, static_demo])

        feat_vectors = {}
        for target_key, col_info_list in target_feat_indices.items():
            feature_vector = []
            for col_idx, is_demo in col_info_list:
                if is_demo:
                    feature_vector.append(float(df_clean.iloc[0, col_idx]))
                else:
                    feature_vector.append(
                        float(df_clean.iloc[idx, col_idx])
                        - float(df_clean.iloc[0, col_idx])
                    )
            feat_vectors[target_key] = np.array(feature_vector, dtype=float)

        records.append({
            "record_idx": idx,
            "full_features": full_feat,
            "sbp_selected": feat_vectors["SBP"],
            "dbp_selected": feat_vectors["DBP"],
            "map_selected": feat_vectors["MAP"],
            "base_sbp": base_sbp,
            "base_dbp": base_dbp,
            "base_map": base_map,
            "cur_sbp": cur_sbp,
            "cur_dbp": cur_dbp,
            "cur_map": cur_map,
            "delta_sbp": delta_sbp,
            "delta_dbp": delta_dbp,
            "delta_map": delta_map,
        })

    return records


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    if (
        len(np.unique(y_pred)) <= 1
        or np.isnan(y_pred).any()
        or np.std(y_true) == 0
        or np.std(y_pred) == 0
    ):
        r = 0.0
    else:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    return mae, rmse, r, r2


def get_sig_label(p_val):
    if pd.isna(p_val):
        return "ns"
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    return "ns"


def corrected_resampled_ttest(fold_diffs, n_train, n_test):
    k = len(fold_diffs)
    diffs = np.asarray(fold_diffs, dtype=float)
    mean_diff = float(np.mean(diffs))
    var_diff = float(np.var(diffs, ddof=1)) if k > 1 else 0.0
    if var_diff == 0.0:
        return 0.0, 1.0
    se_corr = np.sqrt((1.0 / k + float(n_test) / float(n_train)) * var_diff)
    t_stat = mean_diff / se_corr
    p_val = float(2.0 * (1.0 - student_t.cdf(abs(t_stat), df=k - 1)))
    return float(t_stat), p_val


def evaluate_mae_significance_and_ci(df_samples, df_folds, n_train, n_test, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    stats_dict = {}

    for target in ["SBP", "DBP", "MAP"]:
        sub_old = (
            df_samples[
                (df_samples["Target_BP"] == target)
                & (df_samples["Input_Feature_Set"] == "All_161_Features")
            ]
            .sort_values(by=["Fold", "Subject_ID", "Record_Idx"])
            .reset_index(drop=True)
        )
        sub_new = (
            df_samples[
                (df_samples["Target_BP"] == target)
                & (df_samples["Input_Feature_Set"] == "Stage2_Selected")
            ]
            .sort_values(by=["Fold", "Subject_ID", "Record_Idx"])
            .reset_index(drop=True)
        )

        ae_old = np.abs(sub_old["True_BP"].values.astype(float) - sub_old["Pred_BP"].values.astype(float))
        ae_new = np.abs(sub_new["True_BP"].values.astype(float) - sub_new["Pred_BP"].values.astype(float))
        delta_ae = ae_old - ae_new

        _, sw_p = shapiro(delta_ae)
        is_normal = bool(sw_p > 0.05)

        if is_normal:
            _, sample_p = ttest_rel(ae_old, ae_new)
            sample_test_name = "Paired t-test"
        else:
            _, sample_p = wilcoxon(ae_old, ae_new)
            sample_test_name = "Wilcoxon Signed-Rank Test"
        sample_p = float(sample_p)

        fold_old = (
            df_folds[
                (df_folds["Target_BP"] == target)
                & (df_folds["Input_Feature_Set"] == "All_161_Features")
            ]
            .sort_values(by="Fold")["MAE"]
            .values.astype(float)
        )
        fold_new = (
            df_folds[
                (df_folds["Target_BP"] == target)
                & (df_folds["Input_Feature_Set"] == "Stage2_Selected")
            ]
            .sort_values(by="Fold")["MAE"]
            .values.astype(float)
        )
        fold_diffs = fold_old - fold_new
        _, corr_t_p = corrected_resampled_ttest(fold_diffs, n_train=n_train, n_test=n_test)

        n_samples = len(delta_ae)
        boot_means = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            idx = rng.integers(0, n_samples, size=n_samples)
            boot_means[b] = np.mean(delta_ae[idx])

        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        delta_mae_mean = float(np.mean(delta_ae))

        stats_dict[target] = {
            "Shapiro_p": round(float(sw_p), 6),
            "Sample_Test": sample_test_name,
            "Sample_p_value": round(sample_p, 6),
            "Sample_Sig": get_sig_label(sample_p),
            "Corrected_t_p_value": round(corr_t_p, 6),
            "Corrected_t_Sig": get_sig_label(corr_t_p),
            "p_value": round(sample_p, 6),
            "Delta_MAE_95CI": f"{delta_mae_mean:.4f} [{ci_low:.4f}, {ci_high:.4f}]",
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "delta_mae_mean": delta_mae_mean,
        }

    return stats_dict


def evaluate_naive_baselines(subjects, subject_data, kf):
    """Evaluate 'Predicting zero change' and 'Predicting per-posture mean' baselines."""
    target_configs = [
        ("SBP", "delta_sbp", "base_sbp", "cur_sbp"),
        ("DBP", "delta_dbp", "base_dbp", "cur_dbp"),
        ("MAP", "delta_map", "base_map", "cur_map"),
    ]
    baseline_methods = ["Predicting_Zero_Change", "Predicting_Per_Posture_Mean"]

    fold_records = []
    pooled_records = {
        (m, t): {"dt_true": [], "dt_pred": [], "abs_true": [], "abs_pred": []}
        for m in baseline_methods
        for t, _, _, _ in target_configs
    }

    for fold, (train_idx, test_idx) in enumerate(kf.split(subjects), 1):
        train_subs = [subjects[i] for i in train_idx]
        test_subs = [subjects[i] for i in test_idx]
        train_recs = [rec for s in train_subs for rec in subject_data[s]]
        test_recs = [rec for s in test_subs for rec in subject_data[s]]

        for target_bp, delta_key, base_key, abs_key in target_configs:
            y_test_delta = np.array([r[delta_key] for r in test_recs], dtype=float)
            base_test = np.array([r[base_key] for r in test_recs], dtype=float)
            abs_test = np.array([r[abs_key] for r in test_recs], dtype=float)

            # 1. Predicting zero change (delta = 0)
            pred_delta_zero = np.zeros_like(y_test_delta)
            pred_abs_zero = base_test + pred_delta_zero

            # 2. Predicting per-posture mean (computed from training fold)
            posture_means = {
                idx: float(np.mean([r[delta_key] for r in train_recs if r["record_idx"] == idx]))
                for idx in range(1, 5)
            }
            pred_delta_posture = np.array([posture_means[r["record_idx"]] for r in test_recs], dtype=float)
            pred_abs_posture = base_test + pred_delta_posture

            for method_name, dt_pred, abs_pred in [
                ("Predicting_Zero_Change", pred_delta_zero, pred_abs_zero),
                ("Predicting_Per_Posture_Mean", pred_delta_posture, pred_abs_posture),
            ]:
                mae, rmse, r_delta, r2_delta = calculate_metrics(y_test_delta, dt_pred)
                _, _, r_abs, r2_abs = calculate_metrics(abs_test, abs_pred)
                err_abs = abs_pred - abs_test
                sd_err = float(np.std(err_abs, ddof=1)) if len(err_abs) > 1 else 0.0

                fold_records.append({
                    "Fold": fold,
                    "Baseline_Method": method_name,
                    "Target_BP": target_bp,
                    "MAE": mae,
                    "RMSE": rmse,
                    "Delta_r": r_delta,
                    "Delta_R2": r2_delta,
                    "Abs_r": r_abs,
                    "Abs_R2": r2_abs,
                    "BP_std": float(np.std(abs_pred)),
                    "BP_sd": sd_err,
                })

                pooled_records[(method_name, target_bp)]["dt_true"].extend(y_test_delta)
                pooled_records[(method_name, target_bp)]["dt_pred"].extend(dt_pred)
                pooled_records[(method_name, target_bp)]["abs_true"].extend(abs_test)
                pooled_records[(method_name, target_bp)]["abs_pred"].extend(abs_pred)

    df_base_folds = pd.DataFrame(fold_records)
    baseline_summary_rows = []

    for method_name in baseline_methods:
        for target_bp, delta_key, base_key, abs_key in target_configs:
            grp = df_base_folds[
                (df_base_folds["Baseline_Method"] == method_name)
                & (df_base_folds["Target_BP"] == target_bp)
            ]
            p_data = pooled_records[(method_name, target_bp)]
            dt_t = np.array(p_data["dt_true"], dtype=float)
            dt_p = np.array(p_data["dt_pred"], dtype=float)
            ab_t = np.array(p_data["abs_true"], dtype=float)
            ab_p = np.array(p_data["abs_pred"], dtype=float)

            # Out-of-fold pooled metrics
            p_mae, p_rmse, p_r_dt, p_r2_dt = calculate_metrics(dt_t, dt_p)
            _, _, p_r_ab, p_r2_ab = calculate_metrics(ab_t, ab_p)

            # In-sample whole-dataset posture mean (exact match to Table 2 theoretical formula)
            all_recs = [rec for s in subjects for rec in subject_data[s]]
            all_dt_true = np.array([r[delta_key] for r in all_recs], dtype=float)
            if method_name == "Predicting_Zero_Change":
                exact_dt_pred = np.zeros_like(all_dt_true)
            else:
                global_p_means = {
                    idx: float(np.mean([r[delta_key] for r in all_recs if r["record_idx"] == idx]))
                    for idx in range(1, 5)
                }
                exact_dt_pred = np.array([global_p_means[r["record_idx"]] for r in all_recs], dtype=float)
            exact_rmse = float(np.sqrt(mean_squared_error(all_dt_true, exact_dt_pred)))

            baseline_summary_rows.append({
                "Baseline_Method": method_name,
                "Target_BP": target_bp,
                "MAE_Mean": round(grp["MAE"].mean(), 4),
                "MAE_Std": round(grp["MAE"].std(), 4),
                "RMSE_Mean": round(grp["RMSE"].mean(), 4),
                "RMSE_Std": round(grp["RMSE"].std(), 4),
                "Pooled_RMSE": round(p_rmse, 4),
                "Table2_Exact_RMSE": round(exact_rmse, 4),
                "Delta_r_Mean": round(grp["Delta_r"].mean(), 4),
                "Delta_R2_Mean": round(grp["Delta_R2"].mean(), 4),
                "Pooled_Delta_R2": round(p_r2_dt, 4),
                "Abs_r_Mean": round(grp["Abs_r"].mean(), 4),
                "Abs_R2_Mean": round(grp["Abs_R2"].mean(), 4),
                "Pooled_Abs_R2": round(p_r2_ab, 4),
                "BP_std": round(grp["BP_std"].mean(), 4),
                "BP_sd": round(grp["BP_sd"].mean(), 4),
            })

    return pd.DataFrame(baseline_summary_rows)


def plot_bp_regression_scatters(df_samples, output_dir):
  """Generate high-resolution (300 DPI) regression scatter plots for SBP, DBP, and MAP."""
  os.makedirs(output_dir, exist_ok=True)
  bp_targets = ["SBP", "DBP", "MAP"]
  feature_subsets = [
      ("All_161_Features", "All 161 Features"),
      ("Stage2_Selected", "Stage 2 Selected Features"),
  ]

  for target in bp_targets:
    sub_df = df_samples[df_samples["Target_BP"] == target]
    if sub_df.empty:
      continue

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5.5), sharex=False, sharey=False
    )
    fig.suptitle(
        f"Decision Tree Regression: {target} Estimation (5-Fold CV)",
        fontsize=14,
        fontweight="bold",
    )

    val_min = min(sub_df["True_BP"].min(), sub_df["Pred_BP"].min()) - 5.0
    val_max = max(sub_df["True_BP"].max(), sub_df["Pred_BP"].max()) + 5.0

    for ax, (feat_key, feat_title) in zip(axes, feature_subsets):
      data = sub_df[sub_df["Input_Feature_Set"] == feat_key]
      y_true = data["True_BP"].values.astype(float)
      y_pred = data["Pred_BP"].values.astype(float)

      mae, rmse, r, r2 = calculate_metrics(y_true, y_pred)

      pt_color = "#1f77b4" if "161" in feat_key else "#2ca02c"
      ax.scatter(
          y_true,
          y_pred,
          alpha=0.6,
          edgecolors="none",
          color=pt_color,
          s=35,
          label="Samples",
      )

      lims = [val_min, val_max]
      ax.plot(
          lims,
          lims,
          "k--",
          alpha=0.75,
          linewidth=1.5,
          label="Identity line (y = x)",
      )

      if len(y_true) > 1:
        slope, intercept = np.polyfit(y_true, y_pred, 1)
        x_trend = np.linspace(val_min, val_max, 100)
        ax.plot(
            x_trend,
            slope * x_trend + intercept,
            color="red",
            linestyle="-",
            linewidth=1.5,
            label=f"Fit line (slope = {slope:.2f})",
        )

      ax.set_xlim(val_min, val_max)
      ax.set_ylim(val_min, val_max)
      ax.set_aspect("equal", adjustable="box")
      ax.set_title(feat_title, fontsize=12, fontweight="semibold")
      ax.set_xlabel(f"Reference {target} (mmHg)", fontsize=16)
      ax.set_ylabel(f"Predicted {target} (mmHg)", fontsize=16)
      ax.grid(True, linestyle=":", alpha=0.6)

      metric_text = (
          f"MAE = {mae:.2f} mmHg\n"
          f"RMSE = {rmse:.2f} mmHg\n"
          f"Pearson r = {r:.3f}\n"
          f"$R^2$ = {r2:.3f}"
      )
      props = dict(
          boxstyle="round,pad=0.5",
          facecolor="white",
          edgecolor="gray",
          alpha=0.9,
      )
      ax.text(
          0.05,
          0.95,
          metric_text,
          transform=ax.transAxes,
          fontsize=14,
          verticalalignment="top",
          bbox=props,
      )
      ax.legend(loc="lower right", fontsize=14, framealpha=0.85)

    plt.tight_layout()
    output_fig_path = os.path.join(
        output_dir, f"regression_scatter_{target}_cv.png"
    )
    plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 300 DPI regression scatter plot to: {output_fig_path}")


def format_math_minus(val, decimals=2):
    s = f"{val:.{decimals}f}"
    return s.replace("-", "\u2212")


def plot_bland_altman_figures(df_samples, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    bp_targets = ["SBP", "DBP", "MAP"]
    xlim_map = {
        "SBP": (60, 160),
        "DBP": (40, 120),
        "MAP": (40, 120),
    }
    feature_configs = [
        ("All_161_Features", "All 161 Features", "bland_altman_All_161_Features.png", "#1f77b4"),
        ("Stage2_Selected", "Stage 2 Selected Features", "bland_altman_Stage2_Selected_Features.png", "#2ca02c"),
    ]

    tick_formatter = FuncFormatter(
        lambda x, pos: f"{int(x)}".replace("-", "\u2212")
        if np.isclose(x, round(x))
        else f"{x:g}".replace("-", "\u2212")
    )

    for feat_key, feat_title, filename, pt_color in feature_configs:
        fig, axes = plt.subplots(1, 3, figsize=(24, 7.5))

        for ax, target in zip(axes, bp_targets):
            sub_df = df_samples[
                (df_samples["Target_BP"] == target)
                & (df_samples["Input_Feature_Set"] == feat_key)
            ]
            y_true = sub_df["True_BP"].values.astype(float)
            y_pred = sub_df["Pred_BP"].values.astype(float)
            n_pts = len(y_true)

            mean_bp = (y_true + y_pred) / 2.0
            diff_bp = y_pred - y_true

            md = float(np.mean(diff_bp))
            sd = float(np.std(diff_bp, ddof=1))
            loa_upper = md + 1.96 * sd
            loa_lower = md - 1.96 * sd

            ax.scatter(
                mean_bp,
                diff_bp,
                alpha=0.65,
                edgecolors="none",
                color=pt_color,
                s=55,
            )

            ax.axhline(md, color="#d62728", linestyle="-", linewidth=2.2)
            ax.axhline(loa_upper, color="#333333", linestyle="--", linewidth=2.0)
            ax.axhline(loa_lower, color="#333333", linestyle="--", linewidth=2.0)

            ax.set_xlim(xlim_map[target])
            ax.set_ylim(-60, 60)

            ax.xaxis.set_major_formatter(tick_formatter)
            ax.yaxis.set_major_formatter(tick_formatter)

            x_text = ax.get_xlim()[1] - (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.03
            y_offset = 120.0 * 0.025

            ax.text(
                x_text,
                loa_upper + y_offset,
                f"+1.96 SD: {format_math_minus(loa_upper)}",
                ha="right",
                va="bottom",
                fontsize=24,
                fontname="Arial",
                color="#333333",
            )
            ax.text(
                x_text,
                md + y_offset,
                f"Mean: {format_math_minus(md)}",
                ha="right",
                va="bottom",
                fontsize=24,
                fontname="Arial",
                color="#d62728",
            )
            ax.text(
                x_text,
                loa_lower - y_offset,
                f"\u22121.96 SD: {format_math_minus(loa_lower)}",
                ha="right",
                va="top",
                fontsize=24,
                fontname="Arial",
                color="#333333",
            )

            ax.text(
                0.96,
                0.94,
                f"sample size $\\mathit{{n}}$ = {n_pts}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=24,
                fontname="Arial",
                color="#000000",
            )

            ax.set_title(f"{target}", fontsize=24, fontweight="bold", fontname="Arial")
            ax.set_xlabel(f"Mean of Reference and Predicted {target} (mmHg)", fontsize=24, fontname="Arial")
            ax.set_ylabel(f"Difference (Pred \u2212 Ref {target}) (mmHg)", fontsize=24, fontname="Arial")
            ax.tick_params(axis="both", which="both", direction="in", labelsize=24)
            ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved 300 DPI Bland-Altman plot to: {out_path}")


def main():
    search_dirs = [ANALYSIS_DIR, BASE_DIR, os.getcwd()]
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    def_path = find_file(DEFINITION_FILE, search_dirs)
    if def_path is None:
        raise FileNotFoundError(f"Could not locate definition file: {DEFINITION_FILE}")
    feat_mapping, relaxed_mapping, serial_offset = load_feature_definitions(def_path)
    print(f"Loaded feature definitions: {len(feat_mapping)} entries (serial offset = {serial_offset}).")

    selected_features_dict = {
        target: load_stage2_selected_features(filename, search_dirs)
        for target, filename in FEATURE_FILES.items()
    }
    for target, feature_names in selected_features_dict.items():
        print(f"[{target}] Extracted Stage 2 Representative Features: {len(feature_names)}")

    subject_data = {}
    for folder_name in sorted(os.listdir(BASE_DIR)):
        if folder_name in EXCLUDE_SUBJECTS:
            continue
        folder_path = os.path.join(BASE_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        csv_path = os.path.join(folder_path, "all_trimmed.csv")
        if not os.path.exists(csv_path):
            candidates = [
                name for name in os.listdir(folder_path)
                if name.startswith("all_trimmed") and name.endswith(".csv")
            ]
            if candidates:
                csv_path = os.path.join(folder_path, candidates[0])
            else:
                continue

        records = extract_subject_records(
            csv_path, selected_features_dict, feat_mapping, relaxed_mapping, serial_offset
        )
        if records and len(records) == 4:
            subject_data[folder_name] = records

    subjects = sorted(subject_data.keys())
    if len(subjects) < 5:
        raise ValueError("At least five valid subjects are required for 5-fold cross-validation")
    print(f"Successfully loaded {len(subjects)} subjects (each with 4 delta records).")

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    eval_configurations = [
        ("All_161_Features", "SBP", "full_features", "delta_sbp", "base_sbp", "cur_sbp"),
        ("Stage2_Selected", "SBP", "sbp_selected", "delta_sbp", "base_sbp", "cur_sbp"),
        ("All_161_Features", "DBP", "full_features", "delta_dbp", "base_dbp", "cur_dbp"),
        ("Stage2_Selected", "DBP", "dbp_selected", "delta_dbp", "base_dbp", "cur_dbp"),
        ("All_161_Features", "MAP", "full_features", "delta_map", "base_map", "cur_map"),
        ("Stage2_Selected", "MAP", "map_selected", "delta_map", "base_map", "cur_map"),
    ]

    fold_results = []
    sample_predictions = []
    n_train_samples = 0
    n_test_samples = 0

    for fold, (train_idx, test_idx) in enumerate(kf.split(subjects), 1):
        train_subs = [subjects[i] for i in train_idx]
        test_subs = [subjects[i] for i in test_idx]
        train_records = [record for subject in train_subs for record in subject_data[subject]]
        test_records_with_meta = [
            (subject, record) for subject in test_subs for record in subject_data[subject]
        ]
        test_records = [item[1] for item in test_records_with_meta]
        n_train_samples = len(train_records)
        n_test_samples = len(test_records)

        for feat_label, target_bp, feat_key, delta_key, base_key, abs_key in eval_configurations:
            X_train = np.array([record[feat_key] for record in train_records])
            y_train_delta = np.array([record[delta_key] for record in train_records])
            X_test = np.array([record[feat_key] for record in test_records])
            y_test_delta = np.array([record[delta_key] for record in test_records])
            base_test = np.array([record[base_key] for record in test_records])
            abs_test = np.array([record[abs_key] for record in test_records])

            reg = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=RANDOM_STATE)
            reg.fit(X_train, y_train_delta)
            y_pred_delta = reg.predict(X_test)
            y_pred_abs = base_test + y_pred_delta
            mae, rmse, r_delta, r2_delta = calculate_metrics(y_test_delta, y_pred_delta)
            _, _, r_abs, r2_abs = calculate_metrics(abs_test, y_pred_abs)

            error_abs = y_pred_abs - abs_test
            sd_error = (
                float(np.std(error_abs, ddof=1)) if len(error_abs) > 1 else 0.0
            )

            fold_results.append({
                "Fold": fold, "Input_Feature_Set": feat_label, "Target_BP": target_bp,
                "Feature_Count": X_train.shape[1], "MAE": round(mae, 4),
                "RMSE": round(rmse, 4), "Delta_r": round(r_delta, 4),
                "Delta_R2": round(r2_delta, 4), "Abs_r": round(r_abs, 4),
                "Abs_R2": round(r2_abs, 4),
                "BP_std": round(float(np.std(y_pred_abs)), 4),
                "BP_sd": round(sd_error, 4),
            })
            for (subject, record), dt_true, dt_pred, abs_true, abs_pred in zip(
                test_records_with_meta, y_test_delta, y_pred_delta, abs_test, y_pred_abs
            ):
                sample_predictions.append({
                    "Fold": fold, "Subject_ID": subject, "Record_Idx": record["record_idx"],
                    "Input_Feature_Set": feat_label, "Target_BP": target_bp,
                    "Baseline_BP": round(record[base_key], 2),
                    "True_Delta_BP": round(dt_true, 4), "Pred_Delta_BP": round(dt_pred, 4),
                    "True_BP": round(abs_true, 2), "Pred_BP": round(abs_pred, 2),
                    "Abs_Error": round(abs(abs_true - abs_pred), 4),
                })

    df_folds = pd.DataFrame(fold_results)
    df_samples = pd.DataFrame(sample_predictions)

    mae_stats = evaluate_mae_significance_and_ci(
        df_samples=df_samples,
        df_folds=df_folds,
        n_train=n_train_samples,
        n_test=n_test_samples,
        n_boot=1000,
        seed=RANDOM_STATE,
    )

    summary_list = []
    for (feat_label, target_bp), group in df_folds.groupby(["Input_Feature_Set", "Target_BP"], sort=False):
        t_stat_info = mae_stats[target_bp]
        summary_list.append({
            "Input_Feature_Set": feat_label, "Target_BP": target_bp,
            "Feature_Count": int(group["Feature_Count"].iloc[0]),
            "MAE_Mean": round(group["MAE"].mean(), 4), "MAE_Std": round(group["MAE"].std(), 4),
            "RMSE_Mean": round(group["RMSE"].mean(), 4), "RMSE_Std": round(group["RMSE"].std(), 4),
            "Delta_r_Mean": round(group["Delta_r"].mean(), 4), "Delta_r_Std": round(group["Delta_r"].std(), 4),
            "Delta_R2_Mean": round(group["Delta_R2"].mean(), 4), "Delta_R2_Std": round(group["Delta_R2"].std(), 4),
            "Abs_r_Mean": round(group["Abs_r"].mean(), 4), "Abs_r_Std": round(group["Abs_r"].std(), 4),
            "Abs_R2_Mean": round(group["Abs_R2"].mean(), 4), "Abs_R2_Std": round(group["Abs_R2"].std(), 4),
            "BP_std": round(group["BP_std"].mean(), 4), "BP_sd": round(group["BP_sd"].mean(), 4),
            "Shapiro_p": t_stat_info["Shapiro_p"],
            "Stat_Test": t_stat_info["Sample_Test"],
            "p_value": t_stat_info["p_value"],
            "Sig_Level": t_stat_info["Sample_Sig"],
            "Corrected_Resampled_t_p": t_stat_info["Corrected_t_p_value"],
            "Corrected_t_Sig": t_stat_info["Corrected_t_Sig"],
            "Delta_MAE_95CI": t_stat_info["Delta_MAE_95CI"],
        })
    df_summary = pd.DataFrame(summary_list)

    # Calculate within-subject Pearson r and R2, then average across all subjects
    subject_summary_list = []
    for (feat_label, target_bp), group in df_samples.groupby(
        ["Input_Feature_Set", "Target_BP"], sort=False
    ):
      sub_metrics = []
      for sub_id, sub_df in group.groupby("Subject_ID"):
        y_dt_true = sub_df["True_Delta_BP"].values.astype(float)
        y_dt_pred = sub_df["Pred_Delta_BP"].values.astype(float)
        y_abs_true = sub_df["True_BP"].values.astype(float)
        y_abs_pred = sub_df["Pred_BP"].values.astype(float)

        _, _, r_delta, r2_delta = calculate_metrics(y_dt_true, y_dt_pred)
        _, _, r_abs, r2_abs = calculate_metrics(y_abs_true, y_abs_pred)

        sub_metrics.append({
            "Delta_r": r_delta,
            "Delta_R2": r2_delta,
            "Abs_r": r_abs,
            "Abs_R2": r2_abs,
        })

      df_sub = pd.DataFrame(sub_metrics)
      subject_summary_list.append({
          "Input_Feature_Set": feat_label,
          "Target_BP": target_bp,
          "Delta_r_Mean": round(df_sub["Delta_r"].mean(), 4),
          "Delta_r_Std": round(df_sub["Delta_r"].std(), 4),
          "Delta_R2_Mean": round(df_sub["Delta_R2"].mean(), 4),
          "Delta_R2_Std": round(df_sub["Delta_R2"].std(), 4),
          "Abs_r_Mean": round(df_sub["Abs_r"].mean(), 4),
          "Abs_r_Std": round(df_sub["Abs_r"].std(), 4),
          "Abs_R2_Mean": round(df_sub["Abs_R2"].mean(), 4),
          "Abs_R2_Std": round(df_sub["Abs_R2"].std(), 4),
      })
    df_subject_summary = pd.DataFrame(subject_summary_list)

    # Evaluate naive baselines (Predicting zero change & Predicting per-posture mean)
    df_baseline_summary = evaluate_naive_baselines(subjects, subject_data, kf)

    output_fold_path = os.path.join(
        ANALYSIS_DIR, "cv_5fold_prediction_results.csv"
    )
    output_summary_path = os.path.join(
        ANALYSIS_DIR, "cv_5fold_summary_metrics.csv"
    )
    output_sample_path = os.path.join(
        ANALYSIS_DIR, "cv_sample_predictions_detail.csv"
    )

    df_folds.to_csv(output_fold_path, index=False, encoding="utf-8-sig") 
    df_samples.to_csv(output_sample_path, index=False, encoding="utf-8-sig")

    # Save summary metrics: fold-level first, then subject-level, followed by naive baselines
    with open(output_summary_path, "w", encoding="utf-8-sig") as f:
        df_summary.to_csv(f, index=False)
        f.write(
            "\n--------------------------------------------------------------------------------\n"
        )
        f.write(
            "# Within-Subject Averaged Metrics (Subject-Level Pearson r and"
            " R2)\n"
        )
        df_subject_summary.to_csv(f, index=False)
        f.write(
            "\n--------------------------------------------------------------------------------\n"
        )
        f.write(
            "# Naive Baselines (Predicting Zero Change & Predicting Per-Posture Mean)\n"
        )
        df_baseline_summary.to_csv(f, index=False)

    plot_bp_regression_scatters(df_samples, ANALYSIS_DIR)
    plot_bland_altman_figures(df_samples, ANALYSIS_DIR)
    print("All processing and plotting completed successfully.")


if __name__ == "__main__":
    main()