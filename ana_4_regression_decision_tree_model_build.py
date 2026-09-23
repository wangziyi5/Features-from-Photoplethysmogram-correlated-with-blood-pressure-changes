import io
import os
import random
import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

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

# Demographic 1-based column mapping to 0-based column indices in all_trimmed.csv
# Height: Col 6 -> 5, Weight: Col 7 -> 6, BMI: Col 8 -> 7, Gender: Col 9 -> 8, Age: Col 10 -> 9
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
    """
    Map feature name to serial number (ID) from supplement_feature_definitions.xlsx.
    The workbook has no header: column 1 is serial number, column 2 is feature name.
    """
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

    # Determine offset: 1-based serial number n maps to 0-based index n + 9 (Col n + 10 in CSV)
    min_serial = min(serial_numbers) if serial_numbers else 1
    serial_offset = 9 if min_serial >= 1 else 10

    return name_to_id, relaxed_to_id, serial_offset


def load_stage2_selected_features(file_name, search_dirs):
    """
    Extract feature names under 'STAGE 2 FINAL REPRESENTATIVE FEATURES' section.
    Locates the section block, parses the table header, and extracts the 'feature_name' column.
    """
    file_path = find_file(file_name, search_dirs)
    if file_path is None:
        raise FileNotFoundError(f"Could not locate feature file: {file_name}")

    # Read raw lines with multi-encoding support
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

    # Scan for section header containing 'STAGE 2' and 'REPRESENTATIVE'
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
            # Skip separator lines (e.g., '# =====')
            if stripped.startswith("#"):
                if set(stripped.replace("#", "").strip()) <= {"=", "-", "_"}:
                    continue
                # Stop if another major section begins
                if "STAGE" in stripped.upper() or "SUMMARY" in stripped.upper():
                    break
                continue
            table_lines.append(line)

        if table_lines:
            df_stage2 = pd.read_csv(io.StringIO("".join(table_lines)))
            # Strictly locate the 'feature_name' column (never cluster_id)
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

    # Fallback: single-table CSV format with feature_name column
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

    # Demographic variables map directly to static indices (5 to 9)
    if feature_key.upper() in DEMO_COL_MAP_0BASED:
        return DEMO_COL_MAP_0BASED[feature_key.upper()], True

    # Check mapping from supplement_feature_definitions.xlsx
    if feature_key in feat_mapping:
        serial_id = feat_mapping[feature_key]
        return serial_id + serial_offset, False

    if relaxed_key in relaxed_mapping:
        serial_id = relaxed_mapping[relaxed_key]
        return serial_id + serial_offset, False

    # Check direct column name matches in all_trimmed.csv
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

    # Demographic columns 6-10 (0-based indices 5:10: Height, Weight, BMI, Gender, Age)
    demo_indices = [5, 6, 7, 8, 9]
    signal_indices = list(range(10, 166))
    df_clean = df.copy()

    # Map column positions to actual column names to prevent index-string confusion
    demo_cols = [df_clean.columns[i] for i in demo_indices]
    signal_cols = [df_clean.columns[i] for i in signal_indices]
    sbp_col, dbp_col = identify_bp_columns(df_clean)

    # Encode Gender via column name assignment to safely cast int64 column to float64
    gender_col = df_clean.columns[8]
    df_clean[gender_col] = (
        df_clean[gender_col].apply(encode_gender).astype(float)
    )

    # Safely convert demographic (preserving Age decimals), signal, and BP columns to float
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

    # Static demographic vector replicated across delta records
    static_demo = df_clean.iloc[0, demo_indices].values.astype(float)

    # Baseline values (row 0)
    base_signal = df_clean.iloc[0, signal_indices].values.astype(float)
    base_sbp = float(df_clean.iloc[0][sbp_col])
    base_dbp = float(df_clean.iloc[0][dbp_col])
    # Compute MAP strictly according to MAP = (SBP + 2 * DBP) / 3
    base_map = (base_sbp + 2.0 * base_dbp) / 3.0

    col_names = list(df_clean.columns)
    target_feat_indices = resolve_selected_feature_indices(
        selected_features_dict, feat_mapping, relaxed_mapping, serial_offset, col_names
    )

    records = []
    # Generate 4 delta records from rows 1 to 4
    for idx in range(1, 5):
        cur_row = df_clean.iloc[idx]
        cur_sbp = float(cur_row[sbp_col])
        cur_dbp = float(cur_row[dbp_col])
        # Compute MAP strictly according to MAP = (SBP + 2 * DBP) / 3
        cur_map = (cur_sbp + 2.0 * cur_dbp) / 3.0

        delta_sbp = cur_sbp - base_sbp
        delta_dbp = cur_dbp - base_dbp
        delta_map = cur_map - base_map

        # Full 161 features: 156 delta physiological features + 5 static demographic features
        cur_signal = cur_row.iloc[signal_indices].values.astype(float)
        delta_signal = cur_signal - base_signal
        full_feat = np.concatenate([delta_signal, static_demo])

        # Selected features: delta for physiological, static for demographic
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


def plot_bp_regression_scatters(df_samples, output_dir):
  """Generate high-resolution (300 DPI) regression scatter plots for SBP, DBP, and MAP.

  Left subplot: All 161 Features; Right subplot: Stage 2 Selected Features.
  """
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

    # Determine unified axis limits for both subplots to maintain equal aspect ratio
    val_min = min(sub_df["True_BP"].min(), sub_df["Pred_BP"].min()) - 5.0
    val_max = max(sub_df["True_BP"].max(), sub_df["Pred_BP"].max()) + 5.0

    for ax, (feat_key, feat_title) in zip(axes, feature_subsets):
      data = sub_df[sub_df["Input_Feature_Set"] == feat_key]
      y_true = data["True_BP"].values.astype(float)
      y_pred = data["Pred_BP"].values.astype(float)

      mae, rmse, r, r2 = calculate_metrics(y_true, y_pred)

      # Scatter plot of True BP vs Predicted BP
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

      # 45-degree identity reference line (y = x)
      lims = [val_min, val_max]
      ax.plot(
          lims,
          lims,
          "k--",
          alpha=0.75,
          linewidth=1.5,
          label="Identity line (y = x)",
      )

      # Linear regression trendline
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

      # Metric annotation text box
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

    # Sort subjects to guarantee deterministic fold splits
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
    for fold, (train_idx, test_idx) in enumerate(kf.split(subjects), 1):
        train_subs = [subjects[i] for i in train_idx]
        test_subs = [subjects[i] for i in test_idx]
        train_records = [record for subject in train_subs for record in subject_data[subject]]
        test_records_with_meta = [
            (subject, record) for subject in test_subs for record in subject_data[subject]
        ]
        test_records = [item[1] for item in test_records_with_meta]

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
    summary_list = []
    for (feat_label, target_bp), group in df_folds.groupby(["Input_Feature_Set", "Target_BP"], sort=False):
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

    # Save summary metrics: fold-level first, then a separator line, followed by subject-level metrics
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

    # Generate and save 300 DPI regression scatter plots for SBP, DBP, and MAP
    plot_bp_regression_scatters(df_samples, ANALYSIS_DIR)

    print(f"\nSaved fold metrics to: {output_fold_path}")
    print(f"Saved summary metrics to: {output_summary_path}")
    print(f"Saved prediction details to: {output_sample_path}")
    print("\n--- 5-Fold Cross Validation Summary (Fold-Level) ---")
    print(df_summary.to_string(index=False))
    print(
        "\n--------------------------------------------------------------------------------"
    )
    print("--- Within-Subject Averaged Metrics Summary (Subject-Level) ---")
    print(df_subject_summary.to_string(index=False))


if __name__ == "__main__":
    main()