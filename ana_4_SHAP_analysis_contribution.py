import argparse
import os

# Fixed hash seed for reproducibility. Must be set before importing libraries.
os.environ.setdefault("PYTHONHASHSEED", "42")

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


# =============================================================================
# Reproducibility
# =============================================================================
RANDOM_STATE = 42
N_FOLDS = 5
N_ESTIMATORS = 200
N_JOBS = 1  # strict reproducibility


# =============================================================================
# Dataset / feature definition
# =============================================================================
EXCLUDED = {
    "yy04_subject_000",
    "yy04_subject_001",
    "yy04_subject_002",
    "yy04_subject_003",
    "yy04_subject_004",
}

N_MORPH_FEATURES = 156
DEMOGRAPHIC_FEATURE_IDS = ["s1", "s2", "s3", "s4", "s5"]
DEMOGRAPHIC_FEATURE_NAMES = [
    "Gender",
    "Age",
    "Height",
    "Weight",
    "BMI",
]

TOTAL_FEATURES = N_MORPH_FEATURES + len(DEMOGRAPHIC_FEATURE_IDS)
N_BASELINE_ROWS = 5
N_DELTA_SAMPLES_PER_SUBJECT = 4

# all_trimmed.csv is 166 columns in total:
#   1-5   : SBP, DBP, HR, HR_cal, HRV_cal
#   6     : Height
#   7     : Weight
#   8     : BMI
#   9     : Gender
#   10    : Age
#   11-166: 156 morphological/features columns numbered 1-156
#
# Python iloc is 0-based:
MORPH_START_COL_0BASED = 10
MORPH_END_COL_0BASED_EXCLUSIVE = 166

DEMOGRAPHIC_COLUMN_NAMES = {
    "Gender": "Gender",
    "Age": "Age",
    "Height": "Height",
    "Weight": "Weight",
    "BMI": "BMI",
}


# =============================================================================
# Feature categories
# =============================================================================
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
        elif 155 <= feat_idx <= 156:
            return "Contour"

    return "Unknown"


# =============================================================================
# Excel feature mapping
# =============================================================================
def load_feature_name_mapping(
    excel_path: Path,
    num_features: int = N_MORPH_FEATURES,
):
    """
    Load the 156 morphological feature names.

    Supported Excel formats:
      A) 156 x 1: feature names in feature order 1..156
      B) 156 x 2+: first column = feature index, second column = feature name
    """
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Feature definition Excel file not found: {excel_path}"
        )

    df_def = pd.read_excel(excel_path, header=None)

    if len(df_def) != num_features:
        raise ValueError(
            f"Excel feature definition contains {len(df_def)} rows; "
            f"expected exactly {num_features}."
        )

    if df_def.shape[1] == 1:
        names = [
            str(df_def.iloc[i, 0]).strip()
            for i in range(num_features)
        ]
        excel_format = "156 x 1 (feature names only)"

    elif df_def.shape[1] >= 2:
        excel_indices = []
        names = []

        for i in range(num_features):
            try:
                idx = int(df_def.iloc[i, 0])
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid feature index at Excel row {i + 1}: "
                    f"{df_def.iloc[i, 0]!r}"
                )

            excel_indices.append(idx)
            names.append(str(df_def.iloc[i, 1]).strip())

        excel_format = (
            f"{df_def.shape[0]} x {df_def.shape[1]} "
            "(index + feature name)"
        )

        expected_indices = list(range(1, num_features + 1))
        if excel_indices != expected_indices:
            raise ValueError(
                "Excel feature indices are not exactly 1..156 in order. "
                f"Observed: {excel_indices[:5]} ... {excel_indices[-5:]}"
            )

    else:
        raise ValueError("Excel file has no usable columns.")

    if any(name == "" or name.lower() == "nan" for name in names):
        raise ValueError("At least one Excel feature name is empty.")

    return {
        i + 1: names[i]
        for i in range(num_features)
    }, excel_format


# =============================================================================
# CSV mapping validation
# =============================================================================
def inspect_csv_feature_columns(
    df: pd.DataFrame,
    feature_names: list[str],
):
    """
    Verify the actual all_trimmed.csv layout:
      Gender = column 9
      Age    = column 10
      Height = column 6
      Weight = column 7
      BMI    = column 8
      Morphological features 1..156 = columns 11..166

    No CSV column numbers are written to the final feature summary tables.
    They are only used internally for validation/loading.
    """
    if df.shape[1] < 166:
        raise ValueError(
            f"CSV has only {df.shape[1]} columns; at least 166 are required."
        )

    # Verify demographic headers by position where possible.
    expected_demo_headers = {
        6: "Height",
        7: "Weight",
        8: "BMI",
        9: "Gender",
        10: "Age",
    }

    observed_demo_headers = {}
    for col_1based, expected_name in expected_demo_headers.items():
        observed = str(df.columns[col_1based - 1]).strip()
        observed_demo_headers[col_1based] = observed

    demo_header_matches = all(
        observed_demo_headers[pos] == expected_name
        for pos, expected_name in expected_demo_headers.items()
    )

    # Verify morphological feature block headers.
    selected_headers = [
        str(c).strip()
        for c in df.columns[
            MORPH_START_COL_0BASED:
            MORPH_END_COL_0BASED_EXCLUSIVE
        ]
    ]

    if len(selected_headers) != N_MORPH_FEATURES:
        raise ValueError(
            f"Morphological feature block contains {len(selected_headers)} "
            f"columns instead of {N_MORPH_FEATURES}."
        )

    expected_index_headers = [
        str(i)
        for i in range(1, N_MORPH_FEATURES + 1)
    ]

    index_header_match = selected_headers == expected_index_headers

    name_header_match = selected_headers == feature_names

    numeric_header_match = False
    try:
        numeric_headers = [int(float(h)) for h in selected_headers]
        numeric_header_match = (
            numeric_headers == list(range(1, N_MORPH_FEATURES + 1))
        )
    except (ValueError, TypeError):
        pass

    return {
        "total_csv_columns": int(df.shape[1]),
        "demographic_header_matches": demo_header_matches,
        "observed_demographic_headers": observed_demo_headers,
        "morph_index_header_matches": bool(
            index_header_match or numeric_header_match
        ),
        "morph_excel_name_header_matches": bool(name_header_match),
        "selected_morph_feature_count": len(selected_headers),
    }


# =============================================================================
# Target calculation
# =============================================================================
def calculate_target_values(df: pd.DataFrame, target: str) -> np.ndarray:
    """
    Return the first five reference BP values.

    SBP / DBP:
        directly use the corresponding CSV column.

    MAP:
        MAP = (1/3) * SBP + (2/3) * DBP
        calculated separately for each of the first five rows, then delta
        is calculated in the same way as SBP/DBP.
    """
    if target == "SBP":
        if "SBP" not in df.columns:
            raise ValueError("SBP column is missing from all_trimmed.csv.")
        return df["SBP"].iloc[:5].to_numpy(dtype=float)

    if target == "DBP":
        if "DBP" not in df.columns:
            raise ValueError("DBP column is missing from all_trimmed.csv.")
        return df["DBP"].iloc[:5].to_numpy(dtype=float)

    if target == "MAP":
        if "SBP" not in df.columns or "DBP" not in df.columns:
            raise ValueError(
                "Both SBP and DBP columns are required to calculate MAP."
            )

        sbp = df["SBP"].iloc[:5].to_numpy(dtype=float)
        dbp = df["DBP"].iloc[:5].to_numpy(dtype=float)

        # User-specified formula.
        return (1.0 / 3.0) * sbp + (2.0 / 3.0) * dbp

    raise ValueError(f"Unsupported target: {target}")


# =============================================================================
# Data loading
# =============================================================================
def load_all_subject_delta_matrices(
    root_dir: Path,
    target: str,
    feature_names: list[str],
):
    """
    Construct the final feature matrix.

    For each subject:
      1. Calculate 156 morphological delta features:
             feature[rows 1:5] - feature[row 0]
      2. Extract subject-level demographics from the baseline row:
             Height, Weight, BMI, Gender, Age
      3. Repeat the 5 demographic values four times using np.tile.
      4. Horizontally concatenate:
             [156 delta morphology | 5 repeated demographics]
         => 161 features per delta sample.

    Each subject therefore contributes:
        4 samples x 161 features
    """
    subject_dirs = sorted(
        [
            p
            for p in root_dir.glob("yy04_subject_*")
            if p.is_dir() and p.name not in EXCLUDED
        ]
    )

    all_X = []
    all_y = []
    loaded_subjects = []
    csv_mapping_checks = []

    demographic_source_names = [
        "Gender",
        "Age",
        "Height",
        "Weight",
        "BMI",
    ]

    # Final feature order:
    #   1..156 = morphological delta features
    #   s1..s5  = demographics
    if len(feature_names) != N_MORPH_FEATURES:
        raise ValueError(
            f"Expected {N_MORPH_FEATURES} morphological feature names."
        )

    for s_dir in subject_dirs:
        csv_file = s_dir / "all_trimmed.csv"

        if not csv_file.exists():
            continue

        try:
            df = pd.read_csv(csv_file)
        except Exception as exc:
            print(f"WARNING: failed to read {csv_file}: {exc}")
            continue

        if len(df) < N_BASELINE_ROWS:
            continue

        # Validate CSV structure.
        try:
            mapping_check = inspect_csv_feature_columns(
                df,
                feature_names,
            )
        except ValueError as exc:
            print(
                f"WARNING: {s_dir.name} failed CSV layout validation: {exc}"
            )
            continue

        csv_mapping_checks.append(
            (s_dir.name, mapping_check)
        )

        # -------------------------------------------------------------
        # BP target
        # -------------------------------------------------------------
        try:
            bp_vals = calculate_target_values(
                df,
                target,
            )
        except (ValueError, TypeError) as exc:
            print(
                f"WARNING: {s_dir.name} failed target calculation: {exc}"
            )
            continue

        delta_bp = bp_vals[1:5] - bp_vals[0]

        # -------------------------------------------------------------
        # 156 morphological features
        # -------------------------------------------------------------
        try:
            feat_matrix = df.iloc[
                :5,
                MORPH_START_COL_0BASED:MORPH_END_COL_0BASED_EXCLUSIVE,
            ].to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            print(
                f"WARNING: invalid morphological values in {s_dir.name}: "
                f"{exc}"
            )
            continue

        if feat_matrix.shape != (
            N_BASELINE_ROWS,
            N_MORPH_FEATURES,
        ):
            print(
                f"WARNING: {s_dir.name} feature matrix has shape "
                f"{feat_matrix.shape}; expected "
                f"({N_BASELINE_ROWS}, {N_MORPH_FEATURES}); skipped."
            )
            continue

        delta_feat = (
            feat_matrix[1:5, :]
            - feat_matrix[0, :]
        )

        # -------------------------------------------------------------
        # 5 demographics
        # -------------------------------------------------------------
        try:
            demographic_values = df.loc[
                df.index[0],
                demographic_source_names,
            ].to_numpy(dtype=float)
        except (ValueError, TypeError, KeyError) as exc:
            print(
                f"WARNING: invalid demographic values in {s_dir.name}: "
                f"{exc}"
            )
            continue

        # User requested order:
        # s1 = Gender
        # s2 = Age
        # s3 = Height
        # s4 = Weight
        # s5 = BMI
        #
        # df columns themselves are in positions:
        # Height, Weight, BMI, Gender, Age
        # so explicitly construct the requested feature order.
        try:
            demographics = np.array(
                [
                    float(df["Gender"].iloc[0]),
                    float(df["Age"].iloc[0]),
                    float(df["Height"].iloc[0]),
                    float(df["Weight"].iloc[0]),
                    float(df["BMI"].iloc[0]),
                ],
                dtype=float,
            )
        except (ValueError, TypeError, KeyError) as exc:
            print(
                f"WARNING: invalid requested demographic fields in "
                f"{s_dir.name}: {exc}"
            )
            continue

        # Optional validation of Gender coding.
        if demographics[0] not in (0.0, 1.0):
            print(
                f"WARNING: {s_dir.name} Gender={demographics[0]} is not "
                "coded as 0/1; subject skipped."
            )
            continue

        # Repeat subject-level demographics 4 times.
        demographic_matrix = np.tile(
            demographics,
            (N_DELTA_SAMPLES_PER_SUBJECT, 1),
        )

        # Final 161-dimensional feature matrix.
        X_subject = np.hstack(
            [
                delta_feat,
                demographic_matrix,
            ]
        )

        if X_subject.shape != (
            N_DELTA_SAMPLES_PER_SUBJECT,
            TOTAL_FEATURES,
        ):
            raise RuntimeError(
                f"{s_dir.name}: final subject feature matrix shape is "
                f"{X_subject.shape}; expected "
                f"({N_DELTA_SAMPLES_PER_SUBJECT}, {TOTAL_FEATURES})."
            )

        if not np.isfinite(X_subject).all():
            print(
                f"WARNING: non-finite final feature values in "
                f"{s_dir.name}; skipped."
            )
            continue

        if not np.isfinite(delta_bp).all():
            print(
                f"WARNING: non-finite {target} values in "
                f"{s_dir.name}; skipped."
            )
            continue

        all_X.append(X_subject)
        all_y.append(delta_bp)
        loaded_subjects.append(s_dir.name)

    if not all_X:
        raise RuntimeError(
            f"No valid subject data found under: {root_dir.resolve()}"
        )

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    expected_samples = (
        len(loaded_subjects)
        * N_DELTA_SAMPLES_PER_SUBJECT
    )

    if len(y_all) != expected_samples:
        raise RuntimeError(
            f"Unexpected sample count: {len(y_all)}; "
            f"expected {expected_samples}."
        )

    if len(set(loaded_subjects)) != len(loaded_subjects):
        raise RuntimeError("Duplicate subject IDs detected.")

    mapping_summary = {
        "num_checked_csvs": len(csv_mapping_checks),
        "all_demo_headers_match": (
            all(
                item[1]["demographic_header_matches"]
                for item in csv_mapping_checks
            )
            if csv_mapping_checks
            else False
        ),
        "all_morph_index_headers_match": (
            all(
                item[1]["morph_index_header_matches"]
                for item in csv_mapping_checks
            )
            if csv_mapping_checks
            else False
        ),
        "all_morph_excel_names_match": (
            all(
                item[1]["morph_excel_name_header_matches"]
                for item in csv_mapping_checks
            )
            if csv_mapping_checks
            else False
        ),
        "first_checked_subject": (
            csv_mapping_checks[0][0]
            if csv_mapping_checks
            else "N/A"
        ),
        "first_mapping_check": (
            csv_mapping_checks[0][1]
            if csv_mapping_checks
            else None
        ),
    }

    return (
        X_all,
        y_all,
        loaded_subjects,
        mapping_summary,
    )


# =============================================================================
# Statistics and ranking
# =============================================================================
def safe_corr(func, x, y):
    """Return correlation coefficient or NaN if undefined."""
    try:
        return float(func(x, y).statistic)
    except Exception:
        return np.nan


def normalize_ranks_descending(values: np.ndarray) -> np.ndarray:
    """
    Highest SHAP contribution -> 1.0
    Lowest SHAP contribution  -> 0.0
    """
    n_features = len(values)

    if n_features == 1:
        return np.ones(1, dtype=float)

    order = np.argsort(
        -values,
        kind="mergesort",
    )

    coeff = np.empty(
        n_features,
        dtype=float,
    )

    coeff[order] = 1.0 - (
        np.arange(n_features, dtype=float)
        / (n_features - 1)
    )

    return coeff


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Finnegan-style subject-level 5-fold SHAP analysis with "
            "156 delta morphological features + 5 demographic features."
        )
    )

    parser.add_argument(
        "target",
        nargs="?",
        default="SBP",
        choices=["SBP", "DBP", "MAP"],
        help="Target: SBP, DBP, or MAP.",
    )

    parser.add_argument(
        "--excel_defs",
        "-e",
        default="supplement_feature_definitions.xlsx",
        help="156-feature definition Excel file.",
    )

    parser.add_argument(
        "--output_dir",
        "-d",
        default="/Users/ziyiwang/Documents/Project_2026_month1_6/"
                "Work_260223_continuous_BP_monitoring/OUTPUT",
        help="Root directory containing yy04_subject_* folders.",
    )

    parser.add_argument(
        "--out_txt",
        "-o",
        default=None,
        help="Output TXT report filename.",
    )

    parser.add_argument(
        "--n_folds",
        type=int,
        default=N_FOLDS,
        help="Subject-level number of CV folds.",
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=N_ESTIMATORS,
        help="Number of Random Forest trees.",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=RANDOM_STATE,
        help="Fixed random seed.",
    )

    args = parser.parse_args()

    if args.n_folds != 5:
        raise ValueError(
            "This script is configured for the requested 5-fold analysis. "
            "Please use --n_folds 5."
        )

    target = args.target.upper()
    root_dir = Path(args.output_dir)
    excel_path = Path(args.excel_defs)

    out_txt_path = (
        Path(args.out_txt)
        if args.out_txt
        else Path(
            f"{target}_161_features_"
            f"SHAP_analysis_5fold_CV_Finnegan.txt"
        )
    )

    # -------------------------------------------------------------------------
    # 1. Load 156 morphological feature names from Excel
    # -------------------------------------------------------------------------
    (
        feat_name_dict,
        excel_format,
    ) = load_feature_name_mapping(
        excel_path,
        num_features=N_MORPH_FEATURES,
    )

    morph_feature_names = [
        feat_name_dict[i]
        for i in range(1, N_MORPH_FEATURES + 1)
    ]

    # Final feature IDs and names.
    feature_ids = (
        list(range(1, N_MORPH_FEATURES + 1))
        + DEMOGRAPHIC_FEATURE_IDS
    )

    feature_names = (
        morph_feature_names
        + DEMOGRAPHIC_FEATURE_NAMES
    )

    feature_categories = [
        get_feature_category(fid)
        for fid in feature_ids
    ]

    if len(feature_ids) != TOTAL_FEATURES:
        raise RuntimeError("Final feature ID count is not 161.")

    if len(feature_names) != TOTAL_FEATURES:
        raise RuntimeError("Final feature-name count is not 161.")

    # -------------------------------------------------------------------------
    # 2. Load 70 subjects and construct 161-dimensional feature matrix
    # -------------------------------------------------------------------------
    print(f"\nLoading data from: {root_dir.resolve()}")
    print(f"Feature definitions: {excel_path.resolve()}")

    (
        X_all,
        y_all,
        subject_list,
        csv_mapping_summary,
    ) = load_all_subject_delta_matrices(
        root_dir=root_dir,
        target=target,
        feature_names=morph_feature_names,
    )

    n_subjects = len(subject_list)

    print(
        f"Loaded {n_subjects} subjects. "
        f"Final feature matrix X={X_all.shape}, y={y_all.shape}"
    )

    if n_subjects != 70:
        raise RuntimeError(
            f"Expected exactly 70 valid subjects after exclusions, "
            f"but found {n_subjects}."
        )

    if X_all.shape[1] != TOTAL_FEATURES:
        raise RuntimeError(
            f"Expected {TOTAL_FEATURES} features but obtained "
            f"{X_all.shape[1]}."
        )

    # -------------------------------------------------------------------------
    # 3. Subject-level 5-fold CV
    # -------------------------------------------------------------------------
    subject_ids = np.arange(n_subjects)

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=args.random_state,
    )

    # Four samples from one subject always have the same group.
    groups = np.repeat(
        subject_ids,
        N_DELTA_SAMPLES_PER_SUBJECT,
    )

    fold_mean_abs_shap = np.zeros(
        (5, TOTAL_FEATURES),
        dtype=float,
    )
    fold_mean_signed_shap = np.zeros(
        (5, TOTAL_FEATURES),
        dtype=float,
    )
    fold_rank_coefficient = np.zeros(
        (5, TOTAL_FEATURES),
        dtype=float,
    )
    fold_rank = np.zeros(
        (5, TOTAL_FEATURES),
        dtype=int,
    )

    fold_metrics = []
    fold_feature_records = []

    # Kept for consistency with the previous script:
    # OOF prediction and OOF SHAP are based on the unseen test samples.
    oof_pred = np.full(
        len(y_all),
        np.nan,
        dtype=float,
    )
    oof_fold = np.full(
        len(y_all),
        -1,
        dtype=int,
    )
    oof_shap = np.zeros_like(X_all)

    print("\nStarting subject-level 5-fold CV...")

    for fold_no, (
        train_sub_idx,
        test_sub_idx,
    ) in enumerate(
        kfold.split(subject_ids),
        start=1,
    ):
        train_subjects = subject_ids[train_sub_idx]
        test_subjects = subject_ids[test_sub_idx]

        train_mask = np.isin(
            groups,
            train_subjects,
        )
        test_mask = np.isin(
            groups,
            test_subjects,
        )

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]

        X_test = X_all[test_mask]
        y_test = y_all[test_mask]

        # ---------------------------------------------------------------------
        # 3A. Train Random Forest using only the training subjects
        # ---------------------------------------------------------------------
        rf_model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            n_jobs=N_JOBS,
        )

        rf_model.fit(
            X_train,
            y_train,
        )

        # ---------------------------------------------------------------------
        # 3B. PART A: model performance on unseen test subjects
        # ---------------------------------------------------------------------
        y_test_pred = rf_model.predict(
            X_test
        )

        oof_pred[test_mask] = y_test_pred
        oof_fold[test_mask] = fold_no

        mae = mean_absolute_error(
            y_test,
            y_test_pred,
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    y_test_pred,
                )
            )
        )

        r2 = r2_score(
            y_test,
            y_test_pred,
        )

        pearson_r = safe_corr(
            pearsonr,
            y_test,
            y_test_pred,
        )

        spearman_rho = safe_corr(
            spearmanr,
            y_test,
            y_test_pred,
        )

        bias = float(
            np.mean(y_test_pred - y_test)
        )

        fold_metrics.append(
            {
                "fold": fold_no,
                "train_subjects": len(train_subjects),
                "test_subjects": len(test_subjects),
                "train_samples": len(y_train),
                "test_samples": len(y_test),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "Pearson_r": pearson_r,
                "Spearman_rho": spearman_rho,
                "Bias_pred_minus_true": bias,
                "test_subject_names": ";".join(
                    subject_list[i]
                    for i in test_subjects
                ),
            }
        )

        # ---------------------------------------------------------------------
        # 3C. PART B: SHAP feature importance on TRAINING data
        #     This follows the Finnegan-style analysis.
        # ---------------------------------------------------------------------
        print(
            f"Fold {fold_no}: "
            f"{len(train_subjects)} train subjects, "
            f"{len(test_subjects)} test subjects; "
            f"TreeSHAP on training data..."
        )

        explainer = shap.TreeExplainer(
            rf_model
        )

        # Kept for compatibility with the previous script's OOF SHAP output.
        oof_shap[test_mask] = np.asarray(
            explainer.shap_values(X_test)
        )

        # Main SHAP importance is calculated on training data.
        shap_values = np.asarray(
            explainer.shap_values(X_train)
        )

        if shap_values.shape != X_train.shape:
            raise RuntimeError(
                f"Fold {fold_no}: SHAP shape {shap_values.shape} "
                f"does not match X_train shape {X_train.shape}."
            )

        mean_abs_shap = np.mean(
            np.abs(shap_values),
            axis=0,
        )

        mean_signed_shap = np.mean(
            shap_values,
            axis=0,
        )

        order = np.argsort(
            -mean_abs_shap,
            kind="mergesort",
        )

        ranks = np.empty(
            TOTAL_FEATURES,
            dtype=int,
        )

        ranks[order] = np.arange(
            1,
            TOTAL_FEATURES + 1,
        )

        rank_coeff = normalize_ranks_descending(
            mean_abs_shap
        )

        fold_mean_abs_shap[
            fold_no - 1,
            :
        ] = mean_abs_shap

        fold_mean_signed_shap[
            fold_no - 1,
            :
        ] = mean_signed_shap

        fold_rank[
            fold_no - 1,
            :
        ] = ranks

        fold_rank_coefficient[
            fold_no - 1,
            :
        ] = rank_coeff

        for i, feature_id in enumerate(feature_ids):
            fold_feature_records.append(
                {
                    "fold": fold_no,
                    "feature_index": feature_id,
                    "feature_name": feature_names[i],
                    "category": feature_categories[i],
                    "mean_abs_shap": mean_abs_shap[i],
                    "mean_signed_shap": mean_signed_shap[i],
                    "rank": int(ranks[i]),
                    "rank_coefficient": rank_coeff[i],
                }
            )

    # -------------------------------------------------------------------------
    # 4. Final 5-fold SHAP summary
    # -------------------------------------------------------------------------
    mean_abs_shap_cv = np.mean(
        fold_mean_abs_shap,
        axis=0,
    )

    std_abs_shap_cv = np.std(
        fold_mean_abs_shap,
        axis=0,
        ddof=1,
    )

    mean_signed_shap_cv = np.mean(
        fold_mean_signed_shap,
        axis=0,
    )

    mean_rank_coeff_cv = np.mean(
        fold_rank_coefficient,
        axis=0,
    )

    median_rank_coeff_cv = np.median(
        fold_rank_coefficient,
        axis=0,
    )

    min_rank_coeff_cv = np.min(
        fold_rank_coefficient,
        axis=0,
    )

    max_rank_coeff_cv = np.max(
        fold_rank_coefficient,
        axis=0,
    )

    mean_rank_cv = np.mean(
        fold_rank,
        axis=0,
    )

    median_rank_cv = np.median(
        fold_rank,
        axis=0,
    )

    final_records = []

    for i, feature_id in enumerate(feature_ids):
        final_records.append(
            {
                "feature_index": feature_id,
                "feature_name": feature_names[i],
                "category": feature_categories[i],
                "mean_abs_shap_5fold": mean_abs_shap_cv[i],
                "std_abs_shap_5fold": std_abs_shap_cv[i],
                "mean_signed_shap_5fold": mean_signed_shap_cv[i],
                "mean_rank_coefficient": mean_rank_coeff_cv[i],
                "median_rank_coefficient": median_rank_coeff_cv[i],
                "min_rank_coefficient": min_rank_coeff_cv[i],
                "max_rank_coefficient": max_rank_coeff_cv[i],
                "mean_rank": mean_rank_cv[i],
                "median_rank": median_rank_cv[i],
                "fold_1_rank": int(fold_rank[0, i]),
                "fold_2_rank": int(fold_rank[1, i]),
                "fold_3_rank": int(fold_rank[2, i]),
                "fold_4_rank": int(fold_rank[3, i]),
                "fold_5_rank": int(fold_rank[4, i]),
            }
        )

    df_shap = pd.DataFrame(
        final_records
    ).sort_values(
        by="mean_abs_shap_5fold",
        ascending=False,
    ).reset_index(drop=True)

    df_shap["final_rank"] = np.arange(
        1,
        TOTAL_FEATURES + 1,
    )

    # -------------------------------------------------------------------------
    # 5. Model performance summary
    # -------------------------------------------------------------------------
    df_metrics = pd.DataFrame(
        fold_metrics
    )

    metric_summary = {
        "MAE_mean": df_metrics["MAE"].mean(),
        "MAE_std": df_metrics["MAE"].std(ddof=1),
        "MAE_median": df_metrics["MAE"].median(),
        "MAE_IQR": (
            df_metrics["MAE"].quantile(0.75)
            - df_metrics["MAE"].quantile(0.25)
        ),
        "RMSE_mean": df_metrics["RMSE"].mean(),
        "RMSE_std": df_metrics["RMSE"].std(ddof=1),
        "RMSE_median": df_metrics["RMSE"].median(),
        "RMSE_IQR": (
            df_metrics["RMSE"].quantile(0.75)
            - df_metrics["RMSE"].quantile(0.25)
        ),
        "R2_mean": df_metrics["R2"].mean(),
        "R2_std": df_metrics["R2"].std(ddof=1),
        "R2_median": df_metrics["R2"].median(),
        "R2_IQR": (
            df_metrics["R2"].quantile(0.75)
            - df_metrics["R2"].quantile(0.25)
        ),
        "Pearson_mean": df_metrics["Pearson_r"].mean(),
        "Pearson_std": df_metrics["Pearson_r"].std(ddof=1),
        "Pearson_median": df_metrics["Pearson_r"].median(),
        "Pearson_IQR": (
            df_metrics["Pearson_r"].quantile(0.75)
            - df_metrics["Pearson_r"].quantile(0.25)
        ),
        "Spearman_mean": df_metrics["Spearman_rho"].mean(),
        "Spearman_std": df_metrics["Spearman_rho"].std(ddof=1),
        "Spearman_median": df_metrics["Spearman_rho"].median(),
        "Spearman_IQR": (
            df_metrics["Spearman_rho"].quantile(0.75)
            - df_metrics["Spearman_rho"].quantile(0.25)
        ),
        "Bias_mean": df_metrics["Bias_pred_minus_true"].mean(),
        "Bias_std": df_metrics["Bias_pred_minus_true"].std(ddof=1),
    }

    if np.isnan(oof_pred).any():
        raise RuntimeError(
            "Not every sample received an out-of-fold prediction."
        )

    oof_mae = mean_absolute_error(
        y_all,
        oof_pred,
    )

    oof_rmse = float(
        np.sqrt(
            mean_squared_error(
                y_all,
                oof_pred,
            )
        )
    )

    oof_r2 = r2_score(
        y_all,
        oof_pred,
    )

    oof_pearson = safe_corr(
        pearsonr,
        y_all,
        oof_pred,
    )

    oof_spearman = safe_corr(
        spearmanr,
        y_all,
        oof_pred,
    )

    oof_bias = float(
        np.mean(
            oof_pred - y_all
        )
    )

    # -------------------------------------------------------------------------
    # 6. Category-wise SHAP summary
    # -------------------------------------------------------------------------
    category_summary = (
        df_shap
        .groupby("category")["mean_abs_shap_5fold"]
        .agg(
            count="count",
            mean="mean",
            maximum="max",
            total="sum",
        )
        .sort_values(
            by="mean",
            ascending=False,
        )
    )

    # -------------------------------------------------------------------------
    # 7. TXT report
    #
    # IMPORTANT:
    # CSV column numbers are intentionally NOT printed anywhere in feature
    # summary tables, per the user's request.
    # -------------------------------------------------------------------------
    SEP = "=" * 145
    THIN = "-" * 145
    lines = []

    lines.append(SEP)
    lines.append(
        f"FINNEGAN-STYLE 5-FOLD SUBJECT-LEVEL SHAP ANALYSIS FOR Δ{target}"
    )
    lines.append(SEP)
    lines.append(
        "PART A = Model Performance; "
        "PART B = SHAP Feature Importance."
    )
    lines.append("")

    lines.append("REPRODUCIBILITY SETTINGS")
    lines.append(THIN)
    lines.append(
        f"Random state = {args.random_state}"
    )
    lines.append(
        f"Random Forest trees = {args.n_estimators}"
    )
    lines.append(
        f"Random Forest n_jobs = {N_JOBS}"
    )
    lines.append(
        "5-fold KFold: shuffle=True with fixed random_state."
    )
    lines.append(
        "PYTHONHASHSEED = 42."
    )

    lines.append("")
    lines.append("DATASET")
    lines.append(THIN)
    lines.append(
        f"Valid subjects = {n_subjects}"
    )
    lines.append(
        f"Total delta samples = {len(y_all)} "
        f"({N_DELTA_SAMPLES_PER_SUBJECT} per subject)"
    )
    lines.append(
        f"Total model features = {TOTAL_FEATURES} "
        f"(156 delta morphology + 5 demographics)"
    )
    lines.append(
        f"Excluded subjects = {', '.join(sorted(EXCLUDED))}"
    )
    lines.append(
        f"X shape = {X_all.shape}"
    )
    lines.append(
        f"y shape = {y_all.shape}"
    )

    lines.append("")
    lines.append("FEATURE CONSTRUCTION")
    lines.append(THIN)
    lines.append(
        "Morphological features: 1-156 are calculated as rows 1-4 minus "
        "baseline row 0 for each subject."
    )
    lines.append(
        "Demographics are subject-level baseline values repeated four times "
        "and horizontally concatenated with the four delta samples."
    )
    lines.append(
        "Demographic feature order: s1=Gender, s2=Age, "
        "s3=Height, s4=Weight, s5=BMI."
    )
    lines.append(
        "Demographic category: Demographics."
    )

    lines.append("")
    lines.append("CSV / FEATURE-DEFINITION VALIDATION")
    lines.append(THIN)
    lines.append(
        "Verified CSV layout from all_trimmed.csv:"
    )
    lines.append(
        "  Height = column 6; Weight = column 7; BMI = column 8; "
        "Gender = column 9; Age = column 10."
    )
    lines.append(
        "  Morphological feature 1-156 = columns 11-166."
    )
    lines.append(
        f"Excel definition format = {excel_format}"
    )
    lines.append(
        "  Feature names are mapped in Excel row order to morphological "
        "features 1-156."
    )

    if csv_mapping_summary["first_mapping_check"] is not None:
        first_check = csv_mapping_summary["first_mapping_check"]

        lines.append(
            f"First checked subject = "
            f"{csv_mapping_summary['first_checked_subject']}"
        )
        lines.append(
            f"Total CSV columns = "
            f"{first_check['total_csv_columns']}"
        )
        lines.append(
            f"Demographic header check = "
            f"{first_check['demographic_header_matches']}"
        )
        lines.append(
            f"Morphological index header check = "
            f"{first_check['morph_index_header_matches']}"
        )
        lines.append(
            f"Morphological Excel-name header check = "
            f"{first_check['morph_excel_name_header_matches']}"
        )
        all_layout_checks_match = (
            csv_mapping_summary["all_demo_headers_match"]
            and csv_mapping_summary["all_morph_index_headers_match"]
        )
        lines.append(
            "All checked CSVs have consistent header validation = "
            f"{all_layout_checks_match}"
        )

    # Full feature mapping WITHOUT CSV Col.
    lines.append("")
    lines.append("FULL FEATURE MAPPING")
    lines.append(THIN)
    lines.append(
        f"{'Feature':<10}"
        f"{'Feature Name':<30}"
        f"{'Category':<22}"
    )
    lines.append("-" * 62)

    for feature_id, name, category in zip(
        feature_ids,
        feature_names,
        feature_categories,
    ):
        lines.append(
            f"{str(feature_id):<10}"
            f"{name:<30}"
            f"{category:<22}"
        )

    # -------------------------------------------------------------------------
    # PART A: Model Performance
    # -------------------------------------------------------------------------
    lines.append("")
    lines.append(SEP)
    lines.append("PART A. MODEL PERFORMANCE")
    lines.append(SEP)
    lines.append(
        f"Each fold: {int(n_subjects * 4 / 5)}? "
        "training samples and the remaining subject-held-out samples."
    )
    lines.append(
        "Random Forest is trained on 56 subjects and evaluated on "
        "14 completely unseen subjects."
    )
    lines.append(
        f"The model predicts Δ{target} from the 161-dimensional feature vector."
    )
    lines.append(
        "For MAP, MAP = 1/3*SBP + 2/3*DBP is calculated for each measurement "
        "before subtracting the baseline value."
    )
    lines.append("")

    lines.append(
        f"{'Fold':<7}"
        f"{'TrainSubj':<12}"
        f"{'TestSubj':<11}"
        f"{'TrainN':<9}"
        f"{'TestN':<9}"
        f"{'MAE':<12}"
        f"{'RMSE':<12}"
        f"{'R2':<12}"
        f"{'Pearson r':<14}"
        f"{'Spearman rho':<15}"
        f"{'Bias':<12}"
    )
    lines.append(THIN)

    for row in fold_metrics:
        lines.append(
            f"{row['fold']:<7}"
            f"{row['train_subjects']:<12}"
            f"{row['test_subjects']:<11}"
            f"{row['train_samples']:<9}"
            f"{row['test_samples']:<9}"
            f"{row['MAE']:<12.6f}"
            f"{row['RMSE']:<12.6f}"
            f"{row['R2']:<12.6f}"
            f"{row['Pearson_r']:<14.6f}"
            f"{row['Spearman_rho']:<15.6f}"
            f"{row['Bias_pred_minus_true']:<12.6f}"
        )

    lines.append("")
    lines.append("5-fold summary (Mean ± SD):")
    lines.append(
        f"  MAE          = {metric_summary['MAE_mean']:.6f} "
        f"± {metric_summary['MAE_std']:.6f}"
    )
    lines.append(
        f"  RMSE         = {metric_summary['RMSE_mean']:.6f} "
        f"± {metric_summary['RMSE_std']:.6f}"
    )
    lines.append(
        f"  R2           = {metric_summary['R2_mean']:.6f} "
        f"± {metric_summary['R2_std']:.6f}"
    )
    lines.append(
        f"  Pearson r    = {metric_summary['Pearson_mean']:.6f} "
        f"± {metric_summary['Pearson_std']:.6f}"
    )
    lines.append(
        f"  Spearman rho = {metric_summary['Spearman_mean']:.6f} "
        f"± {metric_summary['Spearman_std']:.6f}"
    )
    lines.append(
        f"  Bias         = {metric_summary['Bias_mean']:.6f} "
        f"± {metric_summary['Bias_std']:.6f}"
    )

    lines.append("")
    lines.append("5-fold summary (Median [IQR]):")
    lines.append(
        f"  MAE          = {metric_summary['MAE_median']:.6f} "
        f"[{metric_summary['MAE_IQR']:.6f}]"
    )
    lines.append(
        f"  RMSE         = {metric_summary['RMSE_median']:.6f} "
        f"[{metric_summary['RMSE_IQR']:.6f}]"
    )
    lines.append(
        f"  R2           = {metric_summary['R2_median']:.6f} "
        f"[{metric_summary['R2_IQR']:.6f}]"
    )
    lines.append(
        f"  Pearson r    = {metric_summary['Pearson_median']:.6f} "
        f"[{metric_summary['Pearson_IQR']:.6f}]"
    )
    lines.append(
        f"  Spearman rho = {metric_summary['Spearman_median']:.6f} "
        f"[{metric_summary['Spearman_IQR']:.6f}]"
    )

    lines.append("")
    lines.append("Overall OOF prediction performance:")
    lines.append(
        f"  OOF MAE          = {oof_mae:.6f}"
    )
    lines.append(
        f"  OOF RMSE         = {oof_rmse:.6f}"
    )
    lines.append(
        f"  OOF R2           = {oof_r2:.6f}"
    )
    lines.append(
        f"  OOF Pearson r    = {oof_pearson:.6f}"
    )
    lines.append(
        f"  OOF Spearman rho = {oof_spearman:.6f}"
    )
    lines.append(
        f"  OOF Bias         = {oof_bias:.6f}"
    )

    # -------------------------------------------------------------------------
    # PART B: SHAP Feature Importance
    # -------------------------------------------------------------------------
    lines.append("")
    lines.append(SEP)
    lines.append("PART B. SHAP FEATURE IMPORTANCE")
    lines.append(SEP)
    lines.append(
        "Following the Finnegan-style procedure, TreeSHAP is calculated "
        "on the TRAINING samples of each fold."
    )
    lines.append(
        "The test set is used for model-performance evaluation and is not used "
        "to calculate the main SHAP feature-importance ranking."
    )
    lines.append(
        "For each fold: feature importance = mean absolute SHAP over "
        "the training samples."
    )
    lines.append(
        "Final feature contribution = arithmetic mean of the five "
        "fold-specific mean(|SHAP|) values."
    )
    lines.append(
        "Feature ranks are calculated within each fold and normalized from "
        "1 (highest contribution) to 0 (lowest contribution)."
    )
    lines.append("")

    lines.append(
        f"{'Rank':<7}"
        f"{'Feat':<10}"
        f"{'Feature Name':<30}"
        f"{'Category':<20}"
        f"{'Mean|SHAP|':<15}"
        f"{'SD|SHAP|':<14}"
        f"{'Mean signed':<15}"
        f"{'Mean RankCoeff':<16}"
        f"{'Median Rank':<13}"
        f"{'F1':<6}"
        f"{'F2':<6}"
        f"{'F3':<6}"
        f"{'F4':<6}"
        f"{'F5':<6}"
    )
    lines.append(THIN)

    for _, row in df_shap.iterrows():
        lines.append(
            f"{int(row['final_rank']):<7}"
            f"{str(row['feature_index']):<10}"
            f"{str(row['feature_name'])[:29]:<30}"
            f"{str(row['category'])[:19]:<20}"
            f"{row['mean_abs_shap_5fold']:<15.6f}"
            f"{row['std_abs_shap_5fold']:<14.6f}"
            f"{row['mean_signed_shap_5fold']:<15.6f}"
            f"{row['mean_rank_coefficient']:<16.6f}"
            f"{row['median_rank']:<13.2f}"
            f"{int(row['fold_1_rank']):<6}"
            f"{int(row['fold_2_rank']):<6}"
            f"{int(row['fold_3_rank']):<6}"
            f"{int(row['fold_4_rank']):<6}"
            f"{int(row['fold_5_rank']):<6}"
        )

    lines.append("")
    lines.append("TOP 20 FEATURES BY 5-FOLD AVERAGED MEAN |SHAP|")
    lines.append(THIN)

    for _, row in df_shap.head(20).iterrows():
        lines.append(
            f"{int(row['final_rank']):>2}. "
            f"{row['feature_name']} "
            f"(feature {row['feature_index']}, "
            f"{row['category']}): "
            f"mean|SHAP|={row['mean_abs_shap_5fold']:.6f}, "
            f"SD={row['std_abs_shap_5fold']:.6f}, "
            f"median rank coefficient="
            f"{row['median_rank_coefficient']:.6f}"
        )

    lines.append("")
    lines.append("CATEGORY-WISE SHAP SUMMARY")
    lines.append(THIN)
    lines.append(
        f"{'Category':<20}"
        f"{'Count':<8}"
        f"{'Avg Mean|SHAP|':<18}"
        f"{'Max Mean|SHAP|':<18}"
        f"{'Sum Mean|SHAP|':<18}"
    )
    lines.append("-" * 82)

    for category, row in category_summary.iterrows():
        lines.append(
            f"{category:<20}"
            f"{int(row['count']):<8}"
            f"{row['mean']:<18.6f}"
            f"{row['maximum']:<18.6f}"
            f"{row['total']:<18.6f}"
        )

    lines.append("")
    lines.append("FOLD-SPECIFIC SHAP TOP 20 RANKINGS")
    lines.append(THIN)

    df_fold_shap = pd.DataFrame(
        fold_feature_records
    )

    for fold_no in range(1, 6):
        fold_top = (
            df_fold_shap[
                df_fold_shap["fold"] == fold_no
            ]
            .sort_values("rank")
            .head(20)
        )

        lines.append(f"\nFold {fold_no}:")
        lines.append(
            f"{'Rank':<7}"
            f"{'Feat':<10}"
            f"{'Feature Name':<30}"
            f"{'Mean|SHAP|':<15}"
            f"{'RankCoeff':<12}"
        )

        for _, row in fold_top.iterrows():
            lines.append(
                f"{int(row['rank']):<7}"
                f"{str(row['feature_index']):<10}"
                f"{str(row['feature_name'])[:29]:<30}"
                f"{row['mean_abs_shap']:<15.6f}"
                f"{row['rank_coefficient']:<12.6f}"
            )

    lines.append("")
    lines.append(SEP)
    lines.append("METHOD SUMMARY")
    lines.append(SEP)
    lines.append(
        "1) 70 subjects are divided into 5 subject-level folds using a "
        "fixed random seed."
    )
    lines.append(
        "2) For every subject, four delta morphological samples are generated "
        "from rows 1-4 minus baseline row 0."
    )
    lines.append(
        "3) Five subject-level demographic variables are taken from the "
        "baseline row, ordered as s1=Gender, s2=Age, s3=Height, "
        "s4=Weight, s5=BMI, and repeated four times."
    )
    lines.append(
        "4) The 156 delta morphological features and the five repeated "
        "demographic features are horizontally concatenated into a "
        "161-dimensional input vector."
    )
    lines.append(
        "5) Each fold trains Random Forest on 56 subjects and evaluates "
        "prediction performance on 14 completely unseen subjects."
    )
    lines.append(
        "6) SHAP feature importance is calculated on the 56-subject "
        "training data using TreeSHAP."
    )
    lines.append(
        "7) Final contribution is the mean of the five fold-wise "
        "mean(|SHAP|) values."
    )
    lines.append(
        "8) For MAP, MAP = 1/3*SBP + 2/3*DBP is calculated row-wise first, "
        "and then ΔMAP = MAP[1:5] - MAP[0]."
    )
    lines.append(
        "9) Subject-level separation is preserved so that all four samples "
        "from a subject are always assigned to the same fold."
    )
    lines.append(
        "10) SHAP values quantify model contribution and should not be "
        "interpreted as causal physiological effects."
    )
    lines.append(SEP)

    full_report = "\n".join(lines) + "\n"

    out_txt_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        out_txt_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(full_report)

    # -------------------------------------------------------------------------
    # 8. Four requested machine-readable CSV outputs.
    #
    # No CSV-column-position field is included.
    # -------------------------------------------------------------------------
    output_stem = out_txt_path.with_suffix("")

    feature_summary_path = (
        output_stem.parent /
        f"{output_stem.name}_SHAP_feature_summary.csv"
    )

    fold_metrics_path = (
        output_stem.parent /
        f"{output_stem.name}_model_performance_by_fold.csv"
    )

    fold_shap_path = (
        output_stem.parent /
        f"{output_stem.name}_SHAP_by_fold.csv"
    )

    oof_path = (
        output_stem.parent /
        f"{output_stem.name}_OOF_predictions.csv"
    )

    merged_csv_path = (
        output_stem.parent /
        f"{output_stem.name}_feature_and_SHAP_matrix.csv"
    )

    
    df_shap.to_csv(feature_summary_path, index=False, encoding="utf-8-sig")
    df_metrics.to_csv(fold_metrics_path, index=False, encoding="utf-8-sig")
    df_fold_shap.to_csv(fold_shap_path, index=False, encoding="utf-8-sig")

    oof_df = pd.DataFrame(
        {
            "sample_index": np.arange(len(y_all)),
            "subject": np.repeat(subject_list, N_DELTA_SAMPLES_PER_SUBJECT),
            "fold": oof_fold,
            "y_true_delta_BP": y_all,
            "y_pred_delta_BP": oof_pred,
            "error_pred_minus_true": oof_pred - y_all,
        }
    )
    oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")

    # (label="feature_delta")
    df_feat = pd.DataFrame(X_all, columns=feature_names)
    df_feat.insert(0, "label", "feature_delta")
    df_feat.insert(0, "fold", oof_fold)
    df_feat.insert(0, "subject", np.repeat(subject_list, N_DELTA_SAMPLES_PER_SUBJECT))
    df_feat.insert(0, "sample_index", np.arange(len(y_all)))

    # (label="shap_value")
    df_shap_mat = pd.DataFrame(oof_shap, columns=feature_names)
    df_shap_mat.insert(0, "label", "shap_value")
    df_shap_mat.insert(0, "fold", oof_fold)
    df_shap_mat.insert(0, "subject", np.repeat(subject_list, N_DELTA_SAMPLES_PER_SUBJECT))
    df_shap_mat.insert(0, "sample_index", np.arange(len(y_all)))

    
    merged_df = pd.concat([df_feat, df_shap_mat], axis=0, ignore_index=True)
    merged_df.to_csv(merged_csv_path, index=False, encoding="utf-8-sig")

    print("\nAnalysis complete.")
    print(
        f"Final feature dimension = {TOTAL_FEATURES} "
        f"(156 morphology + 5 demographics)"
    )
    print(f"TXT report: {out_txt_path.resolve()}")
    print(f"SHAP feature summary: {feature_summary_path.resolve()}")
    print(f"Fold model performance: {fold_metrics_path.resolve()}")
    print(f"Fold SHAP results: {fold_shap_path.resolve()}")
    print(f"OOF predictions: {oof_path.resolve()}")


if __name__ == "__main__":
    main()
