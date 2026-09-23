import io
from pathlib import Path
import pandas as pd


def load_representative_features(csv_path: Path) -> pd.DataFrame:
    """Load the Stage 2 clustering CSV file with delimiters and extract final representative features."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature file not found: {csv_path.resolve()}")

    # 1. Attempt to filter directly from the top section (full 161 features) using the boolean flag[cite: 3]
    try:
        df_top = pd.read_csv(csv_path, comment="#")
        if "is_stage2_representative" in df_top.columns:
            return df_top[df_top["is_stage2_representative"] == True].copy()
    except Exception:
        pass

    # 2. If the format contains explicit separators, extract the bottom table[cite: 3]
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    parts = content.split("# ==============================================================================")
    if len(parts) >= 3:
        bottom_csv = parts[-1].strip()
        lines = [
            line for line in bottom_csv.split("\n")
            if not line.strip().startswith("#") and line.strip()
        ]
        return pd.read_csv(io.StringIO("\n".join(lines)))

    raise ValueError(f"Failed to parse file structure: {csv_path.resolve()}")


def analyze_overlapping_features():
    targets = ["SBP", "DBP", "MAP"]
    feature_sets = {}
    df_dict = {}

    for t in targets:
        csv_file = Path(f"{t}_SHAP_stage2_clusters_features.csv")
        df_rep = load_representative_features(csv_file)
        df_dict[t] = df_rep
        feature_sets[t] = set(df_rep["feature_name"])

    sbp_set = feature_sets["SBP"]
    dbp_set = feature_sets["DBP"]
    map_set = feature_sets["MAP"]

    # Set operations[cite: 3]
    common_all_3 = sbp_set & dbp_set & map_set
    common_sbp_dbp = sbp_set & dbp_set
    common_sbp_map = sbp_set & map_set
    common_dbp_map = dbp_set & map_set
    all_union = sbp_set | dbp_set | map_set

    # Print terminal report[cite: 3]
    print("=" * 80)
    print("STAGE 2 REPRESENTATIVE FEATURE OVERLAP ANALYSIS REPORT (SBP / DBP / MAP)")
    print("=" * 80)
    print("Number of representative features selected per target:")
    for t in targets:
        print(f"  - {t:<4} : {len(feature_sets[t])} features")
    print(f"  - Total unique features across all targets (Union): {len(all_union)}")
    print("-" * 80)

    print(f"[1] Core features shared across all 3 targets (SBP ∩ DBP ∩ MAP, Total: {len(common_all_3)}):")
    for feat in sorted(common_all_3):
        print(f"  ★ {feat}")
    print()

    print("[2] Pairwise overlapping features:")
    print(f"  ▶ SBP ∩ DBP (Total: {len(common_sbp_dbp)}): {sorted(common_sbp_dbp)}")
    print(f"  ▶ SBP ∩ MAP (Total: {len(common_sbp_map)}): {sorted(common_sbp_map)}")
    print(f"  ▶ DBP ∩ MAP (Total: {len(common_dbp_map)}): {sorted(common_dbp_map)}")
    print("=" * 80)

    # Compile and export cross-target comparison summary[cite: 3]
    records = []
    for feat in all_union:
        cat = "Unknown"
        for t in targets:
            sub = df_dict[t][df_dict[t]["feature_name"] == feat]
            if not sub.empty and "category" in sub.columns:
                cat = sub["category"].iloc[0]
                break

        records.append({
            "feature_name": feat,
            "category": cat,
            "in_SBP": feat in sbp_set,
            "in_DBP": feat in dbp_set,
            "in_MAP": feat in map_set,
            "overlap_count": (feat in sbp_set) + (feat in dbp_set) + (feat in map_set),
            "mean_abs_shap_SBP": (
                df_dict["SBP"].loc[df_dict["SBP"]["feature_name"] == feat, "mean_abs_shap"].values[0]
                if feat in sbp_set else None
            ),
            "mean_abs_shap_DBP": (
                df_dict["DBP"].loc[df_dict["DBP"]["feature_name"] == feat, "mean_abs_shap"].values[0]
                if feat in dbp_set else None
            ),
            "mean_abs_shap_MAP": (
                df_dict["MAP"].loc[df_dict["MAP"]["feature_name"] == feat, "mean_abs_shap"].values[0]
                if feat in map_set else None
            ),
        })

    df_overlap = pd.DataFrame(records).sort_values(
        by=["overlap_count", "category", "feature_name"],
        ascending=[False, True, True]
    ).reset_index(drop=True)

    out_csv = Path("Stage2_overlapping_features_summary.csv")
    df_overlap.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Detailed cross-target feature comparison table saved to: {out_csv.resolve()}")


if __name__ == "__main__":
    analyze_overlapping_features()