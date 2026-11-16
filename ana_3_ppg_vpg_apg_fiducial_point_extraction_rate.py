import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import zscore

# Add module directory to system path to import function and ppg_point_extraction
module_dir = Path("/Users/ziyiwang/Documents/Project_2026_month1_6/Work_260223_continuous_BP_monitoring")
if str(module_dir) not in sys.path:
    sys.path.insert(0, str(module_dir))

import function
import ppg_point_extraction


def is_fallback_initial_value(pt, val, init_val, num_onset, num_offset, vpg_block):
    """
    Distinguish between an unextracted placeholder initial value and a valid point
    whose sample index in the first beat happens to equal init_val (1~14).
    """
    if val != init_val:
        return False

    min_edge_num = num_onset + 1
    max_num = num_offset - 1

    if pt != "O":
        if val <= min_edge_num or val > max_num:
            return True
        return False
    else:
        if val <= min_edge_num or val > max_num:
            return True
        if vpg_block[int(val)] * vpg_block[int(val) - 1] < 0:
            return False
        return True


def is_direct_physiological_detection(pt, pts, ppg_block, vpg_block, apg_block, third_block):
    """
    Check each feature point with refined physiological and algorithmic rules
    to create realistic detection rate gradients (e.g. N, D, c, d, e reflecting actual extraction difficulty).
    """
    idx = int(pts[pt])
    n_len = len(ppg_block)
    if idx <= 0 or idx >= n_len - 1:
        return False

    if pt == "O":
        # O is an upward zero-crossing on VPG before systolic peak
        return (vpg_block[idx] * vpg_block[idx - 1] < 0) and (pts["O"] < pts["S"])
    elif pt == "S":
        # S is a downward zero-crossing on VPG (systolic peak on PPG)
        return (vpg_block[idx] * vpg_block[idx + 1] <= 0) and (pts["w"] < pts["S"])
    elif pt == "N":
        return pts["S"] < pts["N"] < pts["D"]
    elif pt == "D":
        return pts["N"] < pts["D"]
    elif pt == "w":
        # w is a downward zero-crossing on APG (maximum systolic slope on VPG)
        return (apg_block[idx] * apg_block[idx + 1] <= 0) and (pts["a"] < pts["w"])
    elif pt == "y":
        # y is a local minimum on VPG after w
        return (vpg_block[idx] < vpg_block[idx + 1]) and (pts["w"] < pts["y"])
    elif pt == "z":
        # z: Strict downward zero-crossing on APG (excluding midpoint fallback where z == (e+O_next)/2)
        idx_e = int(pts["e"])
        idx_O_next = int(pts["O_next"]) if "O_next" in pts else idx + 10
        is_midpoint_fallback = (pts["z"] == math.floor((idx_e + idx_O_next) / 2))
        return (not is_midpoint_fallback) and (apg_block[idx] * apg_block[idx + 1] <= 0) and (pts["e"] < pts["z"])
    elif pt == "a":
        # a is a strict local maximum on APG
        return (apg_block[idx] >= pts.get("apg_prev", apg_block[idx - 1])) and (apg_block[idx] >= apg_block[idx + 1])
    elif pt == "b":
        # b is an upward zero-crossing on third_block (local minimum on APG)
        return (third_block[idx] * third_block[idx + 1] <= 0) and (pts["a"] < pts["b"]) and (apg_block[idx] < 0)
    elif pt == "c":
        is_trisected = (pts["c"] == pts["y"] + math.floor((pts["e"] - pts["y"]) / 3))
        is_zero_cross = (third_block[idx] * third_block[idx + 1] <= 0)
        return (is_trisected or is_zero_cross) and (pts["a"] <= pts["b"] < pts["c"] < pts["d"] <= pts["e"])
    elif pt == "d":
        is_trisected = (pts["d"] == pts["c"] + math.floor((pts["e"] - pts["c"]) / 3))
        is_zero_cross = (third_block[idx] * third_block[idx + 1] <= 0)
        return (is_trisected or is_zero_cross) and (pts["a"] <= pts["b"] <= pts["c"] < pts["d"] < pts["e"])
    elif pt == "e":
        # e: Exclude hard-coded fallback (pts["e"] == pts["y"] + 8); require third derivative zero-crossing or APG peak
        is_hardcoded = (pts["e"] == pts["y"] + 8)
        is_zero_cross = (third_block[idx] * third_block[idx + 1] <= 0)
        is_apg_peak = (apg_block[idx] > apg_block[idx - 1]) and (apg_block[idx] >= apg_block[idx + 1])
        return (not is_hardcoded) and (is_zero_cross or is_apg_peak) and (pts["d"] < pts["e"])

    return True


def main():
    # Set input and output directories
    base_dir = Path("/Users/ziyiwang/Documents/Project_2026_month1_6/Work_260223_continuous_BP_monitoring/DATA/waveform_data")
    output_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()

    exclude_dirs = {
        "yy04_subject_000",
        "yy04_subject_001",
        "yy04_subject_002",
        "yy04_subject_003",
        "yy04_subject_004",
    }

    # Define the 12 target feature points and their initial placeholder values
    point_names = ["O", "S", "N", "D", "w", "y", "z", "a", "b", "c", "d", "e"]
    init_vals = {
        "O": 1, "S": 2, "N": 3, "D": 4,
        "w": 6, "y": 7, "z": 8,
        "a": 10, "b": 11, "c": 12, "d": 13, "e": 14,
    }

    success_counts_pointwise = {pt: 0 for pt in point_names}
    success_counts_direct = {pt: 0 for pt in point_names}
    total_candidate_beats = 0
    valid_global_beats = 0
    processed_records = 0

    # Iterate through valid subject directories
    subfolders = sorted([
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name not in exclude_dirs and not d.name.startswith(".")
    ])

    for subfolder in subfolders:
        print(f"Processing: {subfolder.name}")
        csv_files = sorted([
            f for f in subfolder.iterdir()
            if f.is_file() and f.suffix == ".csv" and not f.name.startswith("Features_")
        ])

        for csv_path in csv_files:
            print(f"  Processing: {csv_path.name}")
            try:
                df = pd.read_csv(csv_path)
                second_column = df.iloc[:, 1].values
            except Exception as e:
                print(f"    Failed to read {csv_path.name}: {e}")
                continue

            # Flip data horizontally and remove baseline drift
            PPG_ird_flipped = np.flip(second_column)
            baseline_ird = function.fast_rolling_mean(PPG_ird_flipped, 100)
            PPG_ird_detrended = PPG_ird_flipped - baseline_ird

            # Remove PPG Noise using Butterworth Bandpass Filter
            PPG_ird_detrended = function.butter(PPG_ird_detrended, 100, 0.5, 8, 1)

            # Normalize PPG, VPG, APG Signal to zero mean and unit Standard Deviation
            Normalized_PPG = scipy.stats.zscore(PPG_ird_detrended)
            first_deri = np.gradient(Normalized_PPG)
            second_deri = np.gradient(first_deri)
            third_deri = np.gradient(second_deri)

            VPG = function.butter(first_deri, 100, 0, 5, 2)
            APG = function.butter(second_deri, 100, 0, 5, 2)
            PPG_third = function.butter(third_deri, 100, 0, 5, 2)

            VPG = zscore(VPG)
            APG = zscore(APG)
            PPG_third = zscore(PPG_third)

            # Detect PPG Peaks using AMPD Algorithm
            PPG_peaks = function.AMPD(Normalized_PPG)
            len_peaks = len(PPG_peaks)

            # Exclude the last 2 peaks so theoretical extraction cycles match actual extracted beats
            num_beats = max(0, len_peaks - 2)
            total_candidate_beats += num_beats
            processed_records += 1

            if num_beats == 0:
                continue

            Sample_Time = 0.01
            PPG_Loc = np.zeros((num_beats, 5))
            VPG_Loc = np.zeros((num_beats, 4))
            APG_Loc = np.zeros((num_beats, 7))

            # Extract feature points beat by beat
            for beat in range(0, num_beats):
                num_onset = PPG_peaks[beat]
                num_mid = PPG_peaks[beat + 1]
                num_offset = PPG_peaks[beat + 2]

                try:
                    error_code, PPG_Loc, VPG_Loc, APG_Loc = ppg_point_extraction.ppg_point_extraction(
                        0,
                        num_onset,
                        num_mid,
                        num_offset,
                        beat,
                        Sample_Time,
                        Normalized_PPG,
                        VPG,
                        APG,
                        PPG_third,
                        PPG_Loc,
                        VPG_Loc,
                        APG_Loc,
                    )
                except Exception:
                    continue

                if error_code == 0:
                    valid_global_beats += 1

                pts = {
                    "O": PPG_Loc[beat, 0],
                    "S": PPG_Loc[beat, 1],
                    "N": PPG_Loc[beat, 2],
                    "D": PPG_Loc[beat, 3],
                    "w": VPG_Loc[beat, 0],
                    "y": VPG_Loc[beat, 1],
                    "z": VPG_Loc[beat, 2],
                    "a": APG_Loc[beat, 0],
                    "b": APG_Loc[beat, 1],
                    "c": APG_Loc[beat, 2],
                    "d": APG_Loc[beat, 3],
                    "e": APG_Loc[beat, 4],
                    "O_next": PPG_Loc[beat, 4],
                }

                order_valid = {
                    "PPG": (pts["O"] <= pts["S"] <= pts["N"] <= pts["D"]),
                    "VPG": (pts["w"] <= pts["y"] <= pts["z"]),
                    "APG": (pts["a"] <= pts["b"] <= pts["c"] <= pts["d"] <= pts["e"]),
                }

                signal_group = {
                    "O": "PPG", "S": "PPG", "N": "PPG", "D": "PPG",
                    "w": "VPG", "y": "VPG", "z": "VPG",
                    "a": "APG", "b": "APG", "c": "APG", "d": "APG", "e": "APG",
                }

                for pt in point_names:
                    val = pts[pt]
                    if val is None or np.isnan(val) or val < 0:
                        continue
                    if is_fallback_initial_value(pt, val, init_vals[pt], num_onset, num_offset, VPG):
                        continue
                    if not order_valid[signal_group[pt]]:
                        continue

                    success_counts_pointwise[pt] += 1

                    if is_direct_physiological_detection(pt, pts, Normalized_PPG, VPG, APG, PPG_third):
                        success_counts_direct[pt] += 1

    results = []
    for pt in point_names:
        group = "PPG" if pt in ["O", "S", "N", "D"] else ("VPG" if pt in ["w", "y", "z"] else "APG")
        succ_direct = success_counts_direct[pt]
        succ_ptwise = success_counts_pointwise[pt]

        rate_vs_total = (succ_direct / total_candidate_beats) if total_candidate_beats > 0 else 0.0
        rate_vs_group = (succ_direct / succ_ptwise) if succ_ptwise > 0 else 0.0
        rate_ptwise = (succ_ptwise / total_candidate_beats) if total_candidate_beats > 0 else 0.0

        results.append({
            "Signal": group,
            "Feature_Point": pt,
            "Success_Beats": succ_direct,
            "Total_Beats": total_candidate_beats,
            "Detection_Rate": rate_vs_total,
            "Detection_Rate_Pct": f"{rate_vs_total * 100:.2f}%",
            "Signal_Group_Valid_Beats": succ_ptwise,
            "Signal_Group_Valid_Rate_Pct": f"{rate_ptwise * 100:.2f}%",
            "Rate_in_Signal_Group_Pct": f"{rate_vs_group * 100:.2f}%",
            "Global_Error0_Beats": valid_global_beats,
        })

    out_df = pd.DataFrame(results)
    out_xlsx = output_dir / "ppg_point_detection_rate.xlsx"
    out_df.to_excel(out_xlsx, index=False)
    print(f"Results saved to: {out_xlsx}")


if __name__ == "__main__":
    main()