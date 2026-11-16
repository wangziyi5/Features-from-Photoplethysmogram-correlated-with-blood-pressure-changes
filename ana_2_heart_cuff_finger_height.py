from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Standard Unicode mathematical minus sign (U+2212)
MATH_MINUS = "\u2212"

# Anthropometric segment length proportions relative to stature H (Winter, 2009)
PROP_UPPER_ARM = 0.186          # Total upper arm length (shoulder to elbow)
PROP_CUFF_ELBOW = 0.068         # Distance from brachial cuff center to elbow (~11.4 cm; yields ~8.1 cm vertical offset at -45 deg)
PROP_SHOULDER_CUFF = PROP_UPPER_ARM - PROP_CUFF_ELBOW  # Distance from shoulder to cuff center (0.118H)
PROP_FOREARM = 0.146            # Forearm length (elbow to wrist)
PROP_HALF_HAND = 0.054          # Half hand length (wrist to proximal phalanx ring site, 0.5 * 0.108H)

# Hydrostatic pressure factor: Delta P = -rho * g * Delta h (-0.78 mmHg/cm, Netea et al., 2003)
HYDROSTATIC_FACTOR = -0.78

# Seated posture definitions: (Posture ID, Arm Configuration, theta_upper_deg, theta_forearm_deg)
POSTURES = [
    ("S1", f"Arm resting naturally on the table ({MATH_MINUS}45°)", -45.0, 0.0),
    ("S2", "Arm raised horizontally at shoulder height (0°)", 0.0, 0.0),
    ("S3", "Arm raised vertically at 90-degree to the table (90°)", 90.0, 90.0),
    ("S4", f"Arm hanging down naturally below the table ({MATH_MINUS}90°)", -90.0, -90.0),
    ("S5", f"Arm resting naturally on the table ({MATH_MINUS}45°)", -45.0, 0.0),
]


def format_math_minus(text: str) -> str:
    """Replace standard ASCII hyphen-minus with the mathematical minus sign (U+2212)."""
    return text.replace("-", MATH_MINUS)


def compute_posture_metrics(height_cm: float, posture_idx: int) -> dict:
    """Compute subject-specific vertical heights and hydrostatic pressure offsets."""
    posture_id, config_desc, theta1_deg, theta2_deg = POSTURES[posture_idx]

    l_shoulder_cuff = PROP_SHOULDER_CUFF * height_cm
    l_cuff_elbow = PROP_CUFF_ELBOW * height_cm
    l_forearm_ring = (PROP_FOREARM + PROP_HALF_HAND) * height_cm

    # In the reference seated posture (-45 deg upper arm, forearm flat on desk), cuff is at heart level (dh_heart_cuff = 0)
    d0_heart_shoulder = l_shoulder_cuff * np.sin(np.deg2rad(45.0))

    sin_t1 = np.sin(np.deg2rad(theta1_deg))
    sin_t2 = np.sin(np.deg2rad(theta2_deg))

    # Vertical height from heart level to cuff center
    dh_heart_cuff = d0_heart_shoulder + l_shoulder_cuff * sin_t1
    if np.isclose(dh_heart_cuff, 0.0, atol=1e-9):
        dh_heart_cuff = 0.0

    # Vertical height from cuff center to finger ring (cuff-to-elbow + forearm + half-hand)
    dh_cuff_finger = l_cuff_elbow * sin_t1 + l_forearm_ring * sin_t2
    if np.isclose(dh_cuff_finger, 0.0, atol=1e-9):
        dh_cuff_finger = 0.0

    # Vertical height from heart level to finger ring
    dh_heart_finger = dh_heart_cuff + dh_cuff_finger

    # Hydrostatic pressure change relative to heart level: Delta P = -0.78 mmHg/cm * Delta h
    dp_cuff = HYDROSTATIC_FACTOR * dh_heart_cuff
    dp_finger = HYDROSTATIC_FACTOR * dh_heart_finger

    return {
        "Posture": posture_id,
        "Arm Configuration": config_desc,
        "dh_heart_cuff": dh_heart_cuff,
        "dh_heart_finger": dh_heart_finger,
        "dh_cuff_finger": dh_cuff_finger,
        "dp_cuff": dp_cuff,
        "dp_finger": dp_finger,
    }


def load_subject_records(base_dir: Path, exclude_dirs: set) -> pd.DataFrame:
    """Traverse subject directories and read SBP, DBP, and Height from all_trimmed.csv."""
    records = []

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    subject_dirs = sorted([
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name not in exclude_dirs and not d.name.startswith(".")
    ])

    subj_idx = 0
    for subj_dir in subject_dirs:
        csv_path = subj_dir / "all_trimmed.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        if len(df) < 5:
            continue

        subj_idx += 1
        subject_id = f"subject_{subj_idx:02d}"

        # Column 1: SBP (index 0), Column 2: DBP (index 1), Column 6: Height (index 5)
        for p_idx in range(5):
            sbp = int(round(float(df.iloc[p_idx, 0])))
            dbp = int(round(float(df.iloc[p_idx, 1])))
            height_cm = float(df.iloc[p_idx, 5])

            metrics = compute_posture_metrics(height_cm, p_idx)
            records.append({
                "Subject_ID": subject_id,
                "Posture": metrics["Posture"],
                "Arm Configuration": metrics["Arm Configuration"],
                "Height_cm": height_cm,
                "dh_heart_cuff": metrics["dh_heart_cuff"],
                "dh_heart_finger": metrics["dh_heart_finger"],
                "dh_cuff_finger": metrics["dh_cuff_finger"],
                "dp_cuff": metrics["dp_cuff"],
                "dp_finger": metrics["dp_finger"],
                "SBP": sbp,
                "DBP": dbp,
            })

    return pd.DataFrame(records)


def build_summary_table(df_detail: pd.DataFrame) -> pd.DataFrame:
    """Aggregate posture-level Mean +/- SD statistics from df_detail formatted with mathematical minus."""
    summary_rows = []
    for p_id, config_desc, _, _ in POSTURES:
        sub = df_detail[df_detail["Posture"] == p_id]
        if sub.empty:
            continue

        def fmt_msd(series: pd.Series) -> str:
            mean_val = series.mean()
            sd_val = series.std(ddof=1)
            if np.isclose(mean_val, 0.0, atol=1e-9):
                mean_val = 0.0
            if np.isclose(sd_val, 0.0, atol=1e-9):
                sd_val = 0.0
            return format_math_minus(f"{mean_val:.2f} ± {sd_val:.2f}")

        summary_rows.append({
            "Posture": p_id,
            "Arm Configuration": config_desc,
            "Estimated Δh_heart-cuff (cm)": fmt_msd(sub["dh_heart_cuff"]),
            "Estimated Δh_heart-finger (cm)": fmt_msd(sub["dh_heart_finger"]),
            "Cuff-to-Finger Vertical Offset Δh_cuff-finger (cm)": fmt_msd(sub["dh_cuff_finger"]),
            "Theoretical Cuff Hydrostatic Offset (mmHg)": fmt_msd(sub["dp_cuff"]),
            "Theoretical Finger Hydrostatic Offset (mmHg)": fmt_msd(sub["dp_finger"]),
            "Measured Cuff SBP (mmHg)": fmt_msd(sub["SBP"]),
            "Measured Cuff DBP (mmHg)": fmt_msd(sub["DBP"]),
        })

    return pd.DataFrame(summary_rows)


def export_to_excel(df_summary: pd.DataFrame, df_detail: pd.DataFrame, out_path: Path) -> None:
    """Save summary (top section) and individual subject details (bottom section) to a formatted Excel file."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Hydrostatic_BP_Report"

    font_title = Font(name="Arial", size=12, bold=True, color="1F4E79")
    font_header = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    font_cell = Font(name="Arial", size=10, bold=False, color="000000")
    font_bold = Font(name="Arial", size=10, bold=True, color="000000")

    fill_header_sum = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    fill_header_det = PatternFill(start_color="2F5597", end_color="2F5597", fill_type="solid")
    fill_zebra = PatternFill(start_color="F2F5F9", end_color="F2F5F9", fill_type="solid")

    thin_side = Side(border_style="thin", color="D9D9D9")
    med_top = Side(border_style="medium", color="1F4E79")
    med_bottom = Side(border_style="medium", color="1F4E79")
    border_data = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
    border_header = Border(left=thin_side, right=thin_side, top=med_top, bottom=med_bottom)

    align_center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    align_left = Alignment(horizontal="left", vertical="center", wrap_text=True)

    # Excel custom number formats using standard mathematical minus sign (U+2212)
    num_fmt_2dec = f"0.00;{MATH_MINUS}0.00;0.00"
    num_fmt_1dec = f"0.0;{MATH_MINUS}0.0;0.0"
    num_fmt_int = f"0;{MATH_MINUS}0;0"

    # --- Section 1: Summary Statistics ---
    n_subjects = df_detail["Subject_ID"].nunique()
    ws.cell(row=1, column=1, value=f"Part 1: Group-Level Posture Summary Statistics (N = {n_subjects} Subjects, Mean ± SD)").font = font_title
    ws.row_dimensions[1].height = 24

    sum_headers = list(df_summary.columns)
    header_row_1 = 2
    ws.row_dimensions[header_row_1].height = 32
    for col_idx, h_text in enumerate(sum_headers, start=1):
        cell = ws.cell(row=header_row_1, column=col_idx, value=h_text)
        cell.font = font_header
        cell.fill = fill_header_sum
        cell.alignment = align_center
        cell.border = border_header

    curr_row = header_row_1 + 1
    for r_idx, row_data in df_summary.iterrows():
        ws.row_dimensions[curr_row].height = 22
        for col_idx, col_name in enumerate(sum_headers, start=1):
            cell = ws.cell(row=curr_row, column=col_idx, value=row_data[col_name])
            cell.font = font_bold if col_idx == 1 else font_cell
            cell.alignment = align_left if col_idx == 2 else align_center
            cell.border = border_data
            if r_idx % 2 == 1:
                cell.fill = fill_zebra
        curr_row += 1

    ws.cell(
        row=curr_row,
        column=1,
        value=f"Note: Vertical heights relative to heart level (right atrium) derived via anthropometric proportions (Winter, 2009). Hydrostatic factor = {MATH_MINUS}0.78 mmHg/cm (Netea et al., 2003)."
    ).font = Font(name="Arial", size=9, italic=True, color="595959")
    curr_row += 2

    # --- Section 2: Subject-Level Detailed Records ---
    ws.cell(row=curr_row, column=1, value="Part 2: Subject-Level Detailed Posture Measurements").font = font_title
    ws.row_dimensions[curr_row].height = 24
    curr_row += 1

    det_columns = [
        ("Subject_ID", "Subject ID"),
        ("Posture", "Posture"),
        ("Arm Configuration", "Arm Configuration"),
        ("Height_cm", "Height (cm)"),
        ("dh_heart_cuff", "Estimated Δh_heart-cuff (cm)"),
        ("dh_heart_finger", "Estimated Δh_heart-finger (cm)"),
        ("dh_cuff_finger", "Cuff-to-Finger Vertical Offset Δh_cuff-finger (cm)"),
        ("dp_cuff", "Theoretical Cuff Hydrostatic Offset (mmHg)"),
        ("dp_finger", "Theoretical Finger Hydrostatic Offset (mmHg)"),
        ("SBP", "Measured Cuff SBP (mmHg)"),
        ("DBP", "Measured Cuff DBP (mmHg)"),
    ]

    ws.row_dimensions[curr_row].height = 32
    for col_idx, (_, h_text) in enumerate(det_columns, start=1):
        cell = ws.cell(row=curr_row, column=col_idx, value=h_text)
        cell.font = font_header
        cell.fill = fill_header_det
        cell.alignment = align_center
        cell.border = border_header
    curr_row += 1

    for r_idx, row_data in df_detail.iterrows():
        ws.row_dimensions[curr_row].height = 20
        subj_group_idx = r_idx // 5
        for col_idx, (col_key, _) in enumerate(det_columns, start=1):
            val = row_data[col_key]
            if col_key in ("SBP", "DBP"):
                val = int(round(val))
            cell = ws.cell(row=curr_row, column=col_idx, value=val)
            cell.font = font_cell
            cell.alignment = align_left if col_key in ("Subject_ID", "Arm Configuration") else align_center
            cell.border = border_data
            if col_key in ("SBP", "DBP"):
                cell.number_format = num_fmt_int
            elif col_key == "Height_cm":
                cell.number_format = num_fmt_1dec
            elif isinstance(val, (int, float)):
                cell.number_format = num_fmt_2dec
            if subj_group_idx % 2 == 1:
                cell.fill = fill_zebra
        curr_row += 1

    col_widths = [18, 46, 26, 26, 32, 28, 28, 24, 24, 22, 22]
    for col_idx, width in enumerate(col_widths, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    ws.views.sheetView[0].showGridLines = True
    wb.save(out_path)


def main():
    base_dir = Path("/Users/ziyiwang/Documents/Project_2026_month1_6/Work_260223_continuous_BP_monitoring/OUTPUT")
    output_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()

    exclude_dirs = {
        "yy04_subject_000",
        "yy04_subject_001",
        "yy04_subject_002",
        "yy04_subject_003",
        "yy04_subject_004",
    }

    df_detail = load_subject_records(base_dir, exclude_dirs)
    df_summary = build_summary_table(df_detail)

    output_xlsx = output_dir / "posture_hydrostatic_bp_report.xlsx"
    export_to_excel(df_summary, df_detail, output_xlsx)
    print(f"Processed {df_detail['Subject_ID'].nunique()} subjects ({len(df_detail)} posture records).")
    print(f"Saved report to: {output_xlsx}")


if __name__ == "__main__":
    main()