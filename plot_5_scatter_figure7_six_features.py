import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, t

# ----------------- Global Style Configuration -----------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["axes.unicode_minus"] = False


def format_ticks(val, pos):
    """Format tick labels: display 0.0 as 0 and use standard Unicode minus."""
    if np.isclose(val, 0.0):
        return "0"
    val_str = f"{val:g}"
    return val_str.replace("-", "\u2212")


def format_stat_iqr(med, q1, q3):
    """Format median and IQR: Median [Q1, Q3] with standard Unicode minus."""
    med_str = f"{med:.2f}".replace("-", "\u2212")
    q1_str = f"{q1:.2f}".replace("-", "\u2212")
    q3_str = f"{q3:.2f}".replace("-", "\u2212")
    return f"{med_str} [{q1_str}, {q3_str}]"


def get_flush_ticks(x_vals, max_ticks=6):
    """Calculate neat ticks covering the data and flush with borders without gaps."""
    valid = x_vals[~np.isnan(x_vals)]
    vmin, vmax = np.nanmin(valid), np.nanmax(valid)
    span = vmax - vmin if vmax > vmin else 1.0

    rough_step = span / 4.0
    order = 10 ** np.floor(np.log10(rough_step))
    candidates = [
        1.0 * order,
        2.0 * order,
        2.5 * order,
        5.0 * order,
        10.0 * order,
        20.0 * order,
    ]

    selected_ticks = None
    for step in candidates:
        step = float(f"{step:.6g}")
        start = np.floor(vmin / step) * step
        end = np.ceil(vmax / step) * step
        n_ticks = int(np.round((end - start) / step)) + 1
        if n_ticks <= max_ticks:
            selected_ticks = np.linspace(start, end, n_ticks)
            break

    if selected_ticks is None:
        step = 50.0 * order
        start = np.floor(vmin / step) * step
        end = np.ceil(vmax / step) * step
        n_ticks = int(np.round((end - start) / step)) + 1
        selected_ticks = np.linspace(start, end, n_ticks)

    selected_ticks = np.where(np.isclose(selected_ticks, 0.0, atol=1e-12), 0.0, selected_ticks)
    return selected_ticks


# ----------------- Path & Column Setup -----------------
base_dir = Path("/Users/ziyiwang/Documents/Project_2026_month1_6/Work_260223_continuous_BP_monitoring/OUTPUT")
output_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()

exclude_dirs = {
    "yy04_subject_000",
    "yy04_subject_001",
    "yy04_subject_002",
    "yy04_subject_003",
    "yy04_subject_004",
}

# 1-based column positions mapped to 0-based DataFrame indices[cite: 8, 9, 12, 13]
col_map = {
    "delta_SBP": 0,       # Column 1 (SBP change)
    "Tm_wb": 27,          # Column 28
    "AI_OD_AMS": 46,      # Column 47
    "b/a": 63,            # Column 64
    "IPA": 112,           # Column 113
    "Skewness": 164,      # Column 165
    "Kurtosis": 165,      # Column 166
}

feature_configs = [
    {"key": "Skewness", "x_label": "\u2206 Skewness", "filename": "scatter_delta_Skewness_vs_delta_SBP.png"},
    {"key": "Kurtosis", "x_label": "\u2206 Kurtosis", "filename": "scatter_delta_Kurtosis_vs_delta_SBP.png"},
    {"key": "IPA", "x_label": "\u2206 IPA", "filename": "scatter_delta_IPA_vs_delta_SBP.png"},
    {"key": "b/a", "x_label": "\u2206 b/a", "filename": "scatter_delta_b_a_vs_delta_SBP.png"},
    {"key": "Tm_wb", "x_label": "\u2206 Tm_wb", "filename": "scatter_delta_Tm_wb_vs_delta_SBP.png"},
    {"key": "AI_OD_AMS", "x_label": "\u2206 AI_OD_AMS", "filename": "scatter_delta_AI_OD_AMS_vs_delta_SBP.png"},
]

# ----------------- Data Loading & Delta Calculation -----------------
delta_list = []
subject_id_list = []

sub_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name not in exclude_dirs])

for folder in sub_dirs:
    csv_file = folder / "all_trimmed.csv"
    if not csv_file.exists():
        continue

    df_sub = pd.read_csv(csv_file)
    if len(df_sub) < 5:
        continue

    # Convert first 5 rows to numeric values
    df_numeric = df_sub.iloc[:5].apply(pd.to_numeric, errors="coerce")

    # Row 0 is baseline; rows 1 to 4 are changes relative to baseline
    df_delta = df_numeric.iloc[1:5].sub(df_numeric.iloc[0], axis=1)

    delta_list.append(df_delta)
    subject_id_list.extend([folder.name] * len(df_delta))

if not delta_list:
    raise FileNotFoundError("No valid all_trimmed.csv files found in the specified directory.")

df_all = pd.concat(delta_list, ignore_index=True)
subject_ids = np.array(subject_id_list)

print(f"Loaded {len(df_all)} delta samples from {len(np.unique(subject_ids))} subjects.")

# Extract unified delta_SBP
delta_sbp = df_all.iloc[:, col_map["delta_SBP"]].to_numpy(dtype=float)

# Calculate unified y-axis range across all figures[cite: 8, 9, 12, 13]
y_min_val, y_max_val = np.nanmin(delta_sbp), np.nanmax(delta_sbp)
y_padding = (y_max_val - y_min_val) * 0.08
y_lim = (
    np.floor((y_min_val - y_padding) / 10.0) * 10.0,
    np.ceil((y_max_val + y_padding) / 10.0) * 10.0,
)

# ----------------- Plotting Figures -----------------
for feat in feature_configs:
    col_idx = col_map[feat["key"]]
    x_vals = df_all.iloc[:, col_idx].to_numpy(dtype=float)

    # 1. Participant-wise correlation calculation (4 samples per subject)
    r_p_list = []
    r_s_list = []
    for sid in np.unique(subject_ids):
        mask = (subject_ids == sid)
        xs = x_vals[mask]
        ys = delta_sbp[mask]
        sub_valid = ~(np.isnan(xs) | np.isnan(ys))
        if np.sum(sub_valid) >= 3 and np.std(xs[sub_valid]) > 1e-12 and np.std(ys[sub_valid]) > 1e-12:
            rp, _ = pearsonr(xs[sub_valid], ys[sub_valid])
            rs, _ = spearmanr(xs[sub_valid], ys[sub_valid])
            if not np.isnan(rp):
                r_p_list.append(rp)
            if not np.isnan(rs):
                r_s_list.append(rs)

    # Calculate Median, Q1 (25th percentile), and Q3 (75th percentile)
    r_p_med = np.median(r_p_list)
    r_p_q1, r_p_q3 = np.percentile(r_p_list, [25, 75])

    r_s_med = np.median(r_s_list)
    r_s_q1, r_s_q3 = np.percentile(r_s_list, [25, 75])

    # Overall valid sample points count
    valid_mask = ~(np.isnan(x_vals) | np.isnan(delta_sbp))
    n_samples = int(np.sum(valid_mask))
    x_clean = x_vals[valid_mask]
    y_clean = delta_sbp[valid_mask]

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=300)

    # Background grid lines[cite: 10, 12, 13]
    ax.grid(True, linestyle="--", linewidth=0.8, color="#e0e0e0", alpha=0.8)
    ax.set_axisbelow(True)

    # Zero reference lines matching grid appearance[cite: 10, 11, 12, 13]
    ax.axhline(0, color="#e0e0e0", linestyle="--", linewidth=0.8, alpha=0.8, zorder=0)
    ax.axvline(0, color="#e0e0e0", linestyle="--", linewidth=0.8, alpha=0.8, zorder=0)

    # Scatter points with enlarged size[cite: 10, 11, 12, 13]
    ax.scatter(
        x_vals,
        delta_sbp,
        color="#2b5c8f",
        alpha=0.6,
        s=70,
        edgecolors="none",
        zorder=2,
    )

    # Compute flush x-ticks and strictly set x-limits
    x_ticks = get_flush_ticks(x_vals)
    ax.set_xlim(x_ticks[0], x_ticks[-1])
    ax.set_xticks(x_ticks)

    # 2. Linear regression fit and 95% confidence bounds (Dehghanojamahalleh et al. Fig. 4 style)
    poly_coeffs = np.polyfit(x_clean, y_clean, deg=1)
    poly = np.poly1d(poly_coeffs)

    x_grid = np.linspace(x_ticks[0], x_ticks[-1], 200)
    y_grid = poly(x_grid)

    n_fit = len(x_clean)
    residuals = y_clean - poly(x_clean)
    s_err = np.sqrt(np.sum(residuals**2) / (n_fit - 2))
    t_crit = t.ppf(0.975, df=n_fit - 2)
    ci = t_crit * s_err * np.sqrt(1.0 / n_fit + (x_grid - np.mean(x_clean))**2 / np.sum((x_clean - np.mean(x_clean))**2))
    upper_bound = y_grid + ci
    lower_bound = y_grid - ci

    # Plot solid linear regression fit line and dotted 95% confidence bounds
    ax.plot(x_grid, y_grid, color="#d62728", linestyle="-", linewidth=2.0, label="Fit", zorder=4)
    ax.plot(x_grid, upper_bound, color="#d62728", linestyle=":", linewidth=1.6, label="Confidence bounds", zorder=4)
    ax.plot(x_grid, lower_bound, color="#d62728", linestyle=":", linewidth=1.6, zorder=4)

    # Legend located at the bottom-right
    ax.legend(
    loc="lower right",
    frameon=False,
    fontsize=15,
    )

    # Upper right statistics text showing Median [Q1, Q3] and sample number
    stats_text = (
        f"$\\rho_p$: {format_stat_iqr(r_p_med, r_p_q1, r_p_q3)}\n"
        f"$\\rho_s$: {format_stat_iqr(r_s_med, r_s_q1, r_s_q3)}\n"
        f"sample number = {n_samples}"
    )
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=18,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="square,pad=0.35", facecolor="none", edgecolor="none"),
        zorder=3,
    )

    # Axes configuration
    ax.set_xlabel(feat["x_label"], fontsize=18, fontweight="bold", labelpad=8)
    ax.set_ylabel("\u2206 SBP (mmHg)", fontsize=18, fontweight="bold", labelpad=8)
    ax.set_ylim(y_lim)

    ax.xaxis.set_major_formatter(FuncFormatter(format_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))
    ax.tick_params(axis="both", direction="in", top=True, right=True, length=4.5, width=1.0)

    plt.tight_layout()
    out_file = output_dir / feat["filename"]
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file.resolve()}")