import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ----------------- Global Style Configuration -----------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.unicode_minus"] = False  

# ----------------- Path & Parameter Setup -----------------
csv_path = "/Users/ziyiwang/Documents/Work_continuous_BP_monitoring/OUTPUT/yy04_subject_015/all_trimmed.csv"

output_dir = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in locals()
    else os.getcwd()
)

x_labels = ["S1 (\u221245°)", "S2 (0°)", "S3 (90°)", "S4 (\u221290°)", "S5 (\u221245°)"]
x_indices = np.arange(len(x_labels))

col_map = {
    "SBP": 0,
    "DBP": 1,
    "b/a": 63,
    "d/a": 65,
    "IPA": 112,
    "Skewness": 164,
    "Kurtosis": 165,
}

df = pd.read_csv(csv_path)
data = {key: df.iloc[:5, idx].to_numpy(dtype=float) for key, idx in col_map.items()}

# ----------------- Formatting Function -----------------
def apply_rachim_style(ax, y_label):
    """Apply styling matching Rachim et al. Figure 3."""
    ax.set_xlim(0, len(x_labels) - 1)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, fontsize=13, fontweight="normal")

    ax.set_ylabel(y_label, fontsize=13, fontweight="normal")
    ax.tick_params(
        axis="both",
        direction="in",
        top=True,
        right=True,
        length=3.0,
        width=0.8,
        labelsize=12,
    )

    ax.grid(True, linestyle="-", linewidth=0.6, color="#d0d0d0", alpha=0.8)
    ax.set_axisbelow(True)


def format_negative_ticks(ax):
    """Force standard minus sign (\u2212) on y-axis tick labels."""
    def fmt(x, pos):
        if x < 0:
            return "\u2212{:.2f}".format(abs(x)).rstrip("0").rstrip(".")
        elif x == 0:
            return "0"
        else:
            return "{:.2f}".format(x).rstrip("0").rstrip(".")
    # Use 2 decimal places; trailing zeros stripped for a clean look
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))


# =======================================================================
# 1. Combined SBP & DBP Plot
# =======================================================================
fig, ax = plt.subplots(figsize=(6.2, 3.5))

ax.plot(
    x_indices, data["SBP"],
    color="black", linestyle="-", linewidth=1.2,
    marker="o", markersize=6.5,
    markerfacecolor="black", markeredgecolor="black", label="SBP",
)

ax.plot(
    x_indices, data["DBP"],
    color="black", linestyle="--", linewidth=1.2,
    marker="s", markersize=6.5,
    markerfacecolor="black", markeredgecolor="black", label="DBP",
)

apply_rachim_style(ax, "BP (mmHg)")

# BP ylim: 50 – 140; ticks from 60 to 140, step 20
ax.set_ylim(50, 140)
ax.set_yticks([60, 80, 100, 120, 140])

ax.legend(
    loc="upper right",
    frameon=True, edgecolor="black", facecolor="white",
    framealpha=1.0, fancybox=False, fontsize=12,
    handletextpad=0.4,
)

plt.tight_layout()
fig_path_bp = os.path.join(output_dir, "yy04_subject_005_1_BP.png")
fig.savefig(fig_path_bp, dpi=300)
plt.close(fig)
print(f"Saved: {fig_path_bp}")


# =======================================================================
# 2. Individual Feature Plots
# =======================================================================
single_features = [
    ("IPA",      "IPA",      0.0,  0.3,  0.05),
    ("Kurtosis", "Kurtosis", -1.5, -0.5, 0.25),
    ("Skewness", "Skewness", -0.5, 0.5,  0.25),
    ("b/a",      "b/a",      -1.3, -0.7, 0.1),
    ("d/a",      "d/a",      -0.3, 0.1,  0.1),  
]

output_suffix = {
    "IPA": "2_IPA.png",
    "Kurtosis": "3_Kurtosis.png",
    "Skewness": "4_Skewness.png",
    "b/a": "5_b_a.png",
    "d/a": "6_d_a.png",
}

for feat_key, display_name, y_min, y_max, y_step in single_features:
    fig, ax = plt.subplots(figsize=(6.2, 3.5))

    ax.plot(
        x_indices, data[feat_key],
        color="black", linestyle="-", linewidth=1.2,
        marker="s", markersize=6.5,
        markerfacecolor="black", markeredgecolor="black", label=display_name,
    )

    apply_rachim_style(ax, "Value")

   
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(y_min, y_max + y_step * 0.5, y_step))

   
    format_negative_ticks(ax)

    
    if feat_key == "IPA":
        ax.set_yticklabels(["0" if t == 0 else f"{t:.2f}" for t in ax.get_yticks()])

    ax.legend(
        loc="upper right",
        frameon=True, edgecolor="black", facecolor="white",
        framealpha=1.0, fancybox=False, fontsize=12,
        handletextpad=0.5,
    )

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"yy04_subject_005_{output_suffix[feat_key]}")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig_path}")