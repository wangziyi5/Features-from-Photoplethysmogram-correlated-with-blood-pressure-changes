import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import openpyxl
import pandas as pd
import shap

# Global font configuration
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = True  
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"


def load_xlsx_feature_list(xlsx_path: Path) -> list[str]:
    wb = openpyxl.load_workbook(xlsx_path, rich_text=True)
    ws = wb.active
    feat_list = []
    for row in range(1, ws.max_row + 1):
        cell = ws.cell(row=row, column=2)
        val = cell.value
        if val is None:
            continue
        cell_italic = bool(cell.font and cell.font.italic)
        if isinstance(val, str):
            text = f"$\\mathit{{{val}}}$" if cell_italic else val
        else:
            parts = []
            for part in val:
                if isinstance(part, str):
                    parts.append(f"$\\mathit{{{part}}}$" if cell_italic else part)
                else:
                    t = part.text
                    sub = part.font and part.font.vertAlign == "subscript"
                    sup = part.font and part.font.vertAlign == "superscript"
                    ita = bool(part.font and part.font.i)
                    if sub:
                        parts.append(f"$_{{\\mathrm{{{t}}}}}$")
                    elif sup:
                        parts.append(f"$^{{\\mathrm{{{t}}}}}$")
                    elif ita:
                        parts.append(f"$\\mathit{{{t}}}$")
                    else:
                        parts.append(t)
            text = "".join(parts)
        feat_list.append(text)
    return feat_list


def format_sbp_ticks(x, pos):
    
    if np.isclose(x, 0.0):
        return "0"
    val_str = f"{x:g}"
    return val_str.replace("-", "\u2212")


def render_beeswarm_plot(
    explanation: shap.Explanation,
    order_indices: np.ndarray,
    target: str,
    max_display: int,
    figsize: tuple[float, float],
    out_path: Path,
):
    """Render and save a single SHAP Beeswarm plot."""
    fig = plt.figure(figsize=figsize)
    shap.plots.beeswarm(
        explanation,
        max_display=max_display,
        order=order_indices,
        show=False,
        plot_size=None,
    )

    fig = plt.gcf()
    ax = plt.gca()

    colorbar_ax = None
    for a in fig.axes:
        if a is not ax and a.get_ylabel() != "":
            colorbar_ax = a
            break

    if target == "SBP":
        if colorbar_ax is not None:
            colorbar_ax.remove()
            colorbar_ax = None

    true_yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    ax.set_yticklabels(["r_OS_OO_ppg"] * len(true_yticklabels))
    ax.set_xlabel(
        f"SHAP value (impact on ∆{target}, mmHg)",
        fontsize=18,
        fontweight="bold",
        labelpad=10,
    )
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    if colorbar_ax is not None:
        colorbar_ax.set_ylabel("Feature value", fontsize=18, fontweight="bold", labelpad=10)
        colorbar_ax.tick_params(labelsize=18)

    plt.tight_layout()
    fig.canvas.draw()

    orig_ax_pos = ax.get_position().bounds
    orig_cb_pos = colorbar_ax.get_position().bounds if colorbar_ax is not None else None
    orig_xlim = ax.get_xlim()
    orig_xticks = [t for t in ax.get_xticks() if orig_xlim[0] - 1e-9 <= t <= orig_xlim[1] + 1e-9]

    ax.set_yticklabels(true_yticklabels)
    ax.set_xticks(orig_xticks)
    ax.set_xlim(orig_xlim)
    ax.set_xlabel(
        f"SHAP value (impact on ∆{target}, mmHg)",
        fontsize=22,
        fontweight="bold",
        labelpad=10,
    )
    ax.tick_params(axis="x", labelsize=22)
    ax.tick_params(axis="y", labelsize=22)

    if target == "SBP":
        ax.xaxis.set_major_formatter(FuncFormatter(format_sbp_ticks))
    else:
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x:g}".replace("-", "\u2212"))
        )

    ax.set_position(orig_ax_pos)
    if colorbar_ax is not None and orig_cb_pos is not None:
        colorbar_ax.set_ylabel("Feature value", fontsize=22, fontweight="bold", labelpad=10)
        colorbar_ax.tick_params(labelsize=22)
        colorbar_ax.set_position(orig_cb_pos)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path.resolve()} (300 DPI)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SHAP Beeswarm Plots aligned with 5-fold CV summary ranking."
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
        help="Path to the merged _feature_and_SHAP_matrix.csv file",
    )
    parser.add_argument(
        "--summary_csv",
        "-s",
        default=None,
        help="Path to the _SHAP_feature_summary.csv file",
    )
    args = parser.parse_args()

    target = args.target.upper()
    n_total_features = 161

    # 1. Resolve CSV input paths
    csv_path = (
        Path(args.matrix_csv)
        if args.matrix_csv
        else Path(
            f"{target}_{n_total_features}_features_SHAP_analysis_5fold_CV_Finnegan_feature_and_SHAP_matrix.csv"
        )
    )
    summary_csv_path = (
        Path(args.summary_csv)
        if args.summary_csv
        else Path(
            f"{target}_{n_total_features}_features_SHAP_analysis_5fold_CV_Finnegan_SHAP_feature_summary.csv"
        )
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Merged CSV matrix file not found: {csv_path.resolve()}"
        )

    print(f"\nLoading merged matrix: {csv_path.resolve()}")
    df_merged = pd.read_csv(csv_path)

    # Validate label column
    if "label" not in df_merged.columns:
        raise ValueError(
            "Missing 'label' column in CSV. Cannot distinguish features from SHAP values."
        )

    # Split into feature matrix and SHAP matrix
    df_feat = df_merged[df_merged["label"] == "feature_delta"].copy()
    df_shap = df_merged[df_merged["label"] == "shap_value"].copy()

    # Align by sample_index
    df_feat = df_feat.sort_values(by="sample_index").reset_index(drop=True)
    df_shap = df_shap.sort_values(by="sample_index").reset_index(drop=True)

    if len(df_feat) != len(df_shap):
        raise ValueError(
            f"Row count mismatch: {len(df_feat)} feature rows vs {len(df_shap)} SHAP rows."
        )

    # Align feature ordering strictly with TXT / summary CSV
    meta_cols = {"sample_index", "subject", "fold", "label"}
    feature_cols = [c for c in df_feat.columns if c not in meta_cols]

    if summary_csv_path.exists():
        print(f"Aligning feature order with summary: {summary_csv_path.resolve()}")
        df_summary = pd.read_csv(summary_csv_path)
        sorted_feature_names = [str(f).strip() for f in df_summary["feature_name"].tolist()]

        X_mat = df_feat[sorted_feature_names].to_numpy(dtype=float)
        shap_mat = df_shap[sorted_feature_names].to_numpy(dtype=float)
        feature_names = sorted_feature_names
        order_scores = df_summary["mean_abs_shap_5fold"].to_numpy(dtype=float)
    else:
        print("Summary CSV not found. Using default column order.")
        feature_names = [str(c).strip() for c in feature_cols]
        X_mat = df_feat[feature_cols].to_numpy(dtype=float)
        shap_mat = df_shap[feature_cols].to_numpy(dtype=float)
        order_scores = np.mean(np.abs(shap_mat), axis=0)

    xlsx_path = Path(__file__).resolve().parent / "supplement_feature_definitions.xlsx"
    if not xlsx_path.exists():
        xlsx_path = Path("supplement_feature_definitions.xlsx")
    if xlsx_path.exists():
        xlsx_feat_list = load_xlsx_feature_list(xlsx_path)
        csv_col_map = {}
        for idx, col_name in enumerate(feature_cols[: len(xlsx_feat_list)]):
            csv_col_map[str(col_name).strip()] = xlsx_feat_list[idx]
            csv_col_map[str(idx + 1)] = xlsx_feat_list[idx]
        feature_names = [csv_col_map.get(str(f).strip(), str(f).strip()) for f in feature_names]

    print(f"Parsed {X_mat.shape[0]} samples across {len(feature_names)} features.")

    # Convert importance scores to descending integer indices
    order_indices = np.argsort(-order_scores)

    # 2. Construct SHAP Explanation object
    explanation = shap.Explanation(
        values=shap_mat,
        data=X_mat,
        feature_names=feature_names,
    )

    # 3. Figure 1: Top 20 Features
    out_top20_path = Path(f"{target}_SHAP_beeswarm_top20.png")
    top20_idx = order_indices[:20]
    render_beeswarm_plot(
        explanation=explanation[:, top20_idx],
        order_indices=np.arange(20),
        target=target,
        max_display=20,
        figsize=(11, 9),
        out_path=out_top20_path,
    )

    # 4. Figure 2: All 161 Features
    out_all_path = Path(f"{target}_SHAP_beeswarm_all{len(feature_names)}.png")
    render_beeswarm_plot(
        explanation=explanation,
        order_indices=order_indices,
        target=target,
        max_display=len(feature_names),
        figsize=(12, 33),
        out_path=out_all_path,
    )

    print("\nAll beeswarm plots successfully generated.")


if __name__ == "__main__":
    main()