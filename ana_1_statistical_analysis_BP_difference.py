import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestPower
from statsmodels.stats.anova import AnovaRM
from pathlib import Path
from itertools import combinations
import csv

def format_sci(val):
    if pd.isna(val):
        return "NaN"
    sci_str = f"{val:.2e}"
    base, exp = sci_str.split('e')
    exp = int(exp)
    base = base.replace('-', '−')
    exp_str = str(exp).replace('-', '−')
    superscript_map = str.maketrans("−0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    exp_super = exp_str.translate(superscript_map)
    return f"{base} × 10{exp_super}"

def main():
    base_dir = Path("/Users/ziyiwang/Documents/Work_continuous_BP_monitoring/OUTPUT")
    output_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
    
    exclude_dirs = {
        "yy04_subject_000",
        "yy04_subject_001",
        "yy04_subject_002",
        "yy04_subject_003",
        "yy04_subject_004",
    }
    
    posture_labels = ["(S1, −45°)", "(S2, 0°)", "(S3, 90°)", "(S4, −90°)", "(S5, −45°)"]
    
    sbp_list = []
    dbp_list = []
    map_list = []
    
    if not base_dir.exists():
        print(f"Error: folder {base_dir} not exist.")
        return
        
    for subject_dir in base_dir.iterdir():
        if subject_dir.is_dir() and subject_dir.name not in exclude_dirs:
            csv_file = subject_dir / "all_trimmed.csv"
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    if df.shape[1] >= 2 and len(df) >= 5:
                        sbp_vals = df.iloc[:5, 0].astype(float).values
                        dbp_vals = df.iloc[:5, 1].astype(float).values
                        map_vals = (1 / 3) * sbp_vals + (2 / 3) * dbp_vals
                        
                        sbp_list.append(sbp_vals)
                        dbp_list.append(dbp_vals)
                        map_list.append(map_vals)
                except Exception:
                    pass
                    
    if not sbp_list:
        print("No valid data found.")
        return
        
    sbp_mat = np.vstack(sbp_list)
    dbp_mat = np.vstack(dbp_list)
    map_mat = np.vstack(map_list)
    
    targets = {"SBP": sbp_mat, "DBP": dbp_mat, "MAP": map_mat}
    
    pairs = list(combinations(range(5), 2))
    num_comp = len(pairs)
    
    out_csv = output_dir / "posture_statistical_analysis.csv"
    out_xlsx = output_dir / "posture_statistical_analysis.xlsx"
    
    master_excel_rows = []
    excel_headers = [
        "p1", "p2", "Mean_Diff", "Diff (p1 −p2), 95% CI",
        "ES (Cohen's \"d\" )", "Statistical power",
        "p (raw)", "p (corrected)"
    ]
    
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        
        for idx, (target_name, mat) in enumerate(targets.items()):
            n_subjects, n_postures = mat.shape
            
            is_normal = True
            for i in range(n_postures):
                _, p_norm = stats.shapiro(mat[:, i])
                if p_norm < 0.05:
                    is_normal = False
                    
            stat_f, p_f = stats.friedmanchisquare(*[mat[:, i] for i in range(n_postures)])
            
            df_long = pd.DataFrame({
                'Subject': np.repeat(np.arange(n_subjects), n_postures),
                'Posture': np.tile(np.arange(n_postures), n_subjects),
                'Value': mat.flatten()
            })
            
            try:
                rm_anova = AnovaRM(data=df_long, depvar='Value', subject='Subject', within=['Posture']).fit()
                anova_stat = rm_anova.anova_table['F Value']['Posture']
                anova_p = rm_anova.anova_table['Pr > F']['Posture']
                anova_df1 = int(rm_anova.anova_table['Num DF']['Posture'])
                anova_df2 = int(rm_anova.anova_table['Den DF']['Posture'])
            except Exception:
                anova_stat, anova_p, anova_df1, anova_df2 = np.nan, np.nan, np.nan, np.nan
            
            writer.writerow([f"==================== {target_name} Overall Repeated-Measures Test ===================="])
            writer.writerow([
                "Test_Type", "Statistic", "p_value", 
                "Degrees_of_Freedom", "Normality_Assumption", "Overall_Conclusion"
            ])
            
            writer.writerow([
                "Repeated-Measures ANOVA", 
                f"{anova_stat:.4f}" if pd.notna(anova_stat) else "NaN", 
                f"{anova_p:.4e}" if pd.notna(anova_p) else "NaN", 
                f"({anova_df1}, {anova_df2})" if pd.notna(anova_df1) else "NaN", 
                "Met" if is_normal else "Violated", 
                "Significant (Reject H0)" if pd.notna(anova_p) and anova_p < 0.05 else "Not Significant"
            ])
            
            writer.writerow([
                "Friedman Test", 
                f"{stat_f:.4f}", 
                f"{p_f:.4e}", 
                f"{n_postures - 1}", 
                "Non-parametric (N/A)", 
                "Significant (Reject H0)" if p_f < 0.05 else "Not Significant"
            ])
            
            writer.writerow([])
            
            writer.writerow([f"-------------------- {target_name} Posture Pairwise Comparisons (Post-hoc) --------------------"])
            writer.writerow([
                "Posture_A", "Posture_B", 
                "Mean_Diff", "SD_Diff", 
                "p_value", "p_bonferroni", 
                "95%_CI_Lower", "95%_CI_Upper", 
                "Cohens_dz", "Statistical_Power"
            ])
            
            if idx > 0:
                master_excel_rows.append([""] * len(excel_headers))
                master_excel_rows.append([""] * len(excel_headers))
            master_excel_rows.append(excel_headers)
            
            for p_a, p_b in pairs:
                vec_a = mat[:, p_a]
                vec_b = mat[:, p_b]
                diff = vec_a - vec_b
                n = len(diff)
                
                mean_diff = np.mean(diff)
                std_diff = np.std(diff, ddof=1)
                
                _, p_val = stats.ttest_rel(vec_a, vec_b)
                
                p_bonf = min(p_val * num_comp, 1.0)
                
                sem = std_diff / np.sqrt(n)
                ci_low, ci_high = stats.t.interval(0.95, n-1, loc=mean_diff, scale=sem)
                
                cohen_d = mean_diff / std_diff if std_diff != 0 else np.nan
                
                if not np.isnan(cohen_d) and std_diff != 0:
                    power = TTestPower().solve_power(effect_size=abs(cohen_d), nobs=n, alpha=0.05, alternative='two-sided')
                else:
                    power = np.nan
                    
                writer.writerow([
                    posture_labels[p_a], 
                    posture_labels[p_b], 
                    f"{mean_diff:.4f}", 
                    f"{std_diff:.4f}", 
                    f"{p_val:.4e}", 
                    f"{p_bonf:.4e}", 
                    f"{ci_low:.4f}", 
                    f"{ci_high:.4f}", 
                    f"{cohen_d:.4f}", 
                    f"{power:.4f}"
                ])
                
                p1_val = posture_labels[p_a]
                p2_val = posture_labels[p_b]
                mean_diff_str = f"{mean_diff:.2f}".replace("-", "−")
                ci_str = f"[{ci_low:.2f}, {ci_high:.2f}]".replace("-", "−")
                es_str = f"{cohen_d:.3f}".replace("-", "−") if not np.isnan(cohen_d) else "NaN"
                pow_str = f"{power:.3f}" if not np.isnan(power) else "NaN"
                p_raw_str = format_sci(p_val)
                p_bonf_str = format_sci(p_bonf)
                
                master_excel_rows.append([
                    p1_val, p2_val, mean_diff_str, ci_str, 
                    es_str, pow_str, p_raw_str, p_bonf_str
                ])
                
            writer.writerow([])
            writer.writerow([])
            
    df_xls = pd.DataFrame(master_excel_rows)
    df_xls.to_excel(out_xlsx, index=False, header=False)
            
    print(f"Statistical analysis completed. Results saved to:\n{out_csv}\n{out_xlsx}")

if __name__ == "__main__":
    main()