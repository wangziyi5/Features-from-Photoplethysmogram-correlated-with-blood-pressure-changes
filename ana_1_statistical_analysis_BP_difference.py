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

def fmt_num(val, fmt=".4f"):
    if pd.isna(val):
        return "NaN"
    return f"{val:{fmt}}".replace('-', '−')

def fmt_sci_csv(val):
    if pd.isna(val):
        return "NaN"
    return f"{val:.4e}".replace('-', '−')

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
        "p1", "p2", "Test_Type", "Mean_Diff", "Diff (p1 −p2), 95% CI",
        "ES (Cohen's \"d\" )", "Statistical power",
        "p (raw)", "p (corrected)"
    ]
    csv_pairwise_headers = [
        "Posture_A", "Posture_B", "Test_Type",
        "Mean_Diff", "SD_Diff", 
        "p_value", "p_bonferroni", 
        "95%_CI_Lower", "95%_CI_Upper", 
        "Cohens_dz", "Statistical_Power"
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
                fmt_num(anova_stat, ".4f") if pd.notna(anova_stat) else "NaN", 
                fmt_sci_csv(anova_p) if pd.notna(anova_p) else "NaN", 
                f"({anova_df1}, {anova_df2})" if pd.notna(anova_df1) else "NaN", 
                "Met" if is_normal else "Violated", 
                "Significant (Reject H0)" if pd.notna(anova_p) and anova_p < 0.05 else "Not Significant"
            ])
            
            writer.writerow([
                "Friedman Test", 
                fmt_num(stat_f, ".4f"), 
                fmt_sci_csv(p_f), 
                f"{n_postures - 1}", 
                "Non-parametric (N/A)", 
                "Significant (Reject H0)" if p_f < 0.05 else "Not Significant"
            ])
            
            writer.writerow([])
            
            
            csv_rows_ttest, csv_rows_wilcoxon, csv_rows_cond = [], [], []
            xls_rows_ttest, xls_rows_wilcoxon, xls_rows_cond = [], [], []
            
            for p_a, p_b in pairs:
                vec_a = mat[:, p_a]
                vec_b = mat[:, p_b]
                diff = vec_a - vec_b
                n = len(diff)
                
                mean_diff = np.mean(diff)
                std_diff = np.std(diff, ddof=1)
                
                sem = std_diff / np.sqrt(n)
                ci_low, ci_high = stats.t.interval(0.95, n - 1, loc=mean_diff, scale=sem)
                cohen_d = mean_diff / std_diff if std_diff != 0 else np.nan
                
                if not np.isnan(cohen_d) and std_diff != 0:
                    power = TTestPower().solve_power(effect_size=abs(cohen_d), nobs=n, alpha=0.05, alternative='two-sided')
                else:
                    power = np.nan

                
                _, p_val_t = stats.ttest_rel(vec_a, vec_b)
                p_bonf_t = min(p_val_t * num_comp, 1.0)
                type_t = "Paired t-test"
                
                
                _, p_val_w = stats.wilcoxon(vec_a, vec_b)
                p_bonf_w = min(p_val_w * num_comp, 1.0)
                type_w = "Wilcoxon signed-rank test"
                
                
                _, p_diff_norm = stats.shapiro(diff)
                if p_diff_norm >= 0.05:
                    type_c = "Paired t-test"
                    p_val_c = p_val_t
                    p_bonf_c = p_bonf_t
                else:
                    type_c = "Wilcoxon signed-rank test"
                    p_val_c = p_val_w
                    p_bonf_c = p_bonf_w
                
                p1_val = posture_labels[p_a]
                p2_val = posture_labels[p_b]
                
                
                m_csv = fmt_num(mean_diff, ".4f")
                s_csv = fmt_num(std_diff, ".4f")
                cil_csv = fmt_num(ci_low, ".4f")
                cih_csv = fmt_num(ci_high, ".4f")
                es_csv = fmt_num(cohen_d, ".4f")
                pow_csv = fmt_num(power, ".4f")
                
                csv_rows_ttest.append([
                    p1_val, p2_val, type_t, m_csv, s_csv,
                    fmt_sci_csv(p_val_t), fmt_sci_csv(p_bonf_t),
                    cil_csv, cih_csv, es_csv, pow_csv
                ])
                csv_rows_wilcoxon.append([
                    p1_val, p2_val, type_w, m_csv, s_csv,
                    fmt_sci_csv(p_val_w), fmt_sci_csv(p_bonf_w),
                    cil_csv, cih_csv, es_csv, pow_csv
                ])
                csv_rows_cond.append([
                    p1_val, p2_val, type_c, m_csv, s_csv,
                    fmt_sci_csv(p_val_c), fmt_sci_csv(p_bonf_c),
                    cil_csv, cih_csv, es_csv, pow_csv
                ])
                
                
                mean_diff_str = fmt_num(mean_diff, ".2f")
                ci_str = f"[{fmt_num(ci_low, '.2f')}, {fmt_num(ci_high, '.2f')}]"
                es_str = fmt_num(cohen_d, ".3f")
                pow_str = fmt_num(power, ".3f")
                
                xls_rows_ttest.append([
                    p1_val, p2_val, type_t, mean_diff_str, ci_str,
                    es_str, pow_str, format_sci(p_val_t), format_sci(p_bonf_t)
                ])
                xls_rows_wilcoxon.append([
                    p1_val, p2_val, type_w, mean_diff_str, ci_str,
                    es_str, pow_str, format_sci(p_val_w), format_sci(p_bonf_w)
                ])
                xls_rows_cond.append([
                    p1_val, p2_val, type_c, mean_diff_str, ci_str,
                    es_str, pow_str, format_sci(p_val_c), format_sci(p_bonf_c)
                ])
            
            
            title_t = f"-------------------- {target_name} Posture Pairwise Comparisons (Post-hoc) [Paired t-test Only] --------------------"
            title_w = f"-------------------- {target_name} Posture Pairwise Comparisons (Post-hoc) [Wilcoxon signed-rank test Only] --------------------"
            title_c = f"-------------------- {target_name} Posture Pairwise Comparisons (Post-hoc) [Normality-Conditional Test] --------------------"
            
            
            writer.writerow([title_t])
            writer.writerow(csv_pairwise_headers)
            writer.writerows(csv_rows_ttest)
            writer.writerow([])
            
            
            writer.writerow([title_w])
            writer.writerow(csv_pairwise_headers)
            writer.writerows(csv_rows_wilcoxon)
            writer.writerow([])
            
            
            writer.writerow([title_c])
            writer.writerow(csv_pairwise_headers)
            writer.writerows(csv_rows_cond)
            writer.writerow([])
            writer.writerow([])
            
            
            if idx > 0:
                master_excel_rows.append([""] * len(excel_headers))
                master_excel_rows.append([""] * len(excel_headers))
                
            
            master_excel_rows.append([title_t] + [""] * (len(excel_headers) - 1))
            master_excel_rows.append(excel_headers)
            master_excel_rows.extend(xls_rows_ttest)
            master_excel_rows.append([""] * len(excel_headers))
            
           
            master_excel_rows.append([title_w] + [""] * (len(excel_headers) - 1))
            master_excel_rows.append(excel_headers)
            master_excel_rows.extend(xls_rows_wilcoxon)
            master_excel_rows.append([""] * len(excel_headers))
            
            
            master_excel_rows.append([title_c] + [""] * (len(excel_headers) - 1))
            master_excel_rows.append(excel_headers)
            master_excel_rows.extend(xls_rows_cond)
            
    df_xls = pd.DataFrame(master_excel_rows)
    df_xls.to_excel(out_xlsx, index=False, header=False)
            
    print(f"Statistical analysis completed. Results saved to:\n{out_csv}\n{out_xlsx}")

if __name__ == "__main__":
    main()