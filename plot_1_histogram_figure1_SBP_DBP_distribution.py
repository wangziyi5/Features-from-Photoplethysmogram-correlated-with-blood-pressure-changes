import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 18
plt.rcParams["axes.unicode_minus"] = False 

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

    sbp_data = []
    dbp_data = []
    map_data = []
    delta_sbp_data = []
    delta_dbp_data = []
    delta_map_data = []
    height_data = []
    weight_data = []
    bmi_data = []
    gender_data = []
    age_data = []

    if not base_dir.exists():
        print(f"Error: folder {base_dir} not exist. Please check the path.")
        return

    print("starting to read all_trimmed.csv files...")
    for subject_dir in base_dir.iterdir():
        if subject_dir.is_dir() and subject_dir.name not in exclude_dirs:
            csv_file = subject_dir / "all_trimmed.csv"
            
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    if df.shape[1] >= 10:
                        sbp_vals = df.iloc[:5, 0].astype(float).values
                        dbp_vals = df.iloc[:5, 1].astype(float).values
                        map_vals = (1 / 3) * sbp_vals + (2 / 3) * dbp_vals
                        
                        sbp_data.extend(sbp_vals)
                        dbp_data.extend(dbp_vals)
                        map_data.extend(map_vals)
                        
                        delta_sbp_data.extend(sbp_vals[1:5] - sbp_vals[0])
                        delta_dbp_data.extend(dbp_vals[1:5] - dbp_vals[0])
                        delta_map_data.extend(map_vals[1:5] - map_vals[0])
                        
                        height_data.append(df.iloc[0, 5])
                        weight_data.append(df.iloc[0, 6])
                        bmi_data.append(df.iloc[0, 7])
                        gender_data.append(df.iloc[0, 8])
                        age_data.append(df.iloc[0, 9])
                except Exception as e:
                    print(f"Error reading file {csv_file.name}: {e}")

    sbp_array = np.array(sbp_data, dtype=float)
    dbp_array = np.array(dbp_data, dtype=float)
    map_array = np.array(map_data, dtype=float)
    
    delta_sbp_array = np.array(delta_sbp_data, dtype=float)
    delta_dbp_array = np.array(delta_dbp_data, dtype=float)
    delta_map_array = np.array(delta_map_data, dtype=float)
    
    abs_delta_sbp_array = np.abs(delta_sbp_array)
    abs_delta_dbp_array = np.abs(delta_dbp_array)
    abs_delta_map_array = np.abs(delta_map_array)
    
    height_array = np.array(height_data, dtype=float)
    weight_array = np.array(weight_data, dtype=float)
    bmi_array = np.array(bmi_data, dtype=float)
    gender_array = np.array(gender_data, dtype=float)
    age_array = np.array(age_data, dtype=float)

    print(f"Successfully read data: SBP sample number = {len(sbp_array)}, DBP sample number = {len(dbp_array)}")

    def plot_histogram(data, label, color, filename, x_min, x_max, y_max):
        if len(data) == 0:
            print(f"No data available for plotting {label} histogram")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        n_val = len(data)
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        
        bins_array = np.arange(x_min, x_max + 2.5, 2.5)
        
        ax.hist(data, bins=bins_array, color=color, edgecolor='black', alpha=0.75, zorder=3)
        
        ax.set_xlabel(f"{label} (mmHg)")
        ax.set_ylabel("sample number")
        
        ax.margins(0) 
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)
        ax.set_yticks(np.arange(0, y_max + 1, 10))
        
        xticks_array = np.arange(x_min, x_max + 10, 10)
        ax.set_xticks(xticks_array)
        
        ax.tick_params(axis='both', direction='in')
        
        ax.grid(True, linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        
        legend_text = f"mean ± standard deviation = {mean_val:.2f} ± {std_val:.2f}\nsample size = {n_val}"
        
        ax.plot([], [], ' ', label=legend_text)
        
        legend = ax.legend(loc="upper right", frameon=False)
        legend.get_frame().set_alpha(0)
        
        plt.tight_layout()
        
        out_path = output_dir / filename
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Successfully generated and saved chart: {out_path}")

    plot_histogram(sbp_array, "SBP", "#1f77b4", "SBP_Histogram.png", 60, 150, 40)
    plot_histogram(dbp_array, "DBP", "#1f77b4", "DBP_Histogram.png", 40, 110, 60)

    total_subjects = len(gender_array)
    if total_subjects > 0:
        males = np.sum(gender_array == 0)
        females = np.sum(gender_array == 1)
        male_ratio = (males / total_subjects) * 100
        female_ratio = (females / total_subjects) * 100
        
        age_mean, age_std = np.mean(age_array), np.std(age_array, ddof=1)
        height_mean, height_std = np.mean(height_array), np.std(height_array, ddof=1)
        weight_mean, weight_std = np.mean(weight_array), np.std(weight_array, ddof=1)
        bmi_mean, bmi_std = np.mean(bmi_array), np.std(bmi_array, ddof=1)
        
        sbp_mean, sbp_std = np.mean(sbp_array), np.std(sbp_array, ddof=1)
        dbp_mean, dbp_std = np.mean(dbp_array), np.std(dbp_array, ddof=1)
        map_mean, map_std = np.mean(map_array), np.std(map_array, ddof=1)
        
        d_sbp_med = np.median(delta_sbp_array)
        d_sbp_q1 = np.percentile(delta_sbp_array, 25)
        d_sbp_q3 = np.percentile(delta_sbp_array, 75)
        
        d_dbp_med = np.median(delta_dbp_array)
        d_dbp_q1 = np.percentile(delta_dbp_array, 25)
        d_dbp_q3 = np.percentile(delta_dbp_array, 75)
        
        d_map_med = np.median(delta_map_array)
        d_map_q1 = np.percentile(delta_map_array, 25)
        d_map_q3 = np.percentile(delta_map_array, 75)
        
        abs_d_sbp_mean, abs_d_sbp_std = np.mean(abs_delta_sbp_array), np.std(abs_delta_sbp_array, ddof=1)
        abs_d_dbp_mean, abs_d_dbp_std = np.mean(abs_delta_dbp_array), np.std(abs_delta_dbp_array, ddof=1)
        abs_d_map_mean, abs_d_map_std = np.mean(abs_delta_map_array), np.std(abs_delta_map_array, ddof=1)
        
        print("\n================ Statistical Summary ================")
        print(f"Gender (Male): {males} ({male_ratio:.2f}%)")
        print(f"Gender (Female): {females} ({female_ratio:.2f}%)")
        print(f"Age: {age_mean:.2f} ± {age_std:.2f}")
        print(f"Height (cm): {height_mean:.2f} ± {height_std:.2f}")
        print(f"Weight (kg): {weight_mean:.2f} ± {weight_std:.2f}")
        print(f"BMI: {bmi_mean:.2f} ± {bmi_std:.2f}")
        print(f"SBP: {sbp_mean:.2f} ± {sbp_std:.2f}")
        print(f"DBP: {dbp_mean:.2f} ± {dbp_std:.2f}")
        print(f"MAP: {map_mean:.2f} ± {map_std:.2f}")
        print(f"∆SBP: {d_sbp_med:.2f} [{d_sbp_q1:.2f}, {d_sbp_q3:.2f}]")
        print(f"∆DBP: {d_dbp_med:.2f} [{d_dbp_q1:.2f}, {d_dbp_q3:.2f}]")
        print(f"∆MAP: {d_map_med:.2f} [{d_map_q1:.2f}, {d_map_q3:.2f}]")
        print(f"|∆SBP|: {abs_d_sbp_mean:.2f} ± {abs_d_sbp_std:.2f}")
        print(f"|∆DBP|: {abs_d_dbp_mean:.2f} ± {abs_d_dbp_std:.2f}")
        print(f"|∆MAP|: {abs_d_map_mean:.2f} ± {abs_d_map_std:.2f}")
        print("=====================================================\n")

if __name__ == "__main__":
    main()