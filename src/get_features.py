import os
import pandas as pd
import numpy as np

import glob

TEMPS = [300, 350, 400]

def load_all_eq_std_features(params_df, metrics):
    """Load equilibrium std for all specified metrics."""
    features = {}
    for metric in metrics:
        std_val = load_eq_std(params_df, metric)
        features[metric] = std_val
    return features

def load_all_xvg_std_features(base_path, prefixes, temps):
    """Load std dev from .xvg files for each prefix and temperature."""
    features = {}
    for prefix in prefixes:
        for temp in temps:
            file_path = os.path.join(base_path, f"{prefix}_{temp}.xvg")
            std_val = std_from_xvg(file_path)
            feature_key = f"{prefix}_std_{temp}"
            features[feature_key] = std_val
    return features


def load_xvg_metrics_by_temp(base_path, prefix):
    stds_by_temp = {}
    pattern = os.path.join(base_path, f"{prefix}_*.xvg")
    for file_path in glob.glob(pattern):
        file_name = os.path.basename(file_path)
        try:
            # Example: rmsd_300.xvg or sasa_cdrl2_400.xvg
            temp_str = file_name.replace(".xvg", "").split("_")[-1]
            temp = int(temp_str)
        except ValueError:
            continue  # skip malformed filenames
        std_val = std_from_xvg(file_path)
        key = f"{prefix}_std_{temp}"
        stds_by_temp[key] = std_val
    return stds_by_temp

def load_eq_std(df, metric):
    row = df[df["metric"] == metric]
    if not row.empty:
        return row.iloc[0]["eq_std"]
    return np.nan

def load_eq_mu(df, metric):
    row = df[df["metric"] == metric]
    if not row.empty:
        return row.iloc[0]["eq_mu"]
    return np.nan

def std_from_xvg(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith(("#", "@")):
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    data.append(float(parts[1]))
                except ValueError:
                    continue
    return np.std(data) if data else np.nan

def load_lambda_block_stats(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    return {
        "lambda_mean": df["lamda"].mean() if "lamda" in df.columns else np.nan,
        "r_mean": df["r"].mean() if "r" in df.columns else np.nan,
        "300_mean": df["300"].mean() if "300" in df.columns else np.nan,
    }

def load_s2_mean(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    return df["300"].mean() if "300" in df.columns else np.nan

# Set the working directory for one mAb (e.g. AAD16457_clean)
base_path = "/home2/AbMelt/project/AbMelt/AAD16457_clean"  # ← Update as needed

# Load CSV parameters
params_path = os.path.join(base_path, "_abmelt_eq_20ns_parameters.csv")
params_df = pd.read_csv(params_path)

# Load all lambda stats files
lambda_stats_by_b = {}
for b in [2.5, 5, 10, 12.5]:
    fname = f"order_lambda_{b}block_20start.csv"
    path = os.path.join(base_path, fname)
    lambda_stats_by_b[b] = load_lambda_block_stats(path)

# Load 300K s2 order
s2_300_mean = load_s2_mean(os.path.join(base_path, "order_s2_300K_10block_20start.csv"))

'''
# Load sasa std from xvg
sasa_cdrl2_std_400 = std_from_xvg(os.path.join(base_path, "sasa_cdrl2_400.xvg"))

# Load rmsd_std_300 from rmsd=300.xvg
rmsd_std_300 = std_from_xvg(os.path.join(base_path, "rmsd_300.xvg"))
'''

# --- Load all RMSD stds (e.g., rmsd_300.xvg, rmsd_350.xvg, etc.) ---
rmsd_stds = load_xvg_metrics_by_temp(base_path, "rmsd")

# --- Load all SASA stds for cdrl2 (e.g., sasa_cdrl2_300.xvg, etc.) ---
sasa_cdrl2_stds = load_xvg_metrics_by_temp(base_path, "sasa_cdrl2")

# --- Compose initial output row with fixed features ---
# --- Initial fixed features ---
row = {
    "name": os.path.basename(base_path),
    "300-s2_b=2.5_eq=20": s2_300_mean,
    "300-std_b=2.5_eq=20": load_eq_mu(params_df, "total_std_300"),
    "r-lamda_b=2.5_eq=20": lambda_stats_by_b[2.5]["r_mean"],
    "r-lamda_b=5_eq=20": lambda_stats_by_b[5]["r_mean"],
    "r-lamda_b=12.5_eq=20": lambda_stats_by_b[12.5]["r_mean"],
    "tm": lambda_stats_by_b[10]["300_mean"],
}

# --- .xvg stds for each temp ---
xvg_std_prefixes = ["rmsd", "sasa_cdrl1", "sasa_cdrl2"]
row.update(load_all_xvg_std_features(base_path, xvg_std_prefixes, TEMPS))

# --- EQ stds from CSV for each temp ---
eq_std_metrics = [
    "gyr_cdrs_350_Rg",
    "bonds_350_contacts",
    "rmsf_cdrl1_350",
    "sasa_cdrl1_350"
]
row.update(load_all_eq_std_features(params_df, eq_std_metrics))

# Output
df_out = pd.DataFrame([row])
print(df_out.to_csv(index=False))

df_out.to_csv(os.path.join(base_path, "holdout_try.csv"), index=False)
