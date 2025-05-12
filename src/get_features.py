import os
import pandas as pd
import numpy as np
from glob import glob

root_dir = '/home2/AbMelt/project/AbMelt/AAD16457_clean'
out_file = '/home2/AbMelt/project/AbMelt/holdout_test.csv'

# Load main data (the one with mAb, TEMP, metric, eq_time, eq_mu, eq_std)
all_data = pd.read_csv(os.path.join(root_dir, '_abmelt_eq_20ns_parameters.csv'))

def extract_features(name_dir):
    features = {}
    features['name'] = name_dir

    df = all_data[all_data['mAb'] == name_dir]

    # Example: extract 'rmsd_std_300'
    try:
        features['rmsd_std_300'] = df[df['metric'] == 'rmsd_300']['eq_std'].values[0]
    except IndexError:
        features['rmsd_std_300'] = np.nan

    # Repeat for other metrics like 'gyr_cdrs_350_Rg', etc.
    # e.g., features['gyr_cdrs_Rg_std_350'] = ...

    # Read supplemental .csv for order_lambda
    try:
        lam_df = pd.read_csv(os.path.join(root_dir, name_dir, 'order_lambda_10block_20start.csv'))
        features['r-lamda_b=2.5_eq=20'] = lam_df.iloc[0, 1]  # or whatever logic you need
    except Exception:
        features['r-lamda_b=2.5_eq=20'] = np.nan

    # Repeat similarly for other custom files...

    return features

# Process all folders
all_features = []
for entry in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, entry)) and entry.endswith('_clean'):
        all_features.append(extract_features(entry))

# Save as CSV
df_out = pd.DataFrame(all_features)
df_out.to_csv(out_file, index=False)