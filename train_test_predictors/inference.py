import pandas as pd
import joblib

model = joblib.load('models/tm/efs_best_randomforest.pkl')
input_data = pd.read_csv('holdout_tm.csv')

# Use the same features used during training
feature_cols = ['gyr_cdrs_Rg_std_350', 'bonds_contacts_std_350', 'rmsf_cdrl1_std_350']
X = input_data[feature_cols]

predictions = model.predict(X)
input_data['predicted_tm'] = predictions

input_data.to_csv('predicted_output.csv', index=False)
print("✅ Predictions saved to predicted_output.csv")