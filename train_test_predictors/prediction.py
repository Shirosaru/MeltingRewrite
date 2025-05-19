#!/usr/bin/env python3

import glob
import os
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import logging
import argparse

# Constants
MODEL_TYPES = ['rfs', 'efs', 'afs']
OUTPUT_DIR = 'plots'
LOG_DIR = 'log'
FONT_SETTINGS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif']  # fallback included
}

# Ensure output directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(f'{LOG_DIR}/prediction.log', 'a'))
print = logger.info

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-holdout', type=str, default='holdout.csv', help='Holdout dataset file path')
parser.add_argument('-models', type=str, default='all', choices=MODEL_TYPES + ['all'], help='Models to predict on holdout set')
parser.add_argument('-rescaled', type=int, default=0, choices=[0, 1], help='Whether holdout data is rescaled relative to train data (0 or 1)')
args = parser.parse_args()

# Load datasets
holdout = pd.read_csv(args.holdout)
model_files = glob.glob("./models/*/*pkl")

# Load feature sets
feature_sets = {
    'rfs': pd.read_csv("rf_rfs.csv"),
    'efs': pd.read_csv("rf_efs.csv"),
    'afs': pd.read_csv("tagg_final.csv")
}

def evaluate_model(model, X, y, rescaled):
    """Evaluate the model using R², Pearson, and Spearman correlations."""
    predicted = model.predict(X)
    lsq = LinearRegression(fit_intercept=True)
    y = y.reshape(-1, 1)
    lsq.fit(y, predicted)
    r3 = lsq.score(y, predicted)
    predicted_fit = lsq.predict(y)
    y = np.squeeze(y)
    predicted = np.squeeze(predicted)
    r2 = r2_score(y, predicted)
    pearson = stats.pearsonr(y, predicted).statistic
    spearman = stats.spearmanr(y, predicted).correlation

    if rescaled:
        print(f'{model} (LR R²: {r3:.2f})')
    else:
        print(f'{model} (R²: {r2:.2f})')
    print(f'{model} (Rp: {pearson:.2f})')
    print(f'{model} (Rs: {spearman:.2f})')

    return predicted, predicted_fit, r3, r2, pearson, spearman

def plot_results(y, predicted, predicted_fit, model_name, rescaled, r3, r2, pearson, spearman):
    """Generate and save scatter plot with regression line."""
    fig, ax = plt.subplots()
    plt.rcParams.update(FONT_SETTINGS)
    ax.scatter(y, predicted, c='black', edgecolors=(0, 0, 0))
    if rescaled:
        ax.plot(y, predicted_fit, 'b--', lw=4)
    else:
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title(model_name)

    text_box = AnchoredText(f'LR R²: {r3:.2f}' if rescaled else f'R²: {r2:.2f}', frameon=True, prop=dict(color='blue'), loc='lower center', pad=0.5)
    text_box2 = AnchoredText(f'Rp: {pearson:.2f}\nRs: {spearman:.2f}', frameon=True, prop=dict(color='black'), loc=4, pad=0.5)
    plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
    plt.gca().add_artist(text_box2)
    plt.setp(text_box.patch, facecolor='white', alpha=0.5)
    plt.gca().add_artist(text_box)

    output_path = f'{OUTPUT_DIR}/{model_name}.png'
    print(f"[DEBUG] Saving plot to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

def process_model(model_file, model_type, rescaled):
    """Process and evaluate a single model."""
    model = joblib.load(model_file)
    print(f"[DEBUG] Model expects {model.n_features_in_} features")
    model_name = os.path.basename(model_file).split('.')[0]
    feature_set = feature_sets.get(model_type)
    if feature_set is None:
        print(f"Skipping {model_name}: No feature set found for model type '{model_type}'")
        return

    cols = feature_set.columns.tolist()[1:]  # Skip 'Unnamed: 0'
    feature_cols = cols[:-1]  # Take only feature columns
    target_col = cols[-1]     # The last column is 'tm'

    print(f"[DEBUG] Using features: {feature_cols}")
    print(f"[DEBUG] Using target: {target_col}")

    X = holdout[feature_cols].values
    y = holdout[target_col].values

    # Dynamically match expected number of features
    if model.n_features_in_ != X.shape[1]:
        print(f"[WARNING] Feature mismatch: Model expects {model.n_features_in_} features, but {X.shape[1]} provided.")
        # Try including 'tm' if missing
        if target_col not in feature_cols:
            print("[DEBUG] Adding target column back into features (not ideal).")
            X = holdout[feature_cols + [target_col]].values
        else:
            print("[ERROR] Cannot recover from feature mismatch.")
            return

    predicted, predicted_fit, r3, r2, pearson, spearman = evaluate_model(model, X, y, rescaled)
    plot_results(y, predicted, predicted_fit, model_name, rescaled, r3, r2, pearson, spearman)

def main():
    """Main function to process models based on command-line arguments."""
    if args.models == 'all':
        selected_models = model_files
    else:
        selected_models = [mf for mf in model_files if args.models in mf]

    for model_file in selected_models:
        for model_type in MODEL_TYPES:
            if model_type in model_file:
                process_model(model_file, model_type, args.rescaled)

if __name__ == "__main__":
    main()
