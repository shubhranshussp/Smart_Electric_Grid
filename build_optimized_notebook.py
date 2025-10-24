#!/usr/bin/env python3
"""
Script to build optimized Smart Electric Grid notebook
Implements all optimizations: PCA, improved MLP, enhanced ANFIS
"""

import json

def create_optimized_notebook():
    """Create the complete optimized notebook structure"""

    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    # Helper function to create cells
    def add_markdown_cell(text):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [text + "\n"]
        })

    def add_code_cell(code):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [code + "\n"]
        })

    # =========================================================================
    # CELL 0: Global Imports and Configuration
    # =========================================================================
    add_markdown_cell("# **Smart Electric Grid - Energy Forecasting (OPTIMIZED)**\n\n"
                      "**Models:** ARIMA, MLP (Optimized), ANFIS (Enhanced)\n\n"
                      "**Key Optimizations:**\n"
                      "- PCA feature reduction (26 → 12-15 features)\n"
                      "- Improved MLP with deeper architecture\n"
                      "- Enhanced ANFIS with <3% MAPE target\n"
                      "- Unified metrics and comparison\n"
                      "- Cyclical encoding for temporal features")

    add_markdown_cell("## **Global Imports and Configuration**")

    add_code_cell("""# =============================================================================
# GLOBAL IMPORTS - All libraries in one place (eliminates repetition)
# =============================================================================

# Core libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')

# Machine Learning - Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Machine Learning - Models
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Statistical models
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Utilities
import pickle
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import itertools
import time

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# PCA configuration
N_PCA_COMPONENTS = 12  # Reduced from 26 features

# Model configurations
MLP_CONFIG = {
    'max_iter': 1000,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 30,
    'random_state': RANDOM_SEED
}

ANFIS_CONFIG = {
    'n_membership_functions': 2,  # Reduced for efficiency
    'max_epochs': 300,  # Increased for better training
    'learning_rate': 0.01,
    'patience': 50
}

print("✓ All imports loaded successfully")
print(f"✓ Random seed set to: {RANDOM_SEED}")
print(f"✓ PCA components: {N_PCA_COMPONENTS}")""")

    # =========================================================================
    # CELL 1: Utility Functions
    # =========================================================================
    add_markdown_cell("## **Utility Functions (Unified Metrics & Plotting)**")

    add_code_cell("""# =============================================================================
# UNIFIED METRIC CALCULATION FUNCTION (eliminates duplication across models)
# =============================================================================

def calculate_metrics(y_true, y_pred, model_name="Model"):
    \"\"\"
    Calculate comprehensive metrics for model evaluation

    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model for display

    Returns:
    --------
    dict : Dictionary containing all metrics
    \"\"\"
    # Ensure arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE with safe division
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    mape = np.clip(mape, 0, 100)  # Clip to reasonable range

    # R-squared
    r2 = r2_score(y_true, y_pred)

    # Print results
    print(f"\\n{'='*60}")
    print(f"{model_name} - Performance Metrics")
    print(f"{'='*60}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f} kW")
    print(f"MAE  (Mean Absolute Error):     {mae:.2f} kW")
    print(f"MAPE (Mean Absolute % Error):   {mape:.2f} %")
    print(f"R²   (R-squared Score):         {r2:.4f}")
    print(f"{'='*60}")

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }

# =============================================================================
# UNIFIED PLOTTING FUNCTION
# =============================================================================

def plot_predictions(y_true, y_pred, model_name="Model", save_path=None):
    \"\"\"Create comprehensive prediction visualization\"\"\"

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Actual vs Predicted (Scatter)
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
    axes[0, 0].plot([y_true.min(), y_true.max()],
                     [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Load (kW)')
    axes[0, 0].set_ylabel('Predicted Load (kW)')
    axes[0, 0].set_title(f'{model_name}: Actual vs Predicted')
    axes[0, 0].grid(True)

    # 2. Time series comparison (first 200 points)
    n_points = min(200, len(y_true))
    axes[0, 1].plot(y_true[:n_points], label='Actual', linewidth=2)
    axes[0, 1].plot(y_pred[:n_points], label='Predicted', linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Time Index')
    axes[0, 1].set_ylabel('Load (kW)')
    axes[0, 1].set_title(f'{model_name}: Time Series Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. Residuals
    residuals = y_true - y_pred
    axes[1, 0].scatter(range(len(residuals)), residuals, alpha=0.5)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Residual (kW)')
    axes[1, 0].set_title(f'{model_name}: Residuals Plot')
    axes[1, 0].grid(True)

    # 4. Residual distribution
    axes[1, 1].hist(residuals, bins=50, edgecolor='black')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Residual (kW)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'{model_name}: Residual Distribution')
    axes[1, 1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

print("✓ Utility functions defined successfully")""")

    # =========================================================================
    # CELL 2: Load Dataset
    # =========================================================================
    add_markdown_cell("## **Load Dataset**")

    add_code_cell("""# Load dataset (remove Google Colab dependency)
try:
    from google.colab import files
    uploaded = files.upload()
    print("✓ File uploaded via Google Colab")
except:
    print("✓ Running in local environment - Dataset.csv should be present")

# Load data
df = pd.read_csv('Dataset.csv')
print(f"\\n✓ Dataset loaded successfully: {df.shape}")
print(f"  Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")""")

    return notebook

# Create and save the notebook
notebook = create_optimized_notebook()

with open('/home/user/Smart_Electric_Grid/notebook_part1.json', 'w') as f:
    json.dump(notebook, f, indent=2)

print("✓ Part 1 of optimized notebook created successfully")
print(f"✓ Created {len(notebook['cells'])} cells so far")
