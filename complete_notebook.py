#!/usr/bin/env python3
"""
Complete the optimized notebook with final comparison section
"""

import json

# Load existing notebook
with open('/home/user/Smart_Electric_Grid/notebook_part4.json', 'r') as f:
    notebook = json.load(f)

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

# =============================================================================
# FINAL COMPARISON SECTION
# =============================================================================
print("Adding Final Comparison section...")

add_markdown_cell("## **Final Model Comparison**")

comparison_code = """# =============================================================================
# FINAL MODEL COMPARISON - SIDE-BY-SIDE ANALYSIS
# =============================================================================

print("="*70)
print("FINAL MODEL COMPARISON")
print("="*70)

# =============================================================================
# LOAD ALL PREDICTIONS
# =============================================================================

# Load test data
y_test_actual = np.load('y_test.npy')

# Load predictions from all models
y_pred_arima = np.load('arima_predictions.npy')
y_pred_mlp = np.load('mlp_predictions.npy')
y_pred_anfis = np.load('anfis_predictions.npy')

print(f"\\n✓ Loaded predictions from all 3 models")
print(f"  Test samples: {len(y_test_actual)}")

# =============================================================================
# CALCULATE METRICS FOR ALL MODELS
# =============================================================================

print(f"\\n{'='*70}")
print("COMPREHENSIVE METRICS COMPARISON")
print(f"{'='*70}")

# ARIMA
arima_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_arima))
arima_mae = mean_absolute_error(y_test_actual, y_pred_arima)
mask_arima = y_test_actual != 0
arima_mape = np.mean(np.abs((y_test_actual[mask_arima] - y_pred_arima[mask_arima]) / y_test_actual[mask_arima])) * 100
arima_r2 = r2_score(y_test_actual, y_pred_arima)

# MLP
mlp_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_mlp))
mlp_mae = mean_absolute_error(y_test_actual, y_pred_mlp)
mask_mlp = y_test_actual != 0
mlp_mape = np.mean(np.abs((y_test_actual[mask_mlp] - y_pred_mlp[mask_mlp]) / y_test_actual[mask_mlp])) * 100
mlp_r2 = r2_score(y_test_actual, y_pred_mlp)

# ANFIS
anfis_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_anfis))
anfis_mae = mean_absolute_error(y_test_actual, y_pred_anfis)
mask_anfis = y_test_actual != 0
anfis_mape = np.mean(np.abs((y_test_actual[mask_anfis] - y_pred_anfis[mask_anfis]) / y_test_actual[mask_anfis])) * 100
anfis_r2 = r2_score(y_test_actual, y_pred_anfis)

# =============================================================================
# CREATE COMPARISON TABLE
# =============================================================================

print(f"\\n{'='*70}")
print("MODEL COMPARISON RESULTS (Test Set - Hourly Forecasting)")
print(f"{'='*70}")
print(f"{'Model':<20} | {'RMSE (kW)':<12} | {'MAE (kW)':<12} | {'MAPE (%)':<12} | {'R²':<8}")
print(f"{'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
print(f"{'ARIMA':<20} | {arima_rmse:>10.2f}   | {arima_mae:>10.2f}   | {arima_mape:>10.2f}   | {arima_r2:>6.4f}")
print(f"{'MLP (Optimized)':<20} | {mlp_rmse:>10.2f}   | {mlp_mae:>10.2f}   | {mlp_mape:>10.2f}   | {mlp_r2:>6.4f}")
print(f"{'ANFIS (Enhanced)':<20} | {anfis_rmse:>10.2f}   | {anfis_mae:>10.2f}   | {anfis_mape:>10.2f}   | {anfis_r2:>6.4f}")
print(f"{'='*70}")

# Determine best model
best_mape = min(arima_mape, mlp_mape, anfis_mape)
if anfis_mape == best_mape:
    best_model = "ANFIS (Enhanced)"
elif mlp_mape == best_mape:
    best_model = "MLP (Optimized)"
else:
    best_model = "ARIMA"

print(f"\\n🏆 BEST MODEL: {best_model} (MAPE: {best_mape:.2f}%)")

if anfis_mape < 3.0:
    print(f"\\n🎯 TARGET ACHIEVED: ANFIS MAPE < 3%!")
    print(f"   Accuracy: {100 - anfis_mape:.2f}%")

# =============================================================================
# IMPROVEMENT ANALYSIS
# =============================================================================

print(f"\\n{'='*70}")
print("IMPROVEMENT ANALYSIS (vs ARIMA Baseline)")
print(f"{'='*70}")

mlp_improvement = ((arima_mape - mlp_mape) / arima_mape) * 100
anfis_improvement = ((arima_mape - anfis_mape) / arima_mape) * 100

print(f"MLP improvement:   {mlp_improvement:>6.1f}% better than ARIMA")
print(f"ANFIS improvement: {anfis_improvement:>6.1f}% better than ARIMA")

if anfis_mape < mlp_mape:
    anfis_vs_mlp = ((mlp_mape - anfis_mape) / mlp_mape) * 100
    print(f"ANFIS improvement: {anfis_vs_mlp:>6.1f}% better than MLP")

# =============================================================================
# COMPREHENSIVE VISUALIZATION
# =============================================================================

print(f"\\n{'='*70}")
print("Generating Comparison Visualizations")
print(f"{'='*70}")

fig = plt.figure(figsize=(18, 12))

# 1. Metrics Bar Chart
ax1 = plt.subplot(2, 3, 1)
models = ['ARIMA', 'MLP\\n(Optimized)', 'ANFIS\\n(Enhanced)']
mapes = [arima_mape, mlp_mape, anfis_mape]
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax1.bar(models, mapes, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('MAPE (%)', fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontweight='bold', fontsize=12)
ax1.axhline(y=3.0, color='red', linestyle='--', linewidth=2, label='Target: 3%')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar, mape in zip(bars, mapes):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{mape:.2f}%', ha='center', va='bottom', fontweight='bold')

# 2. RMSE Comparison
ax2 = plt.subplot(2, 3, 2)
rmses = [arima_rmse, mlp_rmse, anfis_rmse]
bars = ax2.bar(models, rmses, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('RMSE (kW)', fontweight='bold')
ax2.set_title('RMSE Comparison', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3)

for bar, rmse in zip(bars, rmses):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{rmse:.1f}', ha='center', va='bottom', fontweight='bold')

# 3. Time Series Comparison (first 200 points)
ax3 = plt.subplot(2, 3, 3)
n_plot = min(200, len(y_test_actual))
x_axis = range(n_plot)
ax3.plot(x_axis, y_test_actual[:n_plot], label='Actual', linewidth=2, color='black', alpha=0.7)
ax3.plot(x_axis, y_pred_arima[:n_plot], label='ARIMA', linewidth=1.5, alpha=0.7)
ax3.plot(x_axis, y_pred_mlp[:n_plot], label='MLP', linewidth=1.5, alpha=0.7)
ax3.plot(x_axis, y_pred_anfis[:n_plot], label='ANFIS', linewidth=1.5, alpha=0.7)
ax3.set_xlabel('Time Index')
ax3.set_ylabel('Load (kW)')
ax3.set_title('Predictions Comparison (First 200 Hours)', fontweight='bold', fontsize=12)
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

# 4. Scatter Plot - ARIMA
ax4 = plt.subplot(2, 3, 4)
ax4.scatter(y_test_actual, y_pred_arima, alpha=0.4, s=10, color=colors[0])
ax4.plot([y_test_actual.min(), y_test_actual.max()],
         [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)
ax4.set_xlabel('Actual Load (kW)')
ax4.set_ylabel('Predicted Load (kW)')
ax4.set_title(f'ARIMA (MAPE: {arima_mape:.2f}%)', fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. Scatter Plot - MLP
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(y_test_actual, y_pred_mlp, alpha=0.4, s=10, color=colors[1])
ax5.plot([y_test_actual.min(), y_test_actual.max()],
         [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)
ax5.set_xlabel('Actual Load (kW)')
ax5.set_ylabel('Predicted Load (kW)')
ax5.set_title(f'MLP Optimized (MAPE: {mlp_mape:.2f}%)', fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. Scatter Plot - ANFIS
ax6 = plt.subplot(2, 3, 6)
ax6.scatter(y_test_actual, y_pred_anfis, alpha=0.4, s=10, color=colors[2])
ax6.plot([y_test_actual.min(), y_test_actual.max()],
         [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)
ax6.set_xlabel('Actual Load (kW)')
ax6.set_ylabel('Predicted Load (kW)')
ax6.set_title(f'ANFIS Enhanced (MAPE: {anfis_mape:.2f}%)', fontweight='bold')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('final_model_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: final_model_comparison.png")
plt.show()

# =============================================================================
# SAVE COMPARISON RESULTS
# =============================================================================

comparison_results = {
    'ARIMA': {'RMSE': arima_rmse, 'MAE': arima_mae, 'MAPE': arima_mape, 'R2': arima_r2},
    'MLP': {'RMSE': mlp_rmse, 'MAE': mlp_mae, 'MAPE': mlp_mape, 'R2': mlp_r2},
    'ANFIS': {'RMSE': anfis_rmse, 'MAE': anfis_mae, 'MAPE': anfis_mape, 'R2': anfis_r2}
}

comparison_df = pd.DataFrame(comparison_results).T
comparison_df.to_csv('final_comparison_metrics.csv')
print("\\n✓ Saved: final_comparison_metrics.csv")

print(f"\\n{'='*70}")
print("OPTIMIZATION COMPLETE!")
print(f"{'='*70}")
print(f"\\n📊 Summary:")
print(f"   • ARIMA:  {arima_mape:.2f}% MAPE (Baseline)")
print(f"   • MLP:    {mlp_mape:.2f}% MAPE (Optimized with deeper architecture)")
print(f"   • ANFIS:  {anfis_mape:.2f}% MAPE (Enhanced with PCA features)")
print(f"\\n🏆 Best Model: {best_model}")
print(f"\\n✓ All models trained, evaluated, and compared successfully!")
"""

add_code_cell(comparison_code)

print("✓ Final comparison section added")

# =============================================================================
# SAVE COMPLETE NOTEBOOK
# =============================================================================

with open('/home/user/Smart_Electric_Grid/CAPSTONE_SmartElectricGrid_OPTIMIZED.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"\n{'='*70}")
print("NOTEBOOK OPTIMIZATION COMPLETE!")
print(f"{'='*70}")
print(f"✓ Total cells: {len(notebook['cells'])}")
print(f"✓ Saved as: CAPSTONE_SmartElectricGrid_OPTIMIZED.ipynb")
print(f"\nStructure:")
print(f"  1. Global Imports & Configuration")
print(f"  2. Utility Functions (unified metrics & plotting)")
print(f"  3. Data Loading")
print(f"  4. Data Cleaning")
print(f"  5. Feature Engineering (PCA, cyclical encoding)")
print(f"  6. ARIMA Model (separate)")
print(f"  7. MLP Model (separate, optimized)")
print(f"  8. ANFIS Model (separate, enhanced)")
print(f"  9. Final Comparison (new)")
print(f"\nKey Optimizations:")
print(f"  ✓ PCA: 26 → 12 features")
print(f"  ✓ ANFIS rules: 2.3M → 4,096 (99.8% reduction!)")
print(f"  ✓ Code reduced by ~25-30%")
print(f"  ✓ Unified metrics function (no duplication)")
print(f"  ✓ Enhanced MLP with deeper architectures")
print(f"  ✓ Target: ANFIS <3% MAPE")
