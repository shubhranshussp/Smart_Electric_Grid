#!/usr/bin/env python3
"""
Add ARIMA, MLP (Optimized), and ANFIS (Enhanced) model sections to notebook
Each model is COMPLETELY SEPARATE with its own cells
"""

import json

# Load existing notebook
with open('/home/user/Smart_Electric_Grid/notebook_part2.json', 'r') as f:
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
# ARIMA MODEL SECTION (SEPARATE)
# =============================================================================
print("Adding ARIMA Model section...")

add_markdown_cell("## **ARIMA Model**")

arima_code = """# =============================================================================
# ARIMA MODEL - TIME SERIES FORECASTING
# =============================================================================

print("="*70)
print("ARIMA MODEL - AUTOREGRESSIVE INTEGRATED MOVING AVERAGE")
print("="*70)

# Load target data (use non-PCA data for ARIMA - it's univariate time series)
y_train_arima = np.load('y_train.npy')
y_val_arima = np.load('y_val.npy')
y_test_arima = np.load('y_test.npy')

# Create time series
train_val_series = np.concatenate([y_train_arima, y_val_arima])

print(f"\\n✓ Data loaded:")
print(f"  Train: {len(y_train_arima)} samples")
print(f"  Val:   {len(y_val_arima)} samples")
print(f"  Test:  {len(y_test_arima)} samples")

# -----------------------------------------------
# STEP 1: Stationarity Check
# -----------------------------------------------
print(f"\\n{'='*70}")
print("STEP 1: Stationarity Check (ADF Test)")
print(f"{'='*70}")

result = adfuller(y_train_arima)
print(f"\\nADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
if result[1] <= 0.05:
    print("✓ Series is STATIONARY (p-value <= 0.05)")
else:
    print("✗ Series is NON-STATIONARY (p-value > 0.05)")

# -----------------------------------------------
# STEP 2: ACF & PACF Analysis
# -----------------------------------------------
print(f"\\n{'='*70}")
print("STEP 2: ACF & PACF Analysis")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(y_train_arima, lags=48, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
plot_pacf(y_train_arima, lags=48, ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('arima_acf_pacf.png', dpi=150, bbox_inches='tight')
print("✓ Saved: arima_acf_pacf.png")
plt.show()

# -----------------------------------------------
# STEP 3: Model Selection (Quick)
# -----------------------------------------------
print(f"\\n{'='*70}")
print("STEP 3: ARIMA Model Selection")
print(f"{'='*70}")

configs = [(1,0,1), (2,0,1), (1,0,2), (2,0,2)]
best_aic = np.inf
best_order = None

print("\\nTesting ARIMA configurations...")
for order in configs:
    try:
        model = ARIMA(y_train_arima, order=order)
        fitted = model.fit()
        print(f"  ARIMA{order}: AIC={fitted.aic:.2f}")
        if fitted.aic < best_aic:
            best_aic = fitted.aic
            best_order = order
    except:
        print(f"  ARIMA{order}: Failed")

print(f"\\n✓ Best order: ARIMA{best_order} (AIC={best_aic:.2f})")

# -----------------------------------------------
# STEP 4: Train Final Model
# -----------------------------------------------
print(f"\\n{'='*70}")
print("STEP 4: Training Final ARIMA Model")
print(f"{'='*70}")

print(f"\\nTraining ARIMA{best_order} on train+val data...")
final_arima = ARIMA(train_val_series, order=best_order)
fitted_arima = final_arima.fit()
print("✓ Model trained successfully")

# -----------------------------------------------
# STEP 5: Predictions
# -----------------------------------------------
print(f"\\n{'='*70}")
print("STEP 5: Generating Predictions")
print(f"{'='*70}")

# Forecast test samples
forecast_result = fitted_arima.forecast(steps=len(y_test_arima))
y_pred_arima = np.array(forecast_result)

print(f"✓ Generated {len(y_pred_arima)} predictions")

# -----------------------------------------------
# STEP 6: Evaluation
# -----------------------------------------------
arima_metrics = calculate_metrics(y_test_arima, y_pred_arima, "ARIMA")

# Save results
np.save('arima_predictions.npy', y_pred_arima)
with open('arima_model.pkl', 'wb') as f:
    pickle.dump(fitted_arima, f)

# Visualizations
plot_predictions(y_test_arima, y_pred_arima, "ARIMA", "arima_predictions.png")

print(f"\\n{'='*70}")
print("ARIMA MODEL COMPLETE")
print(f"{'='*70}")
print(f"✓ MAPE: {arima_metrics['MAPE']:.2f}%")
print(f"✓ Model saved: arima_model.pkl")
"""

add_code_cell(arima_code)

print("✓ ARIMA section added (separate)")

# =============================================================================
# MLP MODEL SECTION (SEPARATE - OPTIMIZED)
# =============================================================================
print("Adding MLP Model section (OPTIMIZED)...")

add_markdown_cell("## **MLP Model (Optimized)**")

mlp_code = """# =============================================================================
# MLP MODEL - MULTI-LAYER PERCEPTRON (OPTIMIZED WITH MODERN TECHNIQUES)
# =============================================================================

print("="*70)
print("MLP MODEL - OPTIMIZED NEURAL NETWORK")
print("="*70)

# Load PCA-transformed data
X_train_mlp = np.load('X_train_pca.npy')
X_val_mlp = np.load('X_val_pca.npy')
X_test_mlp = np.load('X_test_pca.npy')

y_train_mlp = np.load('y_train_scaled.npy')
y_val_mlp = np.load('y_val_scaled.npy')
y_test_mlp = np.load('y_test_scaled.npy')

print(f"\\n✓ Data loaded (PCA features):")
print(f"  Train: {X_train_mlp.shape}")
print(f"  Val:   {X_val_mlp.shape}")
print(f"  Test:  {X_test_mlp.shape}")

# -----------------------------------------------
# STEP 1: Test Multiple Architectures
# -----------------------------------------------
print(f"\\n{'='*70}")
print("STEP 1: Testing Multiple MLP Architectures")
print(f"{'='*70}")

# Define configurations to test (both 2-layer and 3-layer)
mlp_configs = [
    {
        'name': '2-Layer (128, 64)',
        'hidden_layer_sizes': (128, 64),
        'alpha': 0.0001,
        'learning_rate_init': 0.001
    },
    {
        'name': '2-Layer (256, 128) - Wider',
        'hidden_layer_sizes': (256, 128),
        'alpha': 0.0001,
        'learning_rate_init': 0.001
    },
    {
        'name': '3-Layer (256, 128, 64) - Deep',
        'hidden_layer_sizes': (256, 128, 64),
        'alpha': 0.0001,
        'learning_rate_init': 0.001
    },
    {
        'name': '3-Layer (128, 64, 32)',
        'hidden_layer_sizes': (128, 64, 32),
        'alpha': 0.0001,
        'learning_rate_init': 0.0005
    },
    {
        'name': '2-Layer (128, 64) - High Reg',
        'hidden_layer_sizes': (128, 64),
        'alpha': 0.001,
        'learning_rate_init': 0.001
    }
]

print(f"\\nTesting {len(mlp_configs)} configurations...\\n")

best_val_rmse = np.inf
best_config = None
best_model = None

for i, config in enumerate(mlp_configs, 1):
    print(f"[{i}/{len(mlp_configs)}] Testing: {config['name']}")

    # Create model with configuration
    mlp = MLPRegressor(
        hidden_layer_sizes=config['hidden_layer_sizes'],
        activation='relu',
        solver='adam',
        alpha=config['alpha'],
        batch_size=32,
        learning_rate_init=config['learning_rate_init'],
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=RANDOM_SEED,
        verbose=False
    )

    # Train
    start_time = time.time()
    mlp.fit(X_train_mlp, y_train_mlp)
    train_time = time.time() - start_time

    # Validate
    y_val_pred = mlp.predict(X_val_mlp)
    val_rmse = np.sqrt(mean_squared_error(y_val_mlp, y_val_pred))

    print(f"  Val RMSE: {val_rmse:.4f} | Training time: {train_time:.2f}s | Epochs: {mlp.n_iter_}")

    # Track best
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_config = config
        best_model = mlp
        print(f"  ✓ NEW BEST!")
    print()

print(f"{'='*70}")
print(f"BEST CONFIGURATION: {best_config['name']}")
print(f"Validation RMSE: {best_val_rmse:.4f}")
print(f"{'='*70}")

# -----------------------------------------------
# STEP 2: Retrain Best Model on Train+Val
# -----------------------------------------------
print(f"\\n{'='*70}")
print("STEP 2: Retraining Best Model on Train+Val")
print(f"{'='*70}")

# Combine train and val
X_train_val = np.concatenate([X_train_mlp, X_val_mlp])
y_train_val = np.concatenate([y_train_mlp, y_val_mlp])

print(f"\\nTraining {best_config['name']} on {len(X_train_val)} samples...")

final_mlp = MLPRegressor(
    hidden_layer_sizes=best_config['hidden_layer_sizes'],
    activation='relu',
    solver='adam',
    alpha=best_config['alpha'],
    batch_size=32,
    learning_rate_init=best_config['learning_rate_init'],
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=30,
    random_state=RANDOM_SEED,
    verbose=False
)

final_mlp.fit(X_train_val, y_train_val)
print(f"✓ Final model trained ({final_mlp.n_iter_} epochs)")

# -----------------------------------------------
# STEP 3: Predictions on Test Set
# -----------------------------------------------
print(f"\\n{'='*70}")
print("STEP 3: Generating Test Predictions")
print(f"{'='*70}")

y_test_pred_scaled = final_mlp.predict(X_test_mlp)

# Inverse transform predictions
with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

y_test_mlp_actual = scaler_y.inverse_transform(y_test_mlp.reshape(-1, 1)).ravel()
y_test_mlp_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()

print(f"✓ Generated {len(y_test_mlp_pred)} predictions")

# -----------------------------------------------
# STEP 4: Evaluation
# -----------------------------------------------
mlp_metrics = calculate_metrics(y_test_mlp_actual, y_test_mlp_pred, "MLP (Optimized)")

# Save results
np.save('mlp_predictions.npy', y_test_mlp_pred)
with open('mlp_model_optimized.pkl', 'wb') as f:
    pickle.dump(final_mlp, f)

# Visualizations
plot_predictions(y_test_mlp_actual, y_test_mlp_pred, "MLP (Optimized)", "mlp_predictions.png")

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(final_mlp.loss_curve_, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MLP Training Loss Curve')
plt.grid(True)
plt.savefig('mlp_training_curve.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\\n{'='*70}")
print("MLP MODEL COMPLETE")
print(f"{'='*70}")
print(f"✓ Best Architecture: {best_config['name']}")
print(f"✓ MAPE: {mlp_metrics['MAPE']:.2f}%")
print(f"✓ Model saved: mlp_model_optimized.pkl")
"""

add_code_cell(mlp_code)

print("✓ MLP section added (separate, optimized)")

# Save notebook so far
with open('/home/user/Smart_Electric_Grid/notebook_part3.json', 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"\n✓ Part 3 saved: {len(notebook['cells'])} total cells")
print("✓ ARIMA and MLP sections complete")
print("✓ Next: Adding ANFIS section...")
