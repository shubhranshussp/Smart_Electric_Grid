#!/usr/bin/env python3
"""Add Data Cleaning and Feature Engineering cells to optimized notebook"""

import json

# Load existing notebook
with open('/home/user/Smart_Electric_Grid/notebook_part1.json', 'r') as f:
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

# =========================================================================
# DATA CLEANING SECTION
# =========================================================================
add_markdown_cell("## **Data Cleaning**")

add_code_cell("""# =============================================================================
# DATA CLEANING - Streamlined and optimized
# =============================================================================

print("="*70)
print("DATA CLEANING PROCESS")
print("="*70)

# Step 1: Fix DATE column
df['DATE'] = df['DATE'].fillna(method='ffill')
print(f"✓ Step 1: Fixed DATE column")

# Step 2: Select relevant columns
columns_to_keep = [
    'DATE', 'TIME', 'VOLTAGE', 'CURRENT', 'PF',
    'POWER (KW)', '"WEEKEND/WEEKDAY"', 'SEASON',
    'Temp (F)', 'Humidity (%)'
]
df_clean = df[columns_to_keep].copy()
print(f"✓ Step 2: Kept {len(columns_to_keep)} relevant columns")

# Step 3: Rename columns
column_mapping = {
    'DATE': 'date',
    'TIME': 'time',
    'VOLTAGE': 'voltage',
    'CURRENT': 'current',
    'PF': 'power_factor',
    'POWER (KW)': 'load_kw',
    '"WEEKEND/WEEKDAY"': 'is_weekend',
    'SEASON': 'season',
    'Temp (F)': 'temperature_f',
    'Humidity (%)': 'humidity'
}
df_clean.rename(columns=column_mapping, inplace=True)
print(f"✓ Step 3: Renamed columns for consistency")

# Step 4: Create datetime index
df_clean['time'] = df_clean['time'].str.replace('-', ':')
df_clean['datetime'] = pd.to_datetime(
    df_clean['date'] + ' ' + df_clean['time'],
    format='%d/%m/%Y %H:%M',
    errors='coerce'
)
df_clean.set_index('datetime', inplace=True)
df_clean.drop(['date', 'time'], axis=1, inplace=True)
print(f"✓ Step 4: Created datetime index")

# Step 5: Handle missing values
df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
print(f"✓ Step 5: Handled missing values")

# Step 6: Convert is_weekend to binary
df_clean['is_weekend'] = df_clean['is_weekend'].apply(
    lambda x: 1 if str(x).strip().upper() == 'WEEKEND' else 0
)
print(f"✓ Step 6: Converted is_weekend to binary (0/1)")

print(f"\\n✓ CLEANING COMPLETE")
print(f"  Final shape: {df_clean.shape}")
print(f"  Date range: {df_clean.index.min()} to {df_clean.index.max()}")
print(f"  Total hours: {len(df_clean)}")

# Cache cleaned data in memory (eliminates repeated file loading)
CLEANED_DATA = df_clean.copy()

# Save for backup
df_clean.to_csv('cleaned_electric_load_data.csv')
print(f"\\n✓ Saved: cleaned_electric_load_data.csv")""")

# =========================================================================
# FEATURE ENGINEERING WITH PCA
# =========================================================================
add_markdown_cell("## **Feature Engineering (with PCA Optimization)**")

add_code_cell("""# =============================================================================
# FEATURE ENGINEERING - Enhanced with PCA, Cyclical Encoding, One-Hot Encoding
# =============================================================================

print("="*70)
print("FEATURE ENGINEERING PROCESS")
print("="*70)

# Start with cleaned data from memory (no file reload needed!)
df_features = CLEANED_DATA.copy()

# Extract datetime features
df_features['hour'] = df_features.index.hour
df_features['day_of_week'] = df_features.index.dayofweek
df_features['month'] = df_features.index.month

print(f"✓ Extracted basic datetime features")

# -----------------------------------------------
# LAG FEATURES
# -----------------------------------------------
df_features['load_lag_1h'] = df_features['load_kw'].shift(1)
df_features['load_lag_24h'] = df_features['load_kw'].shift(24)
df_features['load_lag_168h'] = df_features['load_kw'].shift(168)
print(f"✓ Created lag features (1h, 24h, 168h)")

# -----------------------------------------------
# ROLLING STATISTICS
# -----------------------------------------------
df_features['load_rolling_mean_3h'] = df_features['load_kw'].rolling(window=3, min_periods=1).mean()
df_features['load_rolling_mean_6h'] = df_features['load_kw'].rolling(window=6, min_periods=1).mean()
df_features['load_rolling_mean_24h'] = df_features['load_kw'].rolling(window=24, min_periods=1).mean()
df_features['load_rolling_std_24h'] = df_features['load_kw'].rolling(window=24, min_periods=1).std()
print(f"✓ Created rolling statistics features")

# -----------------------------------------------
# CYCLICAL ENCODING (NEW! - Better hour representation)
# -----------------------------------------------
df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
print(f"✓ Created cyclical encoding (sin/cos) for temporal features")

# -----------------------------------------------
# TEMPORAL INDICATORS
# -----------------------------------------------
df_features['is_peak_morning'] = ((df_features['hour'] >= 7) & (df_features['hour'] <= 9)).astype(int)
df_features['is_peak_evening'] = ((df_features['hour'] >= 18) & (df_features['hour'] <= 21)).astype(int)
df_features['is_peak_hour'] = (df_features['is_peak_morning'] | df_features['is_peak_evening']).astype(int)
df_features['is_night'] = ((df_features['hour'] >= 0) & (df_features['hour'] <= 5)).astype(int)
df_features['is_working_hours'] = ((df_features['hour'] >= 9) & (df_features['hour'] <= 17)).astype(int)
df_features['weekend_hour'] = df_features['is_weekend'] * df_features['hour']
print(f"✓ Created temporal indicator features")

# -----------------------------------------------
# WEATHER INTERACTIONS
# -----------------------------------------------
df_features['temp_humidity_index'] = df_features['temperature_f'] * (df_features['humidity'] / 100)
df_features['temp_deviation'] = df_features['temperature_f'] - df_features['temperature_f'].mean()
df_features['season_temp'] = df_features['season'] * df_features['temperature_f']
print(f"✓ Created weather interaction features")

# -----------------------------------------------
# ONE-HOT ENCODING FOR SEASON (NEW! - Better categorical handling)
# -----------------------------------------------
season_dummies = pd.get_dummies(df_features['season'], prefix='season')
df_features = pd.concat([df_features, season_dummies], axis=1)
print(f"✓ Created one-hot encoding for season")

# Remove rows with NaN (from lag features)
df_features = df_features.dropna()
print(f"\\n✓ Removed NaN rows: {len(df_features)} samples remaining")

# -----------------------------------------------
# PREPARE FEATURES AND TARGET
# -----------------------------------------------
# Target variable
y = df_features['load_kw'].values

# Feature columns (before PCA)
feature_cols = [col for col in df_features.columns if col != 'load_kw']
X_original = df_features[feature_cols].values

print(f"\\n✓ Original feature set: {X_original.shape[1]} features")
print(f"  Total samples: {X_original.shape[0]}")

# -----------------------------------------------
# TRAIN-VAL-TEST SPLIT (Temporal)
# -----------------------------------------------
n_total = len(X_original)
n_train = int(n_total * TRAIN_RATIO)
n_val = int(n_total * VAL_RATIO)

X_train_orig = X_original[:n_train]
X_val_orig = X_original[n_train:n_train+n_val]
X_test_orig = X_original[n_train+n_val:]

y_train = y[:n_train]
y_val = y[n_train:n_train+n_val]
y_test = y[n_train+n_val:]

print(f"\\n✓ Data split:")
print(f"  Train: {len(X_train_orig)} samples ({TRAIN_RATIO*100:.0f}%)")
print(f"  Val:   {len(X_val_orig)} samples ({VAL_RATIO*100:.0f}%)")
print(f"  Test:  {len(X_test_orig)} samples ({TEST_RATIO*100:.0f}%)")

# -----------------------------------------------
# FEATURE SCALING (fit on train, apply to all)
# -----------------------------------------------
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_orig)
X_val_scaled = scaler_X.transform(X_val_orig)
X_test_scaled = scaler_X.transform(X_test_orig)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

print(f"\\n✓ Feature scaling completed (StandardScaler)")

# -----------------------------------------------
# PCA DIMENSIONALITY REDUCTION (KEY OPTIMIZATION!)
# -----------------------------------------------
print(f"\\n{'='*70}")
print(f"PCA DIMENSIONALITY REDUCTION")
print(f"{'='*70}")

pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_SEED)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print(f"✓ PCA reduced features: {X_original.shape[1]} → {N_PCA_COMPONENTS}")
print(f"✓ Total variance explained: {cumulative_variance[-1]*100:.2f}%")
print(f"\\nVariance by component:")
for i, (var, cum_var) in enumerate(zip(explained_variance[:5], cumulative_variance[:5])):
    print(f"  PC{i+1}: {var*100:.2f}% (cumulative: {cum_var*100:.2f}%)")
print(f"  ...")

# -----------------------------------------------
# SAVE PROCESSED DATA
# -----------------------------------------------
# Save PCA-transformed data (for MLP and ANFIS)
np.save('X_train_pca.npy', X_train_pca)
np.save('X_val_pca.npy', X_val_pca)
np.save('X_test_pca.npy', X_test_pca)

np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('y_test.npy', y_test)

np.save('y_train_scaled.npy', y_train_scaled)
np.save('y_val_scaled.npy', y_val_scaled)
np.save('y_test_scaled.npy', y_test_scaled)

# Save original scaled features (for ARIMA if needed)
np.save('X_train_scaled.npy', X_train_scaled)
np.save('X_val_scaled.npy', X_val_scaled)
np.save('X_test_scaled.npy', X_test_scaled)

# Save scalers and PCA
with open('scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)
with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)

print(f"\\n✓ All processed data saved successfully")
print(f"\\n{'='*70}")
print(f"FEATURE ENGINEERING COMPLETE")
print(f"{'='*70}")
print(f"✓ Ready for model training with {N_PCA_COMPONENTS} PCA features")
print(f"✓ This will dramatically speed up ANFIS training!")
print(f"  (Rules reduced from 3^{X_original.shape[1]} to 3^{N_PCA_COMPONENTS})")""")

# Save the updated notebook
with open('/home/user/Smart_Electric_Grid/notebook_part2.json', 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"✓ Part 2 created: {len(notebook['cells'])} total cells")
