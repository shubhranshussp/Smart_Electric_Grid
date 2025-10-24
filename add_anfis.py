#!/usr/bin/env python3
"""
Add ANFIS (Enhanced) model section to notebook
Optimized for <3% MAPE target with PCA features
"""

import json

# Load existing notebook
with open('/home/user/Smart_Electric_Grid/notebook_part3.json', 'r') as f:
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
# ANFIS MODEL SECTION (SEPARATE - ENHANCED)
# =============================================================================
print("Adding ANFIS Model section (ENHANCED for <3% MAPE)...")

add_markdown_cell("## **ANFIS Model (Enhanced - Target <3% MAPE)**")

anfis_code = """# =============================================================================
# ANFIS MODEL - ADAPTIVE NEURO-FUZZY INFERENCE SYSTEM (ENHANCED)
# =============================================================================

print("="*70)
print("ANFIS MODEL - ENHANCED VERSION (PCA Features)")
print("="*70)

# =============================================================================
# ANFIS CLASS DEFINITION
# =============================================================================

class ANFIS:
    \"\"\"
    Enhanced ANFIS with optimizations for better performance

    Key Improvements:
    - Works with PCA-reduced features (12 instead of 26)
    - Uses n_mfs=2 for efficiency (2^12 = 4,096 rules instead of 3^26 = 2.3M)
    - Better training with early stopping
    - Enhanced initialization
    \"\"\"

    def __init__(self, n_inputs, n_mfs=2, learning_rate=0.01):
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.n_rules = n_mfs ** n_inputs
        self.learning_rate = learning_rate

        print(f"\\n✓ ANFIS initialized:")
        print(f"  Inputs: {n_inputs}")
        print(f"  Membership functions per input: {n_mfs}")
        print(f"  Total fuzzy rules: {self.n_rules:,}")

        self._initialize_parameters()
        self.train_losses = []
        self.val_losses = []

    def _initialize_parameters(self):
        \"\"\"Initialize membership function and consequent parameters\"\"\"

        # Premise parameters (Gaussian MFs: mean and std)
        self.premise_params = np.random.randn(self.n_inputs, self.n_mfs, 2) * 0.5

        # Initialize means uniformly
        for i in range(self.n_inputs):
            self.premise_params[i, :, 0] = np.linspace(-2, 2, self.n_mfs)
            self.premise_params[i, :, 1] = np.ones(self.n_mfs) * 1.0

        # Consequent parameters (TSK model)
        self.consequent_params = np.random.randn(self.n_rules, self.n_inputs + 1) * 0.1

    def _gaussian_mf(self, x, mean, std):
        \"\"\"Gaussian membership function\"\"\"
        return np.exp(-0.5 * ((x - mean) / (std + 1e-10)) ** 2)

    def _forward(self, X):
        \"\"\"
        Forward pass through ANFIS

        Layer 1: Fuzzification
        Layer 2: Rule firing strengths
        Layer 3: Normalization
        Layer 4: Consequent computation
        Layer 5: Defuzzification (output)
        \"\"\"
        batch_size = X.shape[0]

        # Layer 1: Fuzzification - compute membership degrees
        memberships = np.zeros((batch_size, self.n_inputs, self.n_mfs))
        for i in range(self.n_inputs):
            for j in range(self.n_mfs):
                mean = self.premise_params[i, j, 0]
                std = self.premise_params[i, j, 1]
                memberships[:, i, j] = self._gaussian_mf(X[:, i], mean, std)

        # Layer 2: Rule firing strengths (product of membership degrees)
        # Generate all combinations of membership functions
        mf_indices = np.array(list(itertools.product(range(self.n_mfs), repeat=self.n_inputs)))

        firing_strengths = np.ones((batch_size, self.n_rules))
        for rule_idx in range(self.n_rules):
            for input_idx in range(self.n_inputs):
                mf_idx = mf_indices[rule_idx, input_idx]
                firing_strengths[:, rule_idx] *= memberships[:, input_idx, mf_idx]

        # Layer 3: Normalization
        firing_sum = np.sum(firing_strengths, axis=1, keepdims=True) + 1e-10
        normalized_firing = firing_strengths / firing_sum

        # Layer 4: Consequent computation (TSK model)
        # Extend X with bias term
        X_extended = np.column_stack([np.ones(batch_size), X])

        # Compute rule outputs
        rule_outputs = np.dot(X_extended, self.consequent_params.T)

        # Layer 5: Defuzzification (weighted sum)
        output = np.sum(normalized_firing * rule_outputs, axis=1)

        # Store for backward pass
        self._cache = {
            'X': X,
            'X_extended': X_extended,
            'memberships': memberships,
            'firing_strengths': firing_strengths,
            'normalized_firing': normalized_firing,
            'rule_outputs': rule_outputs,
            'mf_indices': mf_indices
        }

        return output

    def _backward(self, X, y_true, y_pred):
        \"\"\"Backward pass - update parameters using gradient descent\"\"\"

        batch_size = X.shape[0]
        error = y_pred - y_true

        # Retrieve cached values
        X_extended = self._cache['X_extended']
        normalized_firing = self._cache['normalized_firing']

        # Update consequent parameters (LSE - Least Squares Estimation)
        # Faster than gradient descent for consequent parameters
        for rule_idx in range(self.n_rules):
            w = normalized_firing[:, rule_idx].reshape(-1, 1)
            weighted_X = X_extended * w

            try:
                # Solve weighted least squares
                self.consequent_params[rule_idx] = np.linalg.lstsq(
                    weighted_X, y_true * w.ravel(), rcond=None
                )[0]
            except:
                # If singular, use gradient descent
                grad = np.dot(weighted_X.T, error) / batch_size
                self.consequent_params[rule_idx] -= self.learning_rate * grad

        # Update premise parameters (gradient descent)
        # Simplified update - full derivation is complex
        memberships = self._cache['memberships']
        firing_strengths = self._cache['firing_strengths']
        mf_indices = self._cache['mf_indices']

        for i in range(self.n_inputs):
            for j in range(self.n_mfs):
                # Find rules using this membership function
                relevant_rules = mf_indices[:, i] == j

                # Compute gradient (simplified)
                mean = self.premise_params[i, j, 0]
                std = self.premise_params[i, j, 1]

                mf_values = memberships[:, i, j]

                # Gradient w.r.t. mean
                grad_mean = np.mean(error * (X[:, i] - mean) / (std**2 + 1e-10) * mf_values)

                # Gradient w.r.t. std
                grad_std = np.mean(error * ((X[:, i] - mean)**2) / (std**3 + 1e-10) * mf_values)

                # Update with clipping
                self.premise_params[i, j, 0] -= self.learning_rate * grad_mean
                self.premise_params[i, j, 1] -= self.learning_rate * grad_std

                # Ensure std stays positive
                self.premise_params[i, j, 1] = max(0.1, self.premise_params[i, j, 1])

    def fit(self, X_train, y_train, X_val=None, y_val=None, max_epochs=300, patience=50, verbose=True):
        \"\"\"
        Train ANFIS model

        Parameters:
        -----------
        X_train, y_train : Training data
        X_val, y_val : Validation data (optional, for early stopping)
        max_epochs : Maximum training epochs
        patience : Early stopping patience
        verbose : Print progress
        \"\"\"

        print(f"\\n{'='*70}")
        print(f"Training ANFIS")
        print(f"{'='*70}")
        print(f"  Max epochs: {max_epochs}")
        print(f"  Patience: {patience}")
        print(f"  Learning rate: {self.learning_rate}")

        best_val_loss = np.inf
        patience_counter = 0

        for epoch in range(max_epochs):
            # Forward pass
            y_pred_train = self._forward(X_train)

            # Compute training loss
            train_loss = np.mean((y_pred_train - y_train) ** 2)
            self.train_losses.append(train_loss)

            # Backward pass
            self._backward(X_train, y_train, y_pred_train)

            # Validation
            if X_val is not None and y_val is not None:
                y_pred_val = self._forward(X_val)
                val_loss = np.mean((y_pred_val - y_val) ** 2)
                self.val_losses.append(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best parameters
                    self.best_premise = self.premise_params.copy()
                    self.best_consequent = self.consequent_params.copy()
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"\\n✓ Early stopping at epoch {epoch+1}")
                    print(f"  Best validation loss: {best_val_loss:.6f}")
                    # Restore best parameters
                    self.premise_params = self.best_premise
                    self.consequent_params = self.best_consequent
                    break

                if verbose and (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1:3d}/{max_epochs}: "
                          f"Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
            else:
                if verbose and (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1:3d}/{max_epochs}: Train Loss={train_loss:.6f}")

        print(f"\\n✓ Training complete ({len(self.train_losses)} epochs)")

        return self

    def predict(self, X):
        \"\"\"Generate predictions\"\"\"
        return self._forward(X)


# =============================================================================
# LOAD DATA (PCA-TRANSFORMED)
# =============================================================================

print(f"\\n{'='*70}")
print("Loading PCA-Transformed Data")
print(f"{'='*70}")

X_train_anfis = np.load('X_train_pca.npy')
X_val_anfis = np.load('X_val_pca.npy')
X_test_anfis = np.load('X_test_pca.npy')

y_train_anfis = np.load('y_train_scaled.npy')
y_val_anfis = np.load('y_val_scaled.npy')
y_test_anfis = np.load('y_test_scaled.npy')

print(f"\\n✓ Data loaded:")
print(f"  Train: {X_train_anfis.shape}")
print(f"  Val:   {X_val_anfis.shape}")
print(f"  Test:  {X_test_anfis.shape}")
print(f"\\n✓ Using {X_train_anfis.shape[1]} PCA features")
print(f"  (Dramatically reduces rule complexity!)")

# =============================================================================
# TRAIN ANFIS MODEL
# =============================================================================

print(f"\\n{'='*70}")
print("Creating and Training ANFIS Model")
print(f"{'='*70}")

# Create ANFIS with optimized configuration
anfis_model = ANFIS(
    n_inputs=X_train_anfis.shape[1],  # 12 PCA features
    n_mfs=ANFIS_CONFIG['n_membership_functions'],  # 2 MFs per input
    learning_rate=ANFIS_CONFIG['learning_rate']  # 0.01
)

# Train model
anfis_model.fit(
    X_train_anfis, y_train_anfis,
    X_val_anfis, y_val_anfis,
    max_epochs=ANFIS_CONFIG['max_epochs'],  # 300
    patience=ANFIS_CONFIG['patience'],  # 50
    verbose=True
)

# =============================================================================
# GENERATE PREDICTIONS
# =============================================================================

print(f"\\n{'='*70}")
print("Generating Test Predictions")
print(f"{'='*70}")

# Predict on test set
y_test_pred_scaled_anfis = anfis_model.predict(X_test_anfis)

# Inverse transform
with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

y_test_anfis_actual = scaler_y.inverse_transform(y_test_anfis.reshape(-1, 1)).ravel()
y_test_anfis_pred = scaler_y.inverse_transform(y_test_pred_scaled_anfis.reshape(-1, 1)).ravel()

print(f"✓ Generated {len(y_test_anfis_pred)} predictions")

# =============================================================================
# EVALUATE MODEL
# =============================================================================

anfis_metrics = calculate_metrics(y_test_anfis_actual, y_test_anfis_pred, "ANFIS (Enhanced)")

# Save results
np.save('anfis_predictions.npy', y_test_anfis_pred)
with open('anfis_model_enhanced.pkl', 'wb') as f:
    pickle.dump(anfis_model, f)

# =============================================================================
# VISUALIZATIONS
# =============================================================================

# Prediction plots
plot_predictions(y_test_anfis_actual, y_test_anfis_pred, "ANFIS (Enhanced)", "anfis_predictions.png")

# Training history
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
epochs = range(1, len(anfis_model.train_losses) + 1)
ax.plot(epochs, anfis_model.train_losses, label='Training Loss', linewidth=2)
if anfis_model.val_losses:
    ax.plot(epochs, anfis_model.val_losses, label='Validation Loss', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('ANFIS Training History')
ax.legend()
ax.grid(True)
plt.savefig('anfis_training_curve.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\\n{'='*70}")
print("ANFIS MODEL COMPLETE")
print(f"{'='*70}")
print(f"✓ Rules: {anfis_model.n_rules:,} (reduced from 2.3M with original features)")
print(f"✓ Epochs trained: {len(anfis_model.train_losses)}")
print(f"✓ MAPE: {anfis_metrics['MAPE']:.2f}%")

if anfis_metrics['MAPE'] < 3.0:
    print(f"\\n🎯 TARGET ACHIEVED: MAPE < 3% !")
else:
    print(f"\\n⚠ Target: <3% MAPE (Current: {anfis_metrics['MAPE']:.2f}%)")

print(f"✓ Model saved: anfis_model_enhanced.pkl")
"""

add_code_cell(anfis_code)

print("✓ ANFIS section added (separate, enhanced)")

# Save final notebook
with open('/home/user/Smart_Electric_Grid/notebook_part4.json', 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"\n✓ Part 4 saved: {len(notebook['cells'])} total cells")
print("✓ All three models added successfully!")
print("✓ Next: Adding final comparison section...")
