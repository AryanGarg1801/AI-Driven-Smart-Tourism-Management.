
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "outputs", "overnight_tourism_decoded.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)

print("========== ANN EXPENDITURE PREDICTION ==========")

# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

target = "total_trip_expenditure"

numeric_features = [
    "household_members",
    "average_member_age",
    "trip_duration_nights"
]

categorical_features = [
    "trip_type_label",
    "travel_mode_label",
    "accommodation_type_label",
    "leading_purpose_label",
    "trip_month_label",
    "main_destination_label",
    "destination_state_label",
    "state",
    "district",
    "sector_label"
]

features = numeric_features + categorical_features

group_columns = [
    "fsu_srno",
    "second_stage_stratum_no",
    "sample_hhld_no"
]

X = df[features].copy()
y = df[target].astype(float).values

groups = (
    df[group_columns]
    .astype(str)
    .agg("_".join, axis=1)
    .values
)

# Survey weights are used for training and evaluation,
# not as prediction features.
weights = (df["mult"].astype(float).values / 100.0)

# ==========================================
# 3. HOUSEHOLD-WISE DATA SPLIT
# ==========================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

w_train = weights[train_idx]
w_test = weights[test_idx]

groups_train = groups[train_idx]
groups_test = groups[test_idx]

print("\n========== DATA SPLIT ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training households:", len(np.unique(groups_train)))
print("Testing households:", len(np.unique(groups_test)))

overlap = set(groups_train).intersection(set(groups_test))
print("Household overlap:", len(overlap))

assert len(overlap) == 0, "Household leakage detected!"

# ==========================================
# 4. DATA PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float32
            ),
            categorical_features
        )
    ],
    verbose_feature_names_out=False
)

print("\nPreprocessing training data...")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

X_train_processed = np.asarray(
    X_train_processed, dtype=np.float32
)

X_test_processed = np.asarray(
    X_test_processed, dtype=np.float32
)

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

# ==========================================
# 5. TARGET TRANSFORMATION
# ==========================================

y_train_log = np.log1p(y_train).astype(np.float32)

# Normalize training weights to mean 1.
# This preserves their relative importance.
w_train_normalized = (
    w_train / np.mean(w_train)
).astype(np.float32)

# ==========================================
# 6. BUILD ANN MODEL
# ==========================================

model = Sequential([
    Input(shape=(X_train_processed.shape[1],)),

    Dense(128, activation="relu"),
    Dropout(0.20),

    Dense(64, activation="relu"),
    Dropout(0.15),

    Dense(32, activation="relu"),

    Dense(1, activation="linear")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
)

print("\n========== ANN MODEL ==========")
model.summary()

# ==========================================
# 7. CALLBACKS
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

# ==========================================
# 8. TRAIN ANN
# ==========================================

print("\n========== ANN TRAINING STARTED ==========")

history = model.fit(
    X_train_processed,
    y_train_log,
    sample_weight=w_train_normalized,
    validation_split=0.15,
    epochs=50,
    batch_size=256,
    callbacks=[early_stopping, reduce_lr],
    verbose=1,
    shuffle=True
)

# ==========================================
# 9. PREDICTION
# ==========================================

print("\n========== MODEL EVALUATION ==========")

predictions_log = model.predict(
    X_test_processed,
    verbose=0
).flatten()

predictions = np.expm1(predictions_log)
predictions = np.maximum(predictions, 0)

# ==========================================
# 10. EVALUATION METRICS
# ==========================================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

weighted_mae = np.average(
    np.abs(y_test - predictions),
    weights=w_test
)

weighted_rmse = np.sqrt(
    np.average(
        (y_test - predictions) ** 2,
        weights=w_test
    )
)

weighted_r2 = r2_score(
    y_test,
    predictions,
    sample_weight=w_test
)

print("\n========== ANN RESULTS ==========")

print(f"MAE: INR {mae:.2f}")
print(f"RMSE: INR {rmse:.2f}")
print(f"R2 Score: {r2:.6f}")

print("\n========== WEIGHTED RESULTS ==========")

print(f"Weighted MAE: INR {weighted_mae:.2f}")
print(f"Weighted RMSE: INR {weighted_rmse:.2f}")
print(f"Weighted R2: {weighted_r2:.6f}")

# ==========================================
# 11. SAVE RESULTS
# ==========================================

results = pd.DataFrame([{
    "Model": "Artificial Neural Network",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Weighted_MAE": weighted_mae,
    "Weighted_RMSE": weighted_rmse,
    "Weighted_R2": weighted_r2,
    "Best_Epoch": len(history.history["loss"])
}])

results_path = os.path.join(
    OUTPUT_DIR,
    "ann_results.csv"
)

results.to_csv(results_path, index=False)

model.save(
    os.path.join(OUTPUT_DIR, "ann_expenditure_model.keras")
)

joblib.dump(
    preprocessor,
    os.path.join(OUTPUT_DIR, "ann_preprocessor.pkl")
)

print("\n========== ANN COMPLETED ==========")
print("Model saved successfully.")
print("Preprocessor saved successfully.")
print("Results saved to:", results_path)
