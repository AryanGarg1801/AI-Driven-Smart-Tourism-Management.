
import os
import random
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


# =====================================================
# 1. REPRODUCIBILITY
# =====================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# =====================================================
# 2. PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "outputs", "overnight_tourism_decoded.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# =====================================================
# 3. LOAD DATA
# =====================================================

df = pd.read_csv(DATA_PATH)

print("\n========== IMPROVED ANN TRAINING ==========")
print("Dataset shape:", df.shape)


# =====================================================
# 4. FEATURES AND TARGET
# =====================================================

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
    "sector_label"
]

features = numeric_features + categorical_features

target = "total_trip_expenditure"

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

weights = (df["mult"].astype(float).values / 100)


# =====================================================
# 5. HOUSEHOLD-WISE 70:10:20 SPLIT
# =====================================================

# First: 80% train + validation, 20% test

split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED
)

train_val_idx, test_idx = next(
    split1.split(X, y, groups)
)

# Second: Divide remaining 80% into
# 70% training and 10% validation overall.

split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.125,
    random_state=SEED
)

train_rel, val_rel = next(
    split2.split(
        X.iloc[train_val_idx],
        y[train_val_idx],
        groups[train_val_idx]
    )
)

train_idx = train_val_idx[train_rel]
val_idx = train_val_idx[val_rel]

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]
X_test = X.iloc[test_idx]

y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]

w_train = weights[train_idx]
w_val = weights[val_idx]
w_test = weights[test_idx]

groups_train = set(groups[train_idx])
groups_val = set(groups[val_idx])
groups_test = set(groups[test_idx])

assert groups_train.isdisjoint(groups_val)
assert groups_train.isdisjoint(groups_test)
assert groups_val.isdisjoint(groups_test)

print("\n========== DATA SPLIT ==========")
print("Training records:", len(train_idx))
print("Validation records:", len(val_idx))
print("Testing records:", len(test_idx))

print("Training households:", len(groups_train))
print("Validation households:", len(groups_val))
print("Testing households:", len(groups_test))

print("Household overlap: 0")


# =====================================================
# 6. PREPROCESSING
# =====================================================

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
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

X_train_processed = np.asarray(
    X_train_processed, dtype=np.float32
)

X_val_processed = np.asarray(
    X_val_processed, dtype=np.float32
)

X_test_processed = np.asarray(
    X_test_processed, dtype=np.float32
)

print("\n========== PREPROCESSING ==========")
print("Training shape:", X_train_processed.shape)
print("Validation shape:", X_val_processed.shape)
print("Testing shape:", X_test_processed.shape)


# =====================================================
# 7. TARGET TRANSFORMATION
# =====================================================

y_train_log = np.log1p(y_train).astype(np.float32)
y_val_log = np.log1p(y_val).astype(np.float32)


# Normalize weights to mean 1 in each set

w_train_norm = w_train / np.mean(w_train)
w_val_norm = w_val / np.mean(w_val)


# =====================================================
# 8. BUILD ANN MODEL
# =====================================================

model = Sequential([
    Input(shape=(X_train_processed.shape[1],)),

    Dense(
        128,
        activation="relu",
        kernel_regularizer=l2(0.001)
    ),
    Dropout(0.25),

    Dense(
        64,
        activation="relu",
        kernel_regularizer=l2(0.001)
    ),
    Dropout(0.20),

    Dense(
        32,
        activation="relu",
        kernel_regularizer=l2(0.001)
    ),

    Dense(1, activation="linear")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss=tf.keras.losses.Huber(delta=1.0),
    metrics=["mae"]
)

print("\n========== MODEL SUMMARY ==========")
model.summary()


# =====================================================
# 9. CALLBACKS
# =====================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=0.00001,
    verbose=1
)


# =====================================================
# 10. TRAIN ANN
# =====================================================

print("\n========== ANN TRAINING STARTED ==========")

history = model.fit(
    X_train_processed,
    y_train_log,

    sample_weight=w_train_norm,

    validation_data=(
        X_val_processed,
        y_val_log,
        w_val_norm
    ),

    epochs=100,
    batch_size=256,

    callbacks=[
        early_stop,
        reduce_lr
    ],

    shuffle=True,
    verbose=1
)


# =====================================================
# 11. PREDICTION
# =====================================================

pred_log = model.predict(
    X_test_processed,
    verbose=0
).flatten()

predictions = np.expm1(pred_log)
predictions = np.maximum(predictions, 0)


# =====================================================
# 12. EVALUATION
# =====================================================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

weighted_mae = mean_absolute_error(
    y_test,
    predictions,
    sample_weight=w_test
)

weighted_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions,
        sample_weight=w_test
    )
)

weighted_r2 = r2_score(
    y_test,
    predictions,
    sample_weight=w_test
)


print("\n========== FINAL ANN RESULTS ==========")

print(f"MAE: INR {mae:.2f}")
print(f"RMSE: INR {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

print(f"Weighted MAE: INR {weighted_mae:.2f}")
print(f"Weighted RMSE: INR {weighted_rmse:.2f}")
print(f"Weighted R2: {weighted_r2:.4f}")


# =====================================================
# 13. SAVE RESULTS
# =====================================================

results = pd.DataFrame([{
    "Model": "Improved ANN",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Weighted_MAE": weighted_mae,
    "Weighted_RMSE": weighted_rmse,
    "Weighted_R2": weighted_r2
}])

results_path = os.path.join(
    MODEL_DIR, "ann_improved_results.csv"
)

results.to_csv(results_path, index=False)


# Save trained ANN

model.save(
    os.path.join(
        MODEL_DIR, "ann_improved_model.keras"
    )
)


# Save preprocessing object

with open(
    os.path.join(
        MODEL_DIR, "ann_improved_preprocessor.pkl"
    ),
    "wb"
) as f:
    pickle.dump(preprocessor, f)


# Save training history

pd.DataFrame(history.history).to_csv(
    os.path.join(
        MODEL_DIR, "ann_improved_history.csv"
    ),
    index=False
)


print("\n========== COMPLETED ==========")
print("Model and results saved in:", MODEL_DIR)
