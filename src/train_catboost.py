
import os
import time
import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "outputs", "overnight_tourism_decoded.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

TARGET = "total_trip_expenditure"

NUMERIC_FEATURES = [
    "household_members",
    "average_member_age",
    "trip_duration_nights"
]

CATEGORICAL_FEATURES = [
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

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

GROUP_COLUMNS = [
    "fsu_srno",
    "second_stage_stratum_no",
    "sample_hhld_no"
]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, weights):

    y_pred = np.maximum(y_pred, 0)

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    r2 = r2_score(y_true, y_pred)

    weighted_mae = np.average(
        np.abs(y_true - y_pred),
        weights=weights
    )

    weighted_rmse = np.sqrt(
        np.average(
            (y_true - y_pred) ** 2,
            weights=weights
        )
    )

    weighted_mean = np.average(y_true, weights=weights)

    weighted_r2 = 1 - (
        np.sum(weights * (y_true - y_pred) ** 2)
        /
        np.sum(weights * (y_true - weighted_mean) ** 2)
    )

    return {
        "R2": r2,
        "Weighted_R2": weighted_r2,
        "MAE": mae,
        "RMSE": rmse,
        "Weighted_MAE": weighted_mae,
        "Weighted_RMSE": weighted_rmse
    }


# ============================================================
# CATBOOST MODEL
# ============================================================

def create_model(iterations=1200):

    return CatBoostRegressor(
        iterations=iterations,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        l2_leaf_reg=5,
        random_seed=RANDOM_STATE,
        thread_count=2,
        task_type="CPU",
        allow_writing_files=False,
        verbose=100
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("CATBOOST EXPENDITURE PREDICTION")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

required_columns = (
    FEATURES + GROUP_COLUMNS + [TARGET, "mult"]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

df = df.dropna(
    subset=GROUP_COLUMNS + [TARGET, "mult"]
).copy()

df["survey_weight"] = df["mult"] / 100

df = df[df["survey_weight"] > 0].copy()

# Numeric features
for col in NUMERIC_FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Handle categorical values directly as strings
for col in CATEGORICAL_FEATURES:
    df[col] = df[col].fillna("Unknown").astype(str)

df = df.dropna(
    subset=NUMERIC_FEATURES
).copy()

df = df.reset_index(drop=True)

print("\nDataset shape:", df.shape)

print(
    "Unique households:",
    df[GROUP_COLUMNS].drop_duplicates().shape[0]
)


# ============================================================
# SAME HOUSEHOLD-WISE SPLIT AS ADVANCED EXPERIMENT
# ============================================================

groups = (
    df[GROUP_COLUMNS]
    .astype(str)
    .agg("_".join, axis=1)
)

outer_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

development_idx, test_idx = next(
    outer_split.split(df, groups=groups)
)

development = df.iloc[development_idx].copy()
test = df.iloc[test_idx].copy()

development_groups = groups.iloc[development_idx]

inner_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=21
)

train_relative_idx, val_relative_idx = next(
    inner_split.split(
        development,
        groups=development_groups
    )
)

train = development.iloc[train_relative_idx].copy()
validation = development.iloc[val_relative_idx].copy()

print("\n========== DATA SPLIT ==========")
print("Training:", len(train))
print("Validation:", len(validation))
print("Testing:", len(test))


# Verify no household overlap
def household_ids(data):
    return set(
        data[GROUP_COLUMNS]
        .astype(str)
        .agg("_".join, axis=1)
    )


assert household_ids(train).isdisjoint(
    household_ids(validation)
)

assert household_ids(train).isdisjoint(
    household_ids(test)
)

assert household_ids(validation).isdisjoint(
    household_ids(test)
)

print("Household overlap: 0")


# ============================================================
# PREPARE DATA
# ============================================================

X_train = train[FEATURES]
X_val = validation[FEATURES]
X_test = test[FEATURES]

y_train = train[TARGET].to_numpy()
y_val = validation[TARGET].to_numpy()
y_test = test[TARGET].to_numpy()

w_train = train["survey_weight"].to_numpy()
w_val = validation["survey_weight"].to_numpy()
w_test = test["survey_weight"].to_numpy()


# ============================================================
# TRAIN RAW AND LOG TARGET MODELS
# ============================================================

results = []

best_validation_r2 = -np.inf
best_target_type = None
best_iterations = None

print("\n========== CATBOOST EXPERIMENTS ==========")

for target_type in ["Raw_Target", "Log_Target"]:

    print("\n" + "=" * 65)
    print("TARGET:", target_type)
    print("=" * 65)

    if target_type == "Log_Target":
        transformed_train = np.log1p(y_train)
        transformed_val = np.log1p(y_val)
    else:
        transformed_train = y_train
        transformed_val = y_val

    model = create_model()

    start_time = time.time()

    model.fit(
        X_train,
        transformed_train,
        cat_features=CATEGORICAL_FEATURES,
        sample_weight=w_train,
        eval_set=(X_val, transformed_val),
        early_stopping_rounds=100,
        use_best_model=True,
        verbose=100
    )

    elapsed = time.time() - start_time

    predictions = model.predict(X_val)

    if target_type == "Log_Target":
        predictions = np.expm1(predictions)

    predictions = np.maximum(predictions, 0)

    metrics = calculate_metrics(
        y_val,
        predictions,
        w_val
    )

    best_iteration = max(
        1,
        model.get_best_iteration() + 1
    )

    result = {
        "Model": "CatBoost",
        "Target": target_type,
        "Best_Iterations": best_iteration,
        **metrics,
        "Training_Time_Seconds": elapsed
    }

    results.append(result)

    print("\nValidation Results:")

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("Best iterations:", best_iteration)
    print("Training time:", round(elapsed, 2), "seconds")

    if metrics["R2"] > best_validation_r2:

        best_validation_r2 = metrics["R2"]
        best_target_type = target_type
        best_iterations = best_iteration


# ============================================================
# SAVE VALIDATION RESULTS
# ============================================================

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    "R2",
    ascending=False
)

comparison_path = os.path.join(
    OUTPUT_DIR,
    "catboost_validation_comparison.csv"
)

comparison.to_csv(comparison_path, index=False)

print("\n========== CATBOOST VALIDATION COMPARISON ==========")
print(comparison.to_string(index=False))


# ============================================================
# FINAL TRAINING ON TRAIN + VALIDATION
# ============================================================

print("\n========== FINAL CATBOOST TRAINING ==========")

final_train = pd.concat(
    [train, validation],
    ignore_index=True
)

X_final_train = final_train[FEATURES]
y_final_train = final_train[TARGET].to_numpy()

w_final_train = final_train["survey_weight"].to_numpy()

if best_target_type == "Log_Target":
    final_y = np.log1p(y_final_train)
else:
    final_y = y_final_train

final_model = create_model(
    iterations=best_iterations
)

final_model.fit(
    X_final_train,
    final_y,
    cat_features=CATEGORICAL_FEATURES,
    sample_weight=w_final_train,
    verbose=100
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n========== FINAL CATBOOST TEST RESULTS ==========")

test_predictions = final_model.predict(X_test)

if best_target_type == "Log_Target":
    test_predictions = np.expm1(test_predictions)

test_predictions = np.maximum(test_predictions, 0)

test_metrics = calculate_metrics(
    y_test,
    test_predictions,
    w_test
)

for key, value in test_metrics.items():
    print(f"{key}: {value:.4f}")


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "best_model_catboost.cbm"
)

final_model.save_model(model_path)


# Save metadata
metadata_path = os.path.join(
    OUTPUT_DIR,
    "catboost_metadata.pkl"
)

joblib.dump(
    {
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_transformation": best_target_type,
        "best_iterations": best_iterations,
        "test_metrics": test_metrics
    },
    metadata_path
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": final_model.get_feature_importance()
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

importance_path = os.path.join(
    OUTPUT_DIR,
    "catboost_feature_importance.csv"
)

importance.to_csv(
    importance_path,
    index=False
)

print("\n========== TOP 10 IMPORTANT FEATURES ==========")
print(importance.head(10).to_string(index=False))


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_path = os.path.join(
    OUTPUT_DIR,
    "catboost_test_predictions.csv"
)

pd.DataFrame({
    "actual_expenditure": y_test,
    "predicted_expenditure": test_predictions,
    "survey_weight": w_test
}).to_csv(
    prediction_path,
    index=False
)


# ============================================================
# COMPLETION
# ============================================================

print("\n========== FILES SAVED ==========")
print("Model:", model_path)
print("Metadata:", metadata_path)
print("Validation comparison:", comparison_path)
print("Feature importance:", importance_path)
print("Test predictions:", prediction_path)

print("\nSELECTED TARGET:", best_target_type)
print("CATBOOST EXPERIMENT COMPLETED.")
