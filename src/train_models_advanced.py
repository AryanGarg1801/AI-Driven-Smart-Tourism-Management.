
import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
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


# ============================================================
# FEATURES
# ============================================================

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
    "district",
    "sector_label"
]

TARGET = "total_trip_expenditure"

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
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Weighted_MAE": weighted_mae,
        "Weighted_RMSE": weighted_rmse,
        "Weighted_R2": weighted_r2
    }


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(numeric_features, categorical_features):

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=True
        )

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("ADVANCED EXPENDITURE PREDICTION EXPERIMENT")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

required_columns = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + GROUP_COLUMNS
    + [TARGET, "mult"]
)

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df = df.dropna(subset=required_columns).copy()

df["survey_weight"] = df["mult"] / 100

df = df[df["survey_weight"] > 0].copy()

print("\nDataset shape:", df.shape)
print("Unique households:", df[GROUP_COLUMNS].drop_duplicates().shape[0])


# ============================================================
# HOUSEHOLD GROUPS
# ============================================================

groups = (
    df[GROUP_COLUMNS]
    .astype(str)
    .agg("_".join, axis=1)
)

# Outer split: reserve 20% for final testing.
outer_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

development_idx, test_idx = next(
    outer_split.split(df, groups=groups)
)

development = df.iloc[development_idx].copy()
test = df.iloc[test_idx].copy()

# Inner split: training and validation, without household overlap.
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
print("Training records:", len(train))
print("Validation records:", len(validation))
print("Testing records:", len(test))

train_households = set(
    train[GROUP_COLUMNS].astype(str).agg("_".join, axis=1)
)

val_households = set(
    validation[GROUP_COLUMNS].astype(str).agg("_".join, axis=1)
)

test_households = set(
    test[GROUP_COLUMNS].astype(str).agg("_".join, axis=1)
)

assert train_households.isdisjoint(val_households)
assert train_households.isdisjoint(test_households)
assert val_households.isdisjoint(test_households)

print("Household overlap: 0")


# ============================================================
# PREPARE TARGETS AND WEIGHTS
# ============================================================

y_train = train[TARGET].to_numpy()
y_val = validation[TARGET].to_numpy()
y_test = test[TARGET].to_numpy()

w_train = train["survey_weight"].to_numpy()
w_val = validation["survey_weight"].to_numpy()
w_test = test["survey_weight"].to_numpy()


# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================

feature_sets = {
    "Full_Features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    "Without_District": NUMERIC_FEATURES + [
        col for col in CATEGORICAL_FEATURES
        if col != "district"
    ]
}

model_types = {
    "Random_Forest": RandomForestRegressor(
        n_estimators=100,
        max_features=0.8,
        min_samples_leaf=2,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=1
    ),

    "Extra_Trees": ExtraTreesRegressor(
        n_estimators=100,
        max_features=0.8,
        min_samples_leaf=2,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=1
    )
}

target_types = ["Raw_Target", "Log_Target"]

results = []
best_validation_r2 = -np.inf
best_configuration = None
best_pipeline = None


# ============================================================
# RUN EXPERIMENTS
# ============================================================

total_experiments = (
    len(feature_sets) * len(model_types) * len(target_types)
)

experiment_number = 0

for feature_name, selected_features in feature_sets.items():

    numeric_features = [
        col for col in NUMERIC_FEATURES
        if col in selected_features
    ]

    categorical_features = [
        col for col in CATEGORICAL_FEATURES
        if col in selected_features
    ]

    X_train = train[selected_features]
    X_val = validation[selected_features]

    for target_name in target_types:

        if target_name == "Raw_Target":
            transformed_y_train = y_train

        else:
            transformed_y_train = np.log1p(y_train)

        for model_name, base_model in model_types.items():

            experiment_number += 1

            print("\n" + "=" * 65)
            print(
                f"Experiment {experiment_number}/{total_experiments}"
            )
            print("Model:", model_name)
            print("Features:", feature_name)
            print("Target:", target_name)

            model = Pipeline(steps=[
                (
                    "preprocessor",
                    create_preprocessor(
                        numeric_features,
                        categorical_features
                    )
                ),
                ("model", base_model)
            ])

            start_time = time.time()

            model.fit(
                X_train,
                transformed_y_train,
                model__sample_weight=w_train
            )

            val_predictions = model.predict(X_val)

            if target_name == "Log_Target":
                val_predictions = np.expm1(val_predictions)

            val_predictions = np.maximum(
                val_predictions, 0
            )

            metrics = calculate_metrics(
                y_val,
                val_predictions,
                w_val
            )

            elapsed = time.time() - start_time

            result = {
                "Model": model_name,
                "Feature_Set": feature_name,
                "Target_Transformation": target_name,
                **metrics,
                "Training_Time_Seconds": elapsed
            }

            results.append(result)

            print("\nValidation Results:")
            print(f"R2: {metrics['R2']:.4f}")
            print(f"Weighted R2: {metrics['Weighted_R2']:.4f}")
            print(f"MAE: INR {metrics['MAE']:.2f}")
            print(f"RMSE: INR {metrics['RMSE']:.2f}")
            print(f"Time: {elapsed:.1f} seconds")

            if metrics["R2"] > best_validation_r2:

                best_validation_r2 = metrics["R2"]

                best_configuration = {
                    "Model": model_name,
                    "Feature_Set": feature_name,
                    "Target_Transformation": target_name,
                    "Features": selected_features
                }

                best_pipeline = model


# ============================================================
# SAVE VALIDATION COMPARISON
# ============================================================

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="R2",
    ascending=False
)

comparison_path = os.path.join(
    OUTPUT_DIR,
    "advanced_validation_comparison.csv"
)

comparison.to_csv(comparison_path, index=False)

print("\n========== VALIDATION MODEL COMPARISON ==========")

print(
    comparison[
        [
            "Model",
            "Feature_Set",
            "Target_Transformation",
            "R2",
            "Weighted_R2",
            "MAE",
            "RMSE"
        ]
    ].to_string(index=False)
)


# ============================================================
# FINAL TRAINING WITH TRAIN + VALIDATION
# ============================================================

print("\n========== FINAL MODEL TRAINING ==========")

print("Selected configuration:", best_configuration)

selected_features = best_configuration["Features"]

numeric_features = [
    col for col in NUMERIC_FEATURES
    if col in selected_features
]

categorical_features = [
    col for col in CATEGORICAL_FEATURES
    if col in selected_features
]

final_train = pd.concat(
    [train, validation],
    axis=0
)

X_final_train = final_train[selected_features]
X_test = test[selected_features]

y_final_train = final_train[TARGET].to_numpy()
w_final_train = final_train["survey_weight"].to_numpy()

if best_configuration["Target_Transformation"] == "Log_Target":
    final_y = np.log1p(y_final_train)
else:
    final_y = y_final_train

final_model = Pipeline(steps=[
    (
        "preprocessor",
        create_preprocessor(
            numeric_features,
            categorical_features
        )
    ),
    ("model", model_types[best_configuration["Model"]])
])

final_model.fit(
    X_final_train,
    final_y,
    model__sample_weight=w_final_train
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

test_predictions = final_model.predict(X_test)

if best_configuration["Target_Transformation"] == "Log_Target":
    test_predictions = np.expm1(test_predictions)

test_predictions = np.maximum(test_predictions, 0)

test_metrics = calculate_metrics(
    y_test,
    test_predictions,
    w_test
)

print("\n========== FINAL TEST RESULTS ==========")

for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nPrevious Random Forest baseline:")
print("R2: 0.2831")
print("Weighted R2: 0.3079")


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "best_model_advanced.pkl"
)

joblib.dump(
    {
        "model": final_model,
        "features": selected_features,
        "configuration": best_configuration,
        "test_metrics": test_metrics
    },
    model_path
)

test_results = pd.DataFrame({
    "actual_expenditure": y_test,
    "predicted_expenditure": test_predictions,
    "survey_weight": w_test
})

test_results_path = os.path.join(
    OUTPUT_DIR,
    "advanced_test_predictions.csv"
)

test_results.to_csv(test_results_path, index=False)

print("\n========== FILES SAVED ==========")
print("Validation comparison:", comparison_path)
print("Advanced model:", model_path)
print("Test predictions:", test_results_path)

print("\nADVANCED EXPERIMENT COMPLETED.")
