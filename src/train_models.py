
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 1. PATHS AND DATA LOADING
# --------------------------------------------------

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_path = os.path.join(
    base_path, "outputs", "overnight_tourism_decoded.csv"
)

output_dir = os.path.join(base_path, "outputs", "models")
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_path)

print("========== MODEL TRAINING PIPELINE ==========")
print("Dataset shape:", df.shape)

# --------------------------------------------------
# 2. FEATURES, TARGET AND SURVEY WEIGHTS
# --------------------------------------------------

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

target = "total_trip_expenditure"

group_columns = [
    "fsu_srno",
    "second_stage_stratum_no",
    "sample_hhld_no"
]

X = df[numeric_features + categorical_features].copy()
y = df[target].copy()

# Official survey weight: Mult / 100
weights = df["mult"] / 100

# Same household must remain in one split
groups = df[group_columns].astype(str).agg("_".join, axis=1)

# --------------------------------------------------
# 3. HOUSEHOLD-WISE TRAIN-TEST SPLIT
# --------------------------------------------------

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

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

w_train = weights.iloc[train_idx]
w_test = weights.iloc[test_idx]

train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])

print("\n========== DATA SPLIT ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training households:", len(train_groups))
print("Testing households:", len(test_groups))
print("Household overlap:", len(train_groups.intersection(test_groups)))

# --------------------------------------------------
# 4. PREPROCESSING
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# --------------------------------------------------
# 5. DEFINE MODELS
# --------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

# Log transformation reduces the effect of extreme target values.
# Predictions are converted back to INR for evaluation.
y_train_log = np.log1p(y_train)

results = []
trained_models = {}

# --------------------------------------------------
# 6. TRAIN AND EVALUATE
# --------------------------------------------------

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(
        X_train,
        y_train_log,
        model__sample_weight=w_train
    )

    predictions_log = pipeline.predict(X_test)
    predictions = np.expm1(predictions_log)

    # Expenditure cannot be negative
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
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

    weighted_r2 = 1 - (
        np.sum(w_test * (y_test - predictions) ** 2)
        /
        np.sum(w_test * (y_test - np.average(y_test, weights=w_test)) ** 2)
    )

    results.append({
        "Model": name,
        "MAE_INR": mae,
        "RMSE_INR": rmse,
        "R2": r2,
        "Weighted_MAE_INR": weighted_mae,
        "Weighted_RMSE_INR": weighted_rmse,
        "Weighted_R2": weighted_r2
    })

    trained_models[name] = pipeline

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2:", round(r2, 4))
    print("Weighted MAE:", round(weighted_mae, 2))
    print("Weighted RMSE:", round(weighted_rmse, 2))
    print("Weighted R2:", round(weighted_r2, 4))

# --------------------------------------------------
# 7. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Weighted_MAE_INR"
)

results_path = os.path.join(
    output_dir, "model_comparison.csv"
)

results_df.to_csv(results_path, index=False)

best_model_name = results_df.iloc[0]["Model"]

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

print("\nBest model by weighted MAE:", best_model_name)

# Save best trained model
import joblib

best_model_path = os.path.join(
    output_dir, "best_model.pkl"
)

joblib.dump(
    trained_models[best_model_name],
    best_model_path
)

print("\nResults saved:", results_path)
print("Best model saved:", best_model_path)

print("\n========== TRAINING COMPLETED ==========")
