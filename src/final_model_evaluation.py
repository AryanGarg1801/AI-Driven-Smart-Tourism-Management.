
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# ============================================================
# FINAL MODEL EVALUATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "final_evaluation")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("FINAL EXPENDITURE PREDICTION MODEL EVALUATION")
print("=" * 65)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):
    """Find a column using possible column names."""
    lower_columns = {
        str(col).lower().strip(): col for col in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_columns:
            return lower_columns[candidate.lower()]

    return None


def calculate_metrics(actual, predicted, weights=None):
    """Calculate regression evaluation metrics."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        raise ValueError("No valid prediction records found.")

    results = {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }

    if weights is not None:
        weights = np.asarray(weights, dtype=float)[valid]

        valid_weights = np.isfinite(weights) & (weights > 0)

        if valid_weights.any():
            y = actual[valid_weights]
            p = predicted[valid_weights]
            w = weights[valid_weights]

            weighted_mae = np.average(np.abs(y - p), weights=w)
            weighted_mse = np.average((y - p) ** 2, weights=w)

            weighted_mean = np.average(y, weights=w)
            weighted_total = np.sum(w * (y - weighted_mean) ** 2)
            weighted_residual = np.sum(w * (y - p) ** 2)

            weighted_r2 = (
                1 - weighted_residual / weighted_total
                if weighted_total > 0 else np.nan
            )

            results.update({
                "Weighted_MAE": weighted_mae,
                "Weighted_RMSE": np.sqrt(weighted_mse),
                "Weighted_R2": weighted_r2
            })

    return results


def load_predictions(filename, model_name):
    """Load saved predictions and identify actual/predicted columns."""

    path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(path):
        print(f"\nWARNING: File not found: {filename}")
        return None

    df = pd.read_csv(path)

    print(f"\nLoading: {filename}")
    print("Columns:", list(df.columns))

    actual_col = find_column(df, [
        "actual",
        "actual_expenditure",
        "actual_total_trip_expenditure",
        "total_trip_expenditure",
        "y_true",
        "true_value"
    ])

    predicted_col = find_column(df, [
        "predicted",
        "predicted_expenditure",
        "predicted_total_trip_expenditure",
        "prediction",
        "y_pred",
        "predicted_value"
    ])

    if actual_col is None or predicted_col is None:
        raise ValueError(
            f"Could not identify actual/predicted columns in {filename}. "
            f"Available columns: {list(df.columns)}"
        )

    actual = pd.to_numeric(df[actual_col], errors="coerce")
    predicted = pd.to_numeric(df[predicted_col], errors="coerce")

    weights_col = find_column(df, [
        "survey_weight",
        "weight",
        "mult"
    ])

    weights = (
        pd.to_numeric(df[weights_col], errors="coerce")
        if weights_col is not None else None
    )

    valid = actual.notna() & predicted.notna()

    if weights is not None:
        weights = weights[valid].to_numpy()

    actual = actual[valid].to_numpy()
    predicted = predicted[valid].to_numpy()

    result = calculate_metrics(actual, predicted, weights)

    print(f"Valid records: {len(actual):,}")
    print(f"R2: {result['R2']:.4f}")
    print(f"MAE: Rs. {result['MAE']:,.2f}")
    print(f"RMSE: Rs. {result['RMSE']:,.2f}")

    return {
        "model": model_name,
        "actual": actual,
        "predicted": predicted,
        "metrics": result
    }


# ============================================================
# 1. LOAD SAVED MODEL PREDICTIONS
# ============================================================

prediction_results = []

catboost_result = load_predictions(
    "catboost_test_predictions.csv",
    "CatBoost"
)

if catboost_result is not None:
    prediction_results.append(catboost_result)

advanced_result = load_predictions(
    "advanced_test_predictions.csv",
    "Advanced Random Forest"
)

if advanced_result is not None:
    prediction_results.append(advanced_result)


# ============================================================
# 2. LOAD ORIGINAL MODEL COMPARISON
# ============================================================

comparison_path = os.path.join(
    MODEL_DIR,
    "model_comparison.csv"
)

comparison_results = []

if os.path.exists(comparison_path):

    original_df = pd.read_csv(comparison_path)

    print("\n========== ORIGINAL MODEL COMPARISON ==========")
    print(original_df.to_string(index=False))

    model_col = find_column(original_df, ["Model", "model"])

    if model_col is not None:

        for _, row in original_df.iterrows():

            comparison_results.append({
                "Model": str(row[model_col]) + " (Baseline)",
                "MAE": row.get("MAE", row.get("MAE_INR", np.nan)),
                "RMSE": row.get("RMSE", row.get("RMSE_INR", np.nan)),
                "R2": row.get("R2", np.nan),
                "Weighted_MAE": row.get(
                    "Weighted_MAE",
                    row.get("Weighted_MAE_INR", np.nan)
    ),
                "Weighted_RMSE": row.get(
                    "Weighted_RMSE",
                    row.get("Weighted_RMSE_INR", np.nan)
    ),
                "Weighted_R2": row.get("Weighted_R2", np.nan),
                "Result_Source": "Original saved test comparison"
})


else:
    print("\nWARNING: model_comparison.csv not found.")


# ============================================================
# 3. BUILD FINAL COMPARISON TABLE
# ============================================================

final_rows = comparison_results.copy()

for result in prediction_results:

    row = {
        "Model": result["model"],
        "Result_Source": "Saved test predictions"
    }

    row.update(result["metrics"])
    final_rows.append(row)

if not final_rows:
    raise RuntimeError("No model results were found.")

final_df = pd.DataFrame(final_rows)

metric_columns = [
    "MAE",
    "RMSE",
    "R2",
    "Weighted_MAE",
    "Weighted_RMSE",
    "Weighted_R2"
]

for col in metric_columns:
    if col not in final_df.columns:
        final_df[col] = np.nan

final_df = final_df[
    ["Model"] + metric_columns + ["Result_Source"]
]

final_path = os.path.join(
    OUTPUT_DIR,
    "final_model_comparison.csv"
)

final_df.to_csv(final_path, index=False)

print("\n========== FINAL MODEL COMPARISON ==========")
print(final_df.round(4).to_string(index=False))


# ============================================================
# 4. MODEL PERFORMANCE COMPARISON CHART
# ============================================================

plot_df = final_df.dropna(subset=["R2"]).copy()

if not plot_df.empty:

    plt.figure(figsize=(11, 6))

    plt.bar(plot_df["Model"], plot_df["R2"])

    plt.title("Model Comparison - R2 Score")
    plt.xlabel("Machine Learning Models")
    plt.ylabel("R2 Score")

    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "model_r2_comparison.png"),
        dpi=300
    )

    plt.close()


# ============================================================
# 5. ACTUAL VS PREDICTED EXPENDITURE
# ============================================================

for result in prediction_results:

    actual = result["actual"]
    predicted = result["predicted"]
    model_name = result["model"]

    upper_limit = np.percentile(
        np.concatenate([actual, predicted]), 99
    )

    plt.figure(figsize=(8, 7))

    plt.scatter(
        actual,
        predicted,
        alpha=0.25,
        s=12
    )

    plt.plot(
        [0, upper_limit],
        [0, upper_limit],
        linestyle="--",
        linewidth=2
    )

    plt.xlim(0, upper_limit)
    plt.ylim(0, upper_limit)

    plt.title(f"Actual vs Predicted Expenditure - {model_name}")
    plt.xlabel("Actual Expenditure (INR)")
    plt.ylabel("Predicted Expenditure (INR)")

    plt.grid(alpha=0.25)
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_actual_vs_predicted.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 6. RESIDUAL ANALYSIS
# ============================================================

for result in prediction_results:

    actual = result["actual"]
    predicted = result["predicted"]

    residuals = actual - predicted

    safe_name = result["model"].lower().replace(" ", "_")

    # Residual distribution
    plt.figure(figsize=(9, 6))

    plt.hist(
        residuals,
        bins=50,
        edgecolor="black"
    )

    plt.axvline(0, linestyle="--", linewidth=2)

    plt.title(f"Residual Distribution - {result['model']}")
    plt.xlabel("Residual (Actual - Predicted) INR")
    plt.ylabel("Frequency")

    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_residual_distribution.png"
        ),
        dpi=300
    )

    plt.close()

    # Residual versus predicted
    plt.figure(figsize=(9, 6))

    plt.scatter(
        predicted,
        residuals,
        alpha=0.25,
        s=12
    )

    plt.axhline(0, linestyle="--", linewidth=2)

    plt.title(f"Residuals vs Predicted - {result['model']}")
    plt.xlabel("Predicted Expenditure (INR)")
    plt.ylabel("Residual (INR)")

    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_residuals_vs_predicted.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 7. CATBOOST FEATURE IMPORTANCE
# ============================================================

importance_path = os.path.join(
    MODEL_DIR,
    "catboost_feature_importance.csv"
)

if os.path.exists(importance_path):

    importance_df = pd.read_csv(importance_path)

    feature_col = find_column(
        importance_df,
        ["Feature", "feature"]
    )

    importance_col = find_column(
        importance_df,
        ["Importance", "importance", "Feature_Importance"]
    )

    if feature_col is not None and importance_col is not None:

        importance_df[importance_col] = pd.to_numeric(
            importance_df[importance_col],
            errors="coerce"
        )

        importance_df = importance_df.dropna(
            subset=[importance_col]
        ).sort_values(
            importance_col,
            ascending=True
        )

        plt.figure(figsize=(10, 7))

        plt.barh(
            importance_df[feature_col],
            importance_df[importance_col]
        )

        plt.title("CatBoost Feature Importance")
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")

        plt.grid(axis="x", alpha=0.25)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                "catboost_feature_importance.png"
            ),
            dpi=300
        )

        plt.close()

        importance_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "catboost_feature_importance_sorted.csv"
            ),
            index=False
        )

else:
    print("\nWARNING: CatBoost feature importance file not found.")


# ============================================================
# 8. SAVE EVALUATION SUMMARY
# ============================================================

summary = []

for result in prediction_results:

    summary.append({
        "Model": result["model"],
        "Test_Records": len(result["actual"]),
        "Actual_Mean": np.mean(result["actual"]),
        "Predicted_Mean": np.mean(result["predicted"]),
        "Actual_Median": np.median(result["actual"]),
        "Predicted_Median": np.median(result["predicted"]),
        "Mean_Residual": np.mean(
            result["actual"] - result["predicted"]
        )
    })

if summary:

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "prediction_summary.csv"
        ),
        index=False
    )


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("FINAL MODEL EVALUATION COMPLETED")
print("=" * 65)

print("\nAll evaluation outputs saved in:")
print(OUTPUT_DIR)

print("\nGenerated files:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)
