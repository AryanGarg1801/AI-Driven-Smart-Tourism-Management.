
# src/finalize_segmentation.py

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR, "outputs", "overnight_tourism_decoded.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "segmentation")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
SILHOUETTE_SAMPLE_SIZE = 2500

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
    "sector_label"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def weighted_average(values, weights):
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")

    valid = values.notna() & weights.notna() & (weights > 0)

    if valid.sum() == 0:
        return np.nan

    return np.average(values[valid], weights=weights[valid])


def weighted_mode(series, weights):
    temp = pd.DataFrame({
        "value": series,
        "weight": weights
    })

    temp = temp.dropna()

    if temp.empty:
        return "Unknown"

    totals = temp.groupby("value")["weight"].sum()

    return totals.idxmax()


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("FINAL TOURIST BEHAVIOURAL SEGMENTATION")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:", df.shape)

required_columns = FEATURES + [
    "total_trip_expenditure",
    "mult"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Survey weights
df["survey_weight"] = df["mult"] / 100

# Clean feature columns
for col in NUMERIC_FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

for col in CATEGORICAL_FEATURES:
    df[col] = df[col].fillna("Unknown").astype(str)

# Remove invalid survey weights only from profile calculations.
# They are not clustering features.
df["survey_weight"] = (
    df["survey_weight"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
    .clip(lower=0)
)

print("Records:", len(df))
print("Unique households:", df[
    ["fsu_srno", "second_stage_stratum_no", "sample_hhld_no"]
].drop_duplicates().shape[0])

# ============================================================
# FEATURE PREPROCESSING
# ============================================================

print("\nPreparing behavioural features...")

try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )
except TypeError:
    # Compatibility with older scikit-learn versions
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=True
    )

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            encoder,
            CATEGORICAL_FEATURES
        )
    ]
)

X = preprocessor.fit_transform(df[FEATURES])

print("Processed feature shape:", X.shape)

# IMPORTANT:
# Expenditure is deliberately NOT included in X.
# It is used only later for understanding cluster profiles.

# ============================================================
# EVALUATE DIFFERENT CLUSTER COUNTS
# ============================================================

print("\n========== CLUSTER EVALUATION ==========")

evaluation_results = []

best_score = -1
best_k = None
best_model = None
best_labels = None

for k in range(2, 8):

    print(f"\nTraining MiniBatch K-Means: K = {k}")

    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        batch_size=2048,
        n_init=10
    )

    labels = model.fit_predict(X)

    score = silhouette_score(
        X,
        labels,
        sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(df)),
        random_state=RANDOM_STATE
    )

    inertia = model.inertia_

    print("Inertia:", round(inertia, 2))
    print("Silhouette Score:", round(score, 4))

    evaluation_results.append({
        "K": k,
        "Inertia": inertia,
        "Silhouette_Score": score
    })

    if score > best_score:
        best_score = score
        best_k = k
        best_model = model
        best_labels = labels

# Save evaluation results
evaluation_df = pd.DataFrame(evaluation_results)

evaluation_df.to_csv(
    os.path.join(OUTPUT_DIR, "cluster_evaluation.csv"),
    index=False
)

print("\n========== SELECTED MODEL ==========")
print("Best K:", best_k)
print("Best Silhouette Score:", round(best_score, 4))

# ============================================================
# ASSIGN FINAL CLUSTERS
# ============================================================

df["tourist_cluster"] = best_labels

cluster_counts = df["tourist_cluster"].value_counts().sort_index()

print("\n========== CLUSTER DISTRIBUTION ==========")

for cluster, count in cluster_counts.items():

    percentage = (count / len(df)) * 100

    print(
        f"Cluster {cluster}: "
        f"{count:,} records ({percentage:.2f}%)"
    )

# ============================================================
# CREATE CLUSTER PROFILES
# ============================================================

print("\n========== BUILDING CLUSTER PROFILES ==========")

profiles = []

for cluster in sorted(df["tourist_cluster"].unique()):

    group = df[df["tourist_cluster"] == cluster]

    weights = group["survey_weight"]

    profile = {
        "Cluster": cluster,
        "Record_Count": len(group),
        "Percentage": len(group) / len(df) * 100,

        "Weighted_Average_Expenditure": weighted_average(
            group["total_trip_expenditure"],
            weights
        ),

        "Weighted_Average_Trip_Duration": weighted_average(
            group["trip_duration_nights"],
            weights
        ),

        "Weighted_Average_Household_Size": weighted_average(
            group["household_members"],
            weights
        ),

        "Weighted_Average_Age": weighted_average(
            group["average_member_age"],
            weights
        )
    }

    # Dominant travel behaviour
    for col in CATEGORICAL_FEATURES:

        profile["Dominant_" + col] = weighted_mode(
            group[col],
            weights
        )

    profiles.append(profile)

profiles_df = pd.DataFrame(profiles)

profiles_df.to_csv(
    os.path.join(OUTPUT_DIR, "cluster_profiles.csv"),
    index=False
)

print("\nCluster Profiles:")
print(profiles_df.to_string(index=False))

# ============================================================
# SAVE CLUSTERED DATASET
# ============================================================

cluster_output = os.path.join(
    OUTPUT_DIR,
    "tourist_segmentation_results.csv"
)

df.to_csv(cluster_output, index=False)

print("\nClustered dataset saved.")

# ============================================================
# SAVE TRAINED MODEL AND PREPROCESSOR
# ============================================================

model_artifact = {
    "preprocessor": preprocessor,
    "model": best_model,
    "features": FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "best_k": best_k,
    "silhouette_score": best_score
}

model_path = os.path.join(
    MODEL_DIR,
    "tourist_segmentation_model.pkl"
)

joblib.dump(model_artifact, model_path)

print("Segmentation model saved.")

# ============================================================
# VISUALIZATION 1: SILHOUETTE SCORES
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    evaluation_df["K"],
    evaluation_df["Silhouette_Score"],
    marker="o"
)

plt.title("Silhouette Score for Different Cluster Counts")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(range(2, 8))
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "silhouette_scores.png"),
    dpi=300
)

plt.close()

# ============================================================
# VISUALIZATION 2: CLUSTER DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

cluster_counts.sort_index().plot(
    kind="bar"
)

plt.title("Tourist Cluster Distribution")
plt.xlabel("Tourist Cluster")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "cluster_distribution.png"),
    dpi=300
)

plt.close()

# ============================================================
# VISUALIZATION 3: EXPENDITURE PROFILE
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    profiles_df["Cluster"].astype(str),
    profiles_df["Weighted_Average_Expenditure"]
)

plt.title("Weighted Average Expenditure by Tourist Cluster")
plt.xlabel("Tourist Cluster")
plt.ylabel("Average Expenditure (INR)")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "cluster_expenditure_profile.png"),
    dpi=300
)

plt.close()

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("TOURIST SEGMENTATION COMPLETED SUCCESSFULLY")
print("=" * 65)

print("Selected clusters:", best_k)
print("Silhouette score:", round(best_score, 4))

print("\nGenerated files:")
print("1. outputs/segmentation/cluster_evaluation.csv")
print("2. outputs/segmentation/cluster_profiles.csv")
print("3. outputs/segmentation/tourist_segmentation_results.csv")
print("4. outputs/segmentation/silhouette_scores.png")
print("5. outputs/segmentation/cluster_distribution.png")
print("6. outputs/segmentation/cluster_expenditure_profile.png")
print("7. outputs/models/tourist_segmentation_model.pkl")

print("\nNote: Expenditure was excluded from clustering.")
print("It was used only to describe the resulting clusters.")
