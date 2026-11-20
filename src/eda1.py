
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------------
# PATHS
# ----------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR, "outputs", "overnight_tourism_eda_ready.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

target = "total_trip_expenditure"
weight = "survey_weight"

print("========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# ----------------------------------
# 1. TARGET STATISTICS
# ----------------------------------

print("\n========== EXPENDITURE STATISTICS ==========")
print(df[target].describe())

weighted_mean = np.average(
    df[target],
    weights=df[weight]
)

print("\nWeighted average expenditure (INR):", round(weighted_mean, 2))
print("Median expenditure (INR):", df[target].median())
# ----------------------------------
# 2. EXPENDITURE DISTRIBUTION
# ----------------------------------

plt.figure(figsize=(10, 6))
plt.hist(df[target], bins=60)
plt.title("Distribution of Total Trip Expenditure")
plt.xlabel("Total Trip Expenditure (₹)")
plt.ylabel("Number of Trips")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "expenditure_distribution.png"))
plt.show()

# Log transformation for visualization only
plt.figure(figsize=(10, 6))
plt.hist(np.log1p(df[target]), bins=60)
plt.title("Log Distribution of Trip Expenditure")
plt.xlabel("Log of Total Trip Expenditure")
plt.ylabel("Number of Trips")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "expenditure_log_distribution.png"))
plt.show()

# ----------------------------------
# 3. OUTLIER ANALYSIS
# ----------------------------------

Q1 = df[target].quantile(0.25)
Q3 = df[target].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df[target] < lower_limit) |
    (df[target] > upper_limit)
]

print("\n========== OUTLIER ANALYSIS ==========")
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)
print("Number of outliers:", len(outliers))
print("Outlier percentage:", round(len(outliers) / len(df) * 100, 2), "%")

plt.figure(figsize=(10, 5))
plt.boxplot(df[target], vert=False)
plt.title("Trip Expenditure Boxplot")
plt.xlabel("Total Trip Expenditure (₹)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "expenditure_boxplot.png"))
plt.show()

# ----------------------------------
# 4. CATEGORICAL FEATURE ANALYSIS
# ----------------------------------

categorical_columns = [
    "trip_type",
    "travel_mode",
    "accommodation_type",
    "leading_purpose",
    "trip_month",
    "main_destination",
    "destination_state",
    "state",
    "sector"
]

for column in categorical_columns:

    counts = df[column].value_counts().head(15)

    print(f"\n========== {column.upper()} ==========")
    print(counts)

    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Number of Trips")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{column}_distribution.png")
    )
    plt.show()

# ----------------------------------
# 5. NUMERICAL FEATURE ANALYSIS
# ----------------------------------

numerical_columns = [
    "household_members",
    "observed_trip_members",
    "average_member_age",
    "trip_duration_nights",
    target
]

print("\n========== NUMERICAL SUMMARY ==========")
print(df[numerical_columns].describe())

# ----------------------------------
# 6. CORRELATION ANALYSIS
# ----------------------------------

correlation = df[numerical_columns].corr()

print("\n========== CORRELATION MATRIX ==========")
print(correlation)

correlation.to_csv(
    os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
)

plt.figure(figsize=(9, 7))
plt.imshow(correlation, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.title("Numerical Feature Correlation")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.show()

# ----------------------------------
# 7. WEIGHTED EXPENDITURE BY TRIP PURPOSE
# ----------------------------------

weighted_purpose = df.groupby("leading_purpose").apply(
    lambda x: np.average(
        x[target],
        weights=x[weight]
    ),
    include_groups=False
)

print("\n========== WEIGHTED EXPENDITURE BY PURPOSE ==========")
print(weighted_purpose)

weighted_purpose.to_csv(
    os.path.join(OUTPUT_DIR, "weighted_expenditure_by_purpose.csv")
)

print("\n========== EDA COMPLETED ==========")
print("All EDA outputs saved in:", OUTPUT_DIR)
