
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR, "outputs", "overnight_tourism_decoded.csv"
)

df = pd.read_csv(INPUT_FILE)

target = "total_trip_expenditure"

print("========== DATA VALIDATION ==========")

# 1. Basic checks
print("\nDataset shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("Missing values:", df.isnull().sum().sum())

# 2. Expenditure analysis
print("\n========== EXPENDITURE CHECK ==========")
print(df[target].describe())

print("\nZero or negative expenditure:")
print((df[target] <= 0).sum())

print("\nExpenditure below Rs. 100:")
print((df[target] < 100).sum())

# 3. Lowest expenditure records
print("\n========== 20 LOWEST EXPENDITURE RECORDS ==========")

print(
    df.nsmallest(20, target)[
        ["leading_purpose_label",
         "travel_mode_label",
         "accommodation_type_label",
         "trip_duration_nights",
         target]
    ].to_string(index=False)
)

# 4. Highest expenditure records
print("\n========== 20 HIGHEST EXPENDITURE RECORDS ==========")

print(
    df.nlargest(20, target)[
        ["leading_purpose_label",
         "travel_mode_label",
         "accommodation_type_label",
         "trip_duration_nights",
         target]
    ].to_string(index=False)
)

# 5. Outlier analysis using IQR
Q1 = df[target].quantile(0.25)
Q3 = df[target].quantile(0.75)
IQR = Q3 - Q1

upper_limit = Q3 + (1.5 * IQR)

outliers = df[df[target] > upper_limit]

print("\n========== OUTLIER ANALYSIS ==========")
print("Upper limit:", upper_limit)
print("Outlier count:", len(outliers))
print("Outlier percentage:", round(
    len(outliers) / len(df) * 100, 2
), "%")

# 6. Trip member validation
print("\n========== TRIP MEMBER VALIDATION ==========")

print("Household members on trip:")
print(df["household_members"].describe())

print("\nObserved trip members:")
print(df["observed_trip_members"].describe())

print("\nRecords where observed members exceed household trip count:")
print(
    (df["observed_trip_members"] > df["household_members"]).sum()
)

# 7. Trip duration
print("\n========== TRIP DURATION ==========")
print(df["trip_duration_nights"].describe())

# 8. Survey weights
print("\n========== SURVEY WEIGHT CHECK ==========")
print(df["survey_weight"].describe())
print("Missing survey weights:", df["survey_weight"].isna().sum())

print("\n========== VALIDATION COMPLETED ==========")
