
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR, "outputs", "overnight_tourism_expenditure.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR, "outputs", "overnight_tourism_eda_ready.csv"
)

df = pd.read_csv(INPUT_FILE)

# Create final survey weight
df["survey_weight"] = df["mult"] / 100

# Mark destination state as not applicable
df["destination_state"] = (
    df["destination_state"]
    .fillna("Not applicable")
    .astype(str)
)

# Save prepared dataset
df.to_csv(OUTPUT_FILE, index=False)

print("========== EDA DATASET READY ==========")
print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

print("\nTarget expenditure:")
print(df["total_trip_expenditure"].describe())

print("\nSurvey weight:")
print(df["survey_weight"].describe())

print("\nSaved successfully:")
print(OUTPUT_FILE)
