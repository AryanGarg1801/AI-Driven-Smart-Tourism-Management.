
import pandas as pd
import os

# Project directory
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_path = os.path.join(
    base_path,
    "outputs",
    "overnight_tourism_decoded.csv"
)

output_path = os.path.join(
    base_path,
    "outputs",
    "model_ready_features.csv"
)

# Load dataset
df = pd.read_csv(input_path)

print("========== MODEL DATA PREPARATION ==========")

# Features selected for prediction
features = [
    "household_members",
    "average_member_age",
    "trip_type_label",
    "travel_mode_label",
    "accommodation_type_label",
    "trip_duration_nights",
    "leading_purpose_label",
    "trip_month_label",
    "main_destination_label",
    "destination_state_label",
    "state",
    "district",
    "sector_label"
]

# Target variable
target = "total_trip_expenditure"

# Household grouping key for a later leakage-safe train/test split
group_columns = [
    "fsu_srno",
    "second_stage_stratum_no",
    "sample_hhld_no"
]

# Keep only selected features, target and grouping columns
model_df = df[features + [target] + group_columns].copy()

# Check missing values
print("\nMissing values:")
print(model_df.isnull().sum())

# Check whether household_members and observed_trip_members
# are identical before deciding whether to retain one
print("\nHousehold member comparison:")
print(
    (df["household_members"] == df["observed_trip_members"]).all()
)

# Create a household group identifier
model_df["household_group"] = (
    model_df[group_columns].astype(str).agg("_".join, axis=1)
)

# Remove individual grouping columns from model dataset
model_df.drop(columns=group_columns, inplace=True)

# Save prepared dataset
model_df.to_csv(output_path, index=False)

print("\n========== FINAL MODEL DATA ==========")
print("Shape:", model_df.shape)

print("\nSelected features:")
print(features)

print("\nTarget:")
print(target)

print("\nUnique households:", model_df["household_group"].nunique())

print("\nSaved successfully:")
print(output_path)

print("\n========== PREPARATION COMPLETED ==========")
