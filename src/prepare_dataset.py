
import pandas as pd
import os

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "DTES25_26_CSV")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading CSV files...")

# Load overnight trip and expenditure files
trips = pd.read_csv(os.path.join(DATA_DIR, "l4_rev.csv"))
expenses = pd.read_csv(os.path.join(DATA_DIR, "l6_rev.csv"))

# Common household and trip identifiers
keys = [
    "fsu_srno",
    "second_stage_stratum_no",
    "sample_hhld_no",
    "srl_no_trip"
]

# Remove summary rows (trip serial number 99)
trips = trips[trips["srl_no_trip"] != 99].copy()

# Find the most frequent non-null value
def get_mode(series):
    values = series.mode()
    return values.iloc[0] if not values.empty else None

# Aggregate member-level records into one row per trip
trip_data = trips.groupby(keys).agg(
    household_members=("no_of_hhd_members_trip", "first"),
    observed_trip_members=("trip_member_serial_no", "nunique"),
    average_member_age=("age", "mean"),
    trip_type=("type_of_trip", get_mode),
    travel_mode=("major_mode_of_travel", get_mode),
    accommodation_type=("major_type_of_stay", get_mode),
    trip_duration_nights=("nights_spent_outside_upr", "median"),
    leading_purpose=("leading_purpose_of_trip", get_mode),
    trip_month=("trip_starting_month", get_mode),
    main_destination=("main_destination", get_mode),
    destination_state=("dest_state_port_departure", get_mode),
    state=("state", "first"),
    district=("district", "first"),
    sector=("sector", "first")
).reset_index()

# Keep expenditure target and survey weight
expense_data = expenses[
    keys + ["total_3_10", "mult"]
].copy()

# Merge trip information with actual expenditure
dataset = pd.merge(
    trip_data,
    expense_data,
    on=keys,
    how="inner",
    validate="one_to_one"
)

# Rename target
dataset.rename(
    columns={"total_3_10": "total_trip_expenditure"},
    inplace=True
)

# Basic checks
print("\n========== DATASET CHECK ==========")
print("Final shape:", dataset.shape)
print("Missing values:", dataset.isnull().sum().sum())
print("Duplicate rows:", dataset.duplicated().sum())
print("Duplicate trip IDs:", dataset.duplicated(keys).sum())

print("\nTarget summary:")
print(dataset["total_trip_expenditure"].describe())

# Save dataset
output_path = os.path.join(
    OUTPUT_DIR,
    "overnight_tourism_expenditure.csv"
)

dataset.to_csv(output_path, index=False)

print("\nDataset saved successfully!")
print(output_path)
