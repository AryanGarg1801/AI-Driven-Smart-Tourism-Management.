
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR, "outputs", "overnight_tourism_eda_ready.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR, "outputs", "overnight_tourism_decoded.csv"
)

df = pd.read_csv(INPUT_FILE)

# 1. Trip type
trip_type_map = {
    1: "Package",
    2: "Non-package"
}

# 2. Mode of travel
travel_mode_map = {
    1: "On foot",
    2: "Bus",
    3: "Train",
    4: "Ship/Boat",
    5: "Air",
    6: "Own motorised transport",
    7: "Own non-motorised transport",
    8: "Hired motorised transport",
    10: "Hired non-motorised transport",
    19: "Others"
}

# 3. Type of accommodation
accommodation_map = {
    1: "Hotel",
    2: "Guest house",
    3: "Dharamshala",
    4: "Rented house",
    5: "Homestay",
    9: "Other commercial accommodation",
    11: "Friends and relatives",
    19: "Other non-commercial accommodation"
}

# 4. Leading purpose of trip (Block 5.1)
purpose_map = {
    2: "Holiday, leisure and recreation",
    4: "Pilgrimage and religious activities",
    6: "Health and medical",
    7: "Shopping"
}

# 5. Month
month_map = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

# 6. Main destination
destination_map = {
    1: "Within the district",
    2: "Outside district but within state",
    3: "Outside state but within India",
    4: "International trip departure port in India"
}

# 7. Destination state
state_map = {
    1: "Jammu and Kashmir",
    2: "Himachal Pradesh",
    3: "Punjab",
    4: "Chandigarh",
    5: "Uttarakhand",
    6: "Haryana",
    7: "Delhi",
    8: "Rajasthan",
    9: "Uttar Pradesh",
    10: "Bihar",
    11: "Sikkim",
    12: "Arunachal Pradesh",
    13: "Nagaland",
    14: "Manipur",
    15: "Mizoram",
    16: "Tripura",
    17: "Meghalaya",
    18: "Assam",
    19: "West Bengal",
    20: "Jharkhand",
    21: "Odisha",
    22: "Chhattisgarh",
    23: "Madhya Pradesh",
    24: "Gujarat",
    25: "Dadra and Nagar Haveli and Daman and Diu",
    27: "Maharashtra",
    28: "Andhra Pradesh",
    29: "Karnataka",
    30: "Goa",
    31: "Lakshadweep",
    32: "Kerala",
    33: "Tamil Nadu",
    34: "Puducherry",
    35: "Andaman and Nicobar Islands",
    36: "Telangana",
    37: "Ladakh"
}

# 8. Sector
sector_map = {
    1: "Rural",
    2: "Urban"
}

# Create new label columns without changing original codes
mappings = {
    "trip_type": trip_type_map,
    "travel_mode": travel_mode_map,
    "accommodation_type": accommodation_map,
    "leading_purpose": purpose_map,
    "trip_month": month_map,
    "main_destination": destination_map,
    "destination_state": state_map,
    "sector": sector_map
}

for column, mapping in mappings.items():
    df[column + "_label"] = pd.to_numeric(
        df[column], errors="coerce"
    ).map(mapping)

    # Keep structural missing values understandable
    if column == "destination_state":
        df.loc[
            df[column] == "Not applicable",
            column + "_label"
        ] = "Not applicable"

    # Report any codes without a mapping
    unmapped = df.loc[
        df[column + "_label"].isna(), column
    ].dropna().unique()

    if len(unmapped) > 0:
        print(f"Unmapped values in {column}:", unmapped)

# Save decoded dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\n========== DECODING COMPLETED ==========")
print("Dataset shape:", df.shape)
print("Total columns:", len(df.columns))
print("Missing values:", df.isnull().sum().sum())
print("\nDecoded columns:")

for column in mappings:
    print(f"\n{column}:")
    print(df[column + "_label"].value_counts(dropna=False))

print("\nSaved decoded dataset:")
print(OUTPUT_FILE)
