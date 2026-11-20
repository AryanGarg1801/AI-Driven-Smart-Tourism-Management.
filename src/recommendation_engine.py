
import os
import re
import warnings
import difflib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# ============================================================
# PERSONALIZED SMART TOURISM RECOMMENDATION SYSTEM
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HISTORY_PATH = os.path.join(
    BASE_DIR, "outputs", "overnight_tourism_decoded.csv"
)

ATTRACTION_PATH = os.path.join(
    BASE_DIR, "dataset", "india_tourist_destinations.csv"
)

N_NEIGHBORS = 2000
TOP_RECOMMENDATIONS = 10

PURPOSES = [
    "Holiday, leisure and recreation",
    "Pilgrimage and religious activities",
    "Health and medical",
    "Shopping"
]

TRAVEL_MODES = [
    "On foot",
    "Bus",
    "Train",
    "Ship/Boat",
    "Air",
    "Own motorised transport",
    "Own non-motorised transport",
    "Hired motorised transport",
    "Hired non-motorised transport",
    "Others"
]

ACCOMMODATIONS = [
    "Hotel",
    "Guest house",
    "Dharamshala",
    "Rented house",
    "Homestay",
    "Other commercial accommodation",
    "Friends and relatives",
    "Other non-commercial accommodation"
]

DESTINATION_TYPES = [
    "Within the district",
    "Outside district but within state",
    "Outside state but within India",
    "International trip departure port in India"
]

INDIAN_REGIONS = {
    "North India": [
        "Jammu and Kashmir", "Himachal Pradesh", "Punjab",
        "Chandigarh", "Uttarakhand", "Haryana", "Delhi",
        "Rajasthan", "Uttar Pradesh", "Ladakh"
    ],
    "South India": [
        "Andhra Pradesh", "Karnataka", "Kerala",
        "Tamil Nadu", "Telangana", "Puducherry"
    ],
    "East India": [
        "Bihar", "Jharkhand", "Odisha", "West Bengal"
    ],
    "West India": [
        "Goa", "Gujarat", "Maharashtra",
        "Dadra and Nagar Haveli and Daman and Diu"
    ],
    "Central India": [
        "Madhya Pradesh", "Chhattisgarh"
    ],
    "North East India": [
        "Sikkim", "Arunachal Pradesh", "Nagaland",
        "Manipur", "Mizoram", "Tripura",
        "Meghalaya", "Assam"
    ],
    "Island Territories": [
        "Lakshadweep", "Andaman and Nicobar Islands"
    ]
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize(value):
    """Normalize text for reliable comparisons."""
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9 ]", " ", value)
    return " ".join(value.split())


def get_user_choice(prompt, options):
    """Get a valid choice with fuzzy matching."""

    print("\n" + prompt)

    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")

    while True:
        value = input("\nEnter choice number or name: ").strip()

        if value.isdigit():
            index = int(value)

            if 1 <= index <= len(options):
                return options[index - 1]

        normalized_options = {
            normalize(option): option for option in options
        }

        if normalize(value) in normalized_options:
            return normalized_options[normalize(value)]

        matches = difflib.get_close_matches(
            normalize(value),
            list(normalized_options.keys()),
            n=1,
            cutoff=0.55
        )

        if matches:
            selected = normalized_options[matches[0]]
            print(f"Matched input to: {selected}")
            return selected

        print("Invalid choice. Please try again.")


def get_positive_number(prompt, allow_zero=False):
    """Read a valid non-negative or positive number."""

    while True:
        try:
            value = float(input(prompt).strip())

            if np.isfinite(value) and (
                value >= 0 if allow_zero else value > 0
            ):
                return value

            print("Please enter a valid positive number.")

        except ValueError:
            print("Please enter a numeric value.")


def optional_catalogue_choice(prompt, options):
    """Optional selection with Any support."""

    print(f"\n{prompt}")
    print("Press Enter for Any.")

    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")

    while True:
        value = input("\nEnter choice number or name: ").strip()

        if value == "":
            return "Any"

        if value.lower() == "any":
            return "Any"

        if value.isdigit():
            index = int(value)

            if 1 <= index <= len(options):
                return options[index - 1]

        option_map = {
            normalize(option): option for option in options
        }

        if normalize(value) in option_map:
            return option_map[normalize(value)]

        matches = difflib.get_close_matches(
            normalize(value),
            list(option_map.keys()),
            n=1,
            cutoff=0.55
        )

        if matches:
            selected = option_map[matches[0]]
            print(f"Matched input to: {selected}")
            return selected

        print("Invalid selection. Please try again.")


def parse_suitable_purpose(value):
    """Split semicolon-separated purpose values."""

    if pd.isna(value):
        return []

    return [
        normalize(item)
        for item in str(value).split(";")
        if normalize(item)
    ]


def calculate_category_priority(purpose, category):
    """Assign relevance scores based on travel purpose."""

    purpose = normalize(purpose)
    category = normalize(category)

    priorities = {
        "pilgrimage and religious activities": {
            "spiritual": 5,
            "heritage": 4,
            "nature": 2,
            "recreation": 1
        },

        "holiday leisure and recreation": {
            "nature": 5,
            "adventure": 5,
            "beach": 5,
            "wildlife": 4,
            "recreation": 4,
            "heritage": 3,
            "rural": 2,
            "spiritual": 1
        },

        "health and medical": {
            "nature": 4,
            "recreation": 3,
            "spiritual": 2,
            "heritage": 1
        },

        "shopping": {
            "recreation": 4,
            "heritage": 3,
            "rural": 2,
            "nature": 1
        }
    }

    purpose_priorities = priorities.get(purpose, {})

    return purpose_priorities.get(category, 1)


# ============================================================
# 1. LOAD DATASETS
# ============================================================

print("=" * 65)
print("PERSONALIZED SMART TOURISM RECOMMENDATION SYSTEM")
print("=" * 65)

if not os.path.exists(HISTORY_PATH):
    raise FileNotFoundError(
        "Historical tourism dataset not found: " + HISTORY_PATH
    )

if not os.path.exists(ATTRACTION_PATH):
    raise FileNotFoundError(
        "Tourist attraction catalogue not found: " + ATTRACTION_PATH
    )

history = pd.read_csv(HISTORY_PATH)
attractions = pd.read_csv(ATTRACTION_PATH)

print("\nHistorical trips:", len(history))
print("Tourist attractions:", len(attractions))
print(
    "States and UTs:",
    attractions["state_or_ut"].nunique()
)


# ============================================================
# 2. VALIDATE REQUIRED COLUMNS
# ============================================================

required_history = [
    "household_members",
    "average_member_age",
    "trip_duration_nights",
    "leading_purpose_label",
    "travel_mode_label",
    "accommodation_type_label",
    "main_destination_label",
    "destination_state_label",
    "total_trip_expenditure",
    "mult",
    "fsu_srno",
    "second_stage_stratum_no",
    "sample_hhld_no"
]

required_attractions = [
    "destination_name",
    "attraction_name",
    "city_or_area",
    "state_or_ut",
    "category",
    "suitable_purpose",
    "description",
    "source_url"
]

missing_history = [
    col for col in required_history
    if col not in history.columns
]

missing_attractions = [
    col for col in required_attractions
    if col not in attractions.columns
]

if missing_history:
    raise ValueError(
        f"Missing historical columns: {missing_history}"
    )

if missing_attractions:
    raise ValueError(
        f"Missing attraction columns: {missing_attractions}"
    )


# ============================================================
# 3. CLEAN DATA
# ============================================================

NUMERIC_FEATURES = [
    "household_members",
    "average_member_age",
    "trip_duration_nights"
]

CATEGORICAL_FEATURES = [
    "leading_purpose_label",
    "travel_mode_label",
    "accommodation_type_label",
    "main_destination_label"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

GROUP_COLUMNS = [
    "fsu_srno",
    "second_stage_stratum_no",
    "sample_hhld_no"
]

for col in NUMERIC_FEATURES + [
    "total_trip_expenditure", "mult"
]:
    history[col] = pd.to_numeric(
        history[col], errors="coerce"
    )

history = history.dropna(
    subset=GROUP_COLUMNS + [
        "total_trip_expenditure", "mult"
    ]
).copy()

history = history[
    (history["total_trip_expenditure"] > 0) &
    (history["mult"] > 0)
].copy()

history = history.dropna(
    subset=NUMERIC_FEATURES
).copy()

history = history.reset_index(drop=True)

for col in CATEGORICAL_FEATURES:
    history[col] = history[col].fillna(
        "Unknown"
    ).astype(str)

history["survey_weight"] = history["mult"] / 100

for col in required_attractions:
    if col != "source_url":
        attractions[col] = attractions[col].fillna("")

attractions = attractions.drop_duplicates(
    subset=["destination_name", "attraction_name"]
).reset_index(drop=True)

print("\nClean historical trips:", len(history))
print("Clean attractions:", len(attractions))


# ============================================================
# 4. GET USER TRAVEL PREFERENCES
# ============================================================

print("\n" + "=" * 65)
print("ENTER TRAVEL PREFERENCES")
print("=" * 65)

purpose = get_user_choice(
    "Select travel purpose:",
    PURPOSES
)

budget = get_positive_number(
    "\nEnter total trip budget in INR: "
)

duration = get_positive_number(
    "Enter trip duration in nights: ",
    allow_zero=True
)

travellers = get_positive_number(
    "Enter number of travellers: "
)

travellers = int(travellers)

travel_mode = get_user_choice(
    "Select preferred travel mode:",
    TRAVEL_MODES
)

accommodation = get_user_choice(
    "Select preferred accommodation:",
    ACCOMMODATIONS
)

destination_type = get_user_choice(
    "Select preferred destination type:",
    DESTINATION_TYPES
)


# ============================================================
# 5. GET DESTINATION AND CATEGORY PREFERENCES
# ============================================================

print("\n" + "=" * 65)
print("DESTINATION PERSONALIZATION")
print("=" * 65)

available_states = sorted(
    attractions["state_or_ut"].dropna().unique().tolist()
)

available_categories = sorted(
    attractions["category"].dropna().unique().tolist()
)

available_regions = list(INDIAN_REGIONS.keys())

preferred_state = optional_catalogue_choice(
    "Preferred state or UT:",
    available_states
)

preferred_region = optional_catalogue_choice(
    "Preferred region:",
    available_regions
)

preferred_category = optional_catalogue_choice(
    "Preferred attraction category:",
    available_categories
)

if preferred_state != "Any":
    preferred_region = "Any"

print("\n========== PREFERENCE SUMMARY ==========")
print("Purpose:", purpose)
print("Budget: INR", budget)
print("Duration:", duration, "nights")
print("Travellers:", travellers)
print("Travel mode:", travel_mode)
print("Accommodation:", accommodation)
print("Destination type:", destination_type)
print("Preferred state:", preferred_state)
print("Preferred region:", preferred_region)
print("Preferred category:", preferred_category)


# ============================================================
# 6. PREPARE HISTORICAL KNN MODEL
# ============================================================

print("\nPreparing historical recommendation model...")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            CATEGORICAL_FEATURES
        )
    ]
)

X_history = history[FEATURES]

X_processed = preprocessor.fit_transform(X_history)

print("Processed feature shape:", X_processed.shape)

knn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)

knn.fit(X_processed)

print("Historical recommendation model prepared.")


# ============================================================
# 7. FIND SIMILAR HISTORICAL TRIPS
# ============================================================

query = pd.DataFrame([{
    "household_members": travellers,
    "average_member_age": history[
        "average_member_age"
    ].median(),
    "trip_duration_nights": duration,
    "leading_purpose_label": purpose,
    "travel_mode_label": travel_mode,
    "accommodation_type_label": accommodation,
    "main_destination_label": destination_type
}])

query = query[FEATURES]

query_processed = preprocessor.transform(query)

number_of_neighbors = min(
    N_NEIGHBORS,
    len(history)
)

distances, indices = knn.kneighbors(
    query_processed,
    n_neighbors=number_of_neighbors
)

similar_trips = history.iloc[
    indices[0]
].copy()

similar_trips["similarity"] = (
    1 - distances[0]
)

similar_trips["similarity"] = (
    similar_trips["similarity"].clip(0, 1)
)

print("\nSimilar historical trips found:", len(similar_trips))


# ============================================================
# 8. APPLY BUDGET FILTER
# ============================================================

budget_matches = similar_trips[
    similar_trips["total_trip_expenditure"] <= budget
].copy()

if len(budget_matches) > 0:

    selected_trips = budget_matches

    print(
        "\nHistorical trips within budget:",
        len(selected_trips)
    )

else:

    selected_trips = similar_trips.copy()

    print("\nWARNING: No similar historical trips were")
    print("found within the entered budget.")

    print(
        "Showing closest historical matches, including "
        "some that may exceed the budget."
    )


# ============================================================
# 9. HISTORICAL TRAVEL PATTERN RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 65)
print("HISTORICAL TRAVEL PATTERN RECOMMENDATIONS")
print("=" * 65)

selected_trips["weighted_support"] = (
    selected_trips["survey_weight"] *
    selected_trips["similarity"]
)

pattern_summary = (
    selected_trips.groupby(
        [
            "travel_mode_label",
            "accommodation_type_label"
        ],
        dropna=False
    )
    .agg(
        historical_trips=(
            "total_trip_expenditure", "size"
        ),
        weighted_support=(
            "weighted_support", "sum"
        ),
        median_expenditure=(
            "total_trip_expenditure", "median"
        ),
        average_duration=(
            "trip_duration_nights", "mean"
        ),
        average_similarity=(
            "similarity", "mean"
        ),
        common_destination=(
            "main_destination_label",
            lambda x: x.mode().iloc[0]
            if not x.mode().empty else "Unknown"
        )
    )
    .reset_index()
)

pattern_summary = pattern_summary.sort_values(
    ["average_similarity", "weighted_support"],
    ascending=False
).head(5)

if not pattern_summary.empty:

    for index, row in pattern_summary.iterrows():

        print(f"\nRecommendation {index + 1}")

        print(
            "Travel mode:",
            row["travel_mode_label"]
        )

        print(
            "Accommodation:",
            row["accommodation_type_label"]
        )

        print(
            "Historical trips:",
            int(row["historical_trips"])
        )

        print(
            "Weighted support:",
            round(row["weighted_support"], 2)
        )

        print(
            "Median historical expenditure: INR",
            round(row["median_expenditure"], 2)
        )

        print(
            "Average trip duration:",
            round(row["average_duration"], 2),
            "nights"
        )

        print(
            "Average similarity:",
            round(row["average_similarity"], 4)
        )

        print(
            "Common destination type:",
            row["common_destination"]
        )

else:
    print("No historical patterns available.")


# ============================================================
# 10. FILTER ATTRACTIONS BY TRAVEL PURPOSE
# ============================================================

print("\n" + "=" * 65)
print("PERSONALIZED TOURIST DESTINATION RECOMMENDATIONS")
print("=" * 65)

purpose_normalized = normalize(purpose)

attractions["purpose_list"] = attractions[
    "suitable_purpose"
].apply(parse_suitable_purpose)

purpose_matches = attractions[
    attractions["purpose_list"].apply(
        lambda values: purpose_normalized in values
    )
].copy()

if purpose_matches.empty:
    print("\nNo attractions match the selected purpose.")
    print("Please check the attraction catalogue.")
    raise SystemExit


# ============================================================
# 11. APPLY STATE OR REGION PREFERENCE
# ============================================================

preferred_attractions = purpose_matches.copy()

if preferred_state != "Any":

    state_matches = preferred_attractions[
        preferred_attractions["state_or_ut"].apply(
            normalize
        ) == normalize(preferred_state)
    ]

    if not state_matches.empty:
        preferred_attractions = state_matches
    else:
        print(
            "\nNo matching attractions in the selected state."
        )
        print("Showing recommendations from all states instead.")

elif preferred_region != "Any":

    region_states = INDIAN_REGIONS[preferred_region]

    region_matches = preferred_attractions[
        preferred_attractions["state_or_ut"].apply(
            lambda x: normalize(x) in {
                normalize(state) for state in region_states
            }
        )
    ]

    if not region_matches.empty:
        preferred_attractions = region_matches
    else:
        print(
            "\nNo matching attractions in the selected region."
        )
        print("Showing recommendations from all regions instead.")


# ============================================================
# 12. PERSONALIZED DESTINATION RANKING
# ============================================================

preferred_attractions["purpose_score"] = (
    preferred_attractions["category"].apply(
        lambda x: calculate_category_priority(
            purpose, x
        )
    )
)

preferred_attractions["category_preference_score"] = (
    preferred_attractions["category"].apply(
        lambda x: (
            5 if preferred_category != "Any"
            and normalize(x) == normalize(preferred_category)
            else 0
        )
    )
)

preferred_attractions["state_preference_score"] = (
    preferred_attractions["state_or_ut"].apply(
        lambda x: (
            5 if preferred_state != "Any"
            and normalize(x) == normalize(preferred_state)
            else 0
        )
    )
)

preferred_attractions["region_preference_score"] = (
    preferred_attractions["state_or_ut"].apply(
        lambda x: (
            3 if preferred_region != "Any"
            and any(
                normalize(x) == normalize(state)
                for state in INDIAN_REGIONS[preferred_region]
            )
            else 0
        )
    )
)

preferred_attractions["relevance_score"] = (
    preferred_attractions["purpose_score"] +
    preferred_attractions["category_preference_score"] +
    preferred_attractions["state_preference_score"] +
    preferred_attractions["region_preference_score"]
)

# Aggregate attraction scores by destination.
destination_ranking = (
    preferred_attractions.groupby(
        ["destination_name", "state_or_ut"],
        dropna=False
    )
    .agg(
        relevance_score=("relevance_score", "max"),
        attraction_count=("attraction_name", "count"),
        categories=(
            "category",
            lambda x: ", ".join(sorted(set(x)))
        )
    )
    .reset_index()
    .sort_values(
        ["relevance_score", "attraction_count", "destination_name"],
        ascending=[False, False, True]
    )
    .reset_index(drop=True)
)

preferred_attractions = preferred_attractions.merge(
    destination_ranking[
        ["destination_name", "state_or_ut", "relevance_score"]
    ],
    on=["destination_name", "state_or_ut"],
    how="left",
    suffixes=("", "_destination")
)

preferred_attractions = preferred_attractions.sort_values(
    ["relevance_score", "destination_name", "attraction_name"],
    ascending=[False, True, True]
)

# Select top destinations and show their attractions.
top_destinations = destination_ranking.head(
    TOP_RECOMMENDATIONS
)

print("\nTOP PERSONALIZED DESTINATIONS")

for rank, (_, destination) in enumerate(
    top_destinations.iterrows(),
    start=1
):

    print("\n" + "-" * 55)

    print(
        f"{rank}. {destination['destination_name']}"
    )

    print("State/UT:", destination["state_or_ut"])

    print(
        "Relevance score:",
        round(destination["relevance_score"], 2)
    )

    print(
        "Available attraction categories:",
        destination["categories"]
    )

    destination_attractions = preferred_attractions[
        (
            preferred_attractions["destination_name"]
            == destination["destination_name"]
        ) &
        (
            preferred_attractions["state_or_ut"]
            == destination["state_or_ut"]
        )
    ]

    for _, attraction in destination_attractions.head(3).iterrows():

        print(
            "  -",
            attraction["attraction_name"],
            "| Category:",
            attraction["category"]
        )

        print(
            "    Area:",
            attraction["city_or_area"]
        )

        if attraction["description"]:
            print(
                "    Description:",
                attraction["description"]
            )

        if attraction["source_url"]:
            print(
                "    Source:",
                attraction["source_url"]
            )


# ============================================================
# 13. SAVE RECOMMENDATION OUTPUTS
# ============================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR, "outputs", "recommendations"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

preferred_attractions.drop(
    columns=["purpose_list"],
    errors="ignore"
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "personalized_attraction_recommendations.csv"
    ),
    index=False
)

destination_ranking.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "destination_ranking.csv"
    ),
    index=False
)

pattern_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "historical_travel_patterns.csv"
    ),
    index=False
)

print("\n" + "=" * 65)
print("RECOMMENDATION SYSTEM COMPLETED")
print("=" * 65)

print("\nFiles saved in:")
print(OUTPUT_DIR)

print("\n1. personalized_attraction_recommendations.csv")
print("2. destination_ranking.csv")
print("3. historical_travel_patterns.csv")

print("\nNote:")
print(
    "Historical expenditure represents past survey observations, "
    "not live travel prices or guaranteed affordability."
)
print(
    "Relevance scores are rule-based ranking scores, "
    "not machine learning accuracy scores."
)
