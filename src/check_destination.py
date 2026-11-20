
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

file_path = os.path.join(
    BASE_DIR, "outputs", "overnight_tourism_expenditure.csv"
)

df = pd.read_csv(file_path)

print("========== MAIN DESTINATION COUNTS ==========")
print(df["main_destination"].value_counts(dropna=False).sort_index())

print("\n========== DESTINATION STATE MISSING CHECK ==========")
print(
    pd.crosstab(
        df["main_destination"],
        df["destination_state"].isnull(),
        margins=True
    )
)

print("\n========== SURVEY WEIGHT CHECK ==========")
print(df["mult"].describe())
print("Missing weights:", df["mult"].isnull().sum())
print("Unique weight values:", df["mult"].nunique())
