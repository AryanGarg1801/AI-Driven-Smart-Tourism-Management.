
import pandas as pd
import os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

file_path = os.path.join(
    base_path, "dataset", "DTES25_26_CSV", "l6_rev.csv"
)

df = pd.read_csv(file_path)

print("========== EXPENDITURE FORMULA CHECK ==========")

package_cols = [
    col for col in df.columns
    if "package" in col.lower()
]

print("Package-related columns:", package_cols)

print("\nRelevant subtotal and total columns:")
for col in df.columns:
    if "sub_tot" in col.lower() or "total_3_10" in col.lower():
        print(col)

if len(package_cols) == 1:
    package_col = package_cols[0]

    package = pd.to_numeric(
        df[package_col], errors="coerce"
    ).fillna(0)

    non_package = pd.to_numeric(
        df["sub_tot4_5_6_7_8_9"], errors="coerce"
    ).fillna(0)

    actual_total = pd.to_numeric(
        df["total_3_10"], errors="coerce"
    )

    calculated_total = package + non_package

    difference = actual_total - calculated_total

    print("\nPackage column:", package_col)
    print("Total records:", len(df))
    print("Exact matches:", (difference == 0).sum())
    print("Mismatches:", (difference != 0).sum())
    print("Missing total values:", actual_total.isna().sum())

    if (difference != 0).any():
        print("\nSample mismatches:")
        print(df.loc[
            difference != 0,
            [package_col, "sub_tot4_5_6_7_8_9", "total_3_10"]
        ].head(10).to_string(index=False))

else:
    print("Please inspect the package-related columns manually.")

print("\n========== CHECK COMPLETED ==========")
