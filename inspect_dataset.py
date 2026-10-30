
import pandas as pd

df = pd.read_csv("dataset/train.csv")

# Check hidden missing values
print("\nHIDDEN MISSING VALUES:")

for column in df.columns:
    hidden = df[column].astype(str).str.strip().isin(
        ["NaN", "nan", "NULL", ""]
    ).sum()

    if hidden > 0:
        print(column, ":", hidden)

# Check unique values in important columns
columns_to_check = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Weatherconditions",
    "Road_traffic_density",
    "multiple_deliveries",
    "Festival",
    "City",
    "Time_taken(min)"
]

print("\nUNIQUE VALUES:")

for column in columns_to_check:
    print("\n", column)
    print(df[column].unique()[:20])
