
import pandas as pd

# STEP 1: Load the original dataset
df = pd.read_csv("dataset/train.csv")

print("Original dataset shape:", df.shape)

# STEP 2: Remove unnecessary spaces
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

# STEP 3: Remove unnecessary text from Weatherconditions
df["Weatherconditions"] = (
    df["Weatherconditions"]
    .str.replace("conditions ", "", regex=False)
)

# STEP 4: Replace hidden missing values
df = df.replace(
    ["NaN", "nan", "NULL", ""],
    pd.NA
)

# STEP 5: Convert numerical columns
numeric_columns = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column], errors="coerce"
    )

# STEP 6: Clean delivery time
df["Time_taken(min)"] = (
    df["Time_taken(min)"]
    .str.replace("(min)", "", regex=False)
    .str.strip()
)

df["Time_taken(min)"] = pd.to_numeric(
    df["Time_taken(min)"], errors="coerce"
)

# STEP 7: Handle missing values in numerical columns
for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )

# STEP 8: Handle missing categorical values
categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:
    df[column] = df[column].fillna("Unknown")

# STEP 9: Remove rows with missing target values
df = df.dropna(subset=["Time_taken(min)"])

# STEP 10: Save cleaned dataset
df.to_csv("dataset/cleaned_train.csv", index=False)

# STEP 11: Display final results
print("\nDATA CLEANING COMPLETED!")

print("\nCleaned dataset shape:")
print(df.shape)

print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nUpdated data types:")
print(df.dtypes)

print("\nFirst five cleaned records:")
print(df.head())

print("\nCleaned dataset saved successfully!")
