
import pandas as pd

# Load the training dataset
df = pd.read_csv("dataset/train.csv")

print("\nFIRST FIVE RECORDS:")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nDATASET INFORMATION:")
df.info()

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDUPLICATE RECORDS:")
print(df.duplicated().sum())
