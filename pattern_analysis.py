
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("dataset/cleaned_train.csv")

print("========== DELIVERY PATTERN ANALYSIS ==========")

# 1. Delivery count by city
print("\n1. DELIVERIES BY CITY")
print(df["City"].value_counts())

# 2. Average delivery time by city
print("\n2. AVERAGE DELIVERY TIME BY CITY")
print(df.groupby("City")["Time_taken(min)"].mean().round(2))

# 3. Average delivery time by traffic density
print("\n3. DELIVERY TIME BY TRAFFIC DENSITY")
print(df.groupby("Road_traffic_density")["Time_taken(min)"].mean().round(2))

# 4. Average delivery time by weather
print("\n4. DELIVERY TIME BY WEATHER")
print(df.groupby("Weatherconditions")["Time_taken(min)"].mean().round(2))

# 5. Average delivery time by vehicle condition
print("\n5. DELIVERY TIME BY VEHICLE CONDITION")
print(df.groupby("Vehicle_condition")["Time_taken(min)"].mean().round(2))

# 6. Average delivery time by number of multiple deliveries
print("\n6. DELIVERY TIME BY MULTIPLE DELIVERIES")
print(df.groupby("multiple_deliveries")["Time_taken(min)"].mean().round(2))

# 7. Average delivery time by festival
print("\n7. DELIVERY TIME BY FESTIVAL")
print(df.groupby("Festival")["Time_taken(min)"].mean().round(2))

# 8. Overall delivery time statistics
print("\n8. OVERALL DELIVERY TIME STATISTICS")
print(df["Time_taken(min)"].describe())

print("\n========== PATTERN ANALYSIS COMPLETED ==========")
