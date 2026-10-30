
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_train.csv")

# Create folder to save graphs
os.makedirs("visualizations", exist_ok=True)

sns.set_theme(style="whitegrid")

print("EXPLORATORY DATA ANALYSIS")
print("=" * 40)

# 1. Basic statistics
print("\n1. DELIVERY TIME STATISTICS")
print(df["Time_taken(min)"].describe())

# 2. Orders by city
print("\n2. ORDERS BY CITY")
print(df["City"].value_counts())

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="City")
plt.title("Number of Deliveries by City")
plt.xlabel("City")
plt.ylabel("Number of Deliveries")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("visualizations/orders_by_city.png")
plt.show()

# 3. Delivery time by traffic density
print("\n3. AVERAGE DELIVERY TIME BY TRAFFIC")
print(df.groupby("Road_traffic_density")["Time_taken(min)"].mean())

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Road_traffic_density",
            y="Time_taken(min)", errorbar=None)
plt.title("Average Delivery Time by Traffic Density")
plt.xlabel("Traffic Density")
plt.ylabel("Average Delivery Time (minutes)")
plt.tight_layout()
plt.savefig("visualizations/traffic_delivery_time.png")
plt.show()

# 4. Orders by food order type
print("\n4. ORDERS BY TYPE")
print(df["Type_of_order"].value_counts())

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Type_of_order")
plt.title("Number of Orders by Food Type")
plt.xlabel("Food Order Type")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.savefig("visualizations/orders_by_type.png")
plt.show()

# 5. Average delivery time by weather
print("\n5. DELIVERY TIME BY WEATHER")
print(df.groupby("Weatherconditions")["Time_taken(min)"].mean())

plt.figure(figsize=(9, 5))
sns.barplot(data=df, x="Weatherconditions",
            y="Time_taken(min)", errorbar=None)
plt.title("Average Delivery Time by Weather")
plt.xlabel("Weather Conditions")
plt.ylabel("Average Delivery Time (minutes)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("visualizations/weather_delivery_time.png")
plt.show()

# 6. Delivery time by vehicle condition
print("\n6. DELIVERY TIME BY VEHICLE CONDITION")
print(df.groupby("Vehicle_condition")["Time_taken(min)"].mean())

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Vehicle_condition",
            y="Time_taken(min)", errorbar=None)
plt.title("Delivery Time by Vehicle Condition")
plt.xlabel("Vehicle Condition")
plt.ylabel("Average Delivery Time (minutes)")
plt.tight_layout()
plt.savefig("visualizations/vehicle_delivery_time.png")
plt.show()

# 7. Delivery person ratings
print("\n7. DELIVERY PERSON RATINGS")
print(df["Delivery_person_Ratings"].describe())

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Delivery_person_Ratings",
             bins=15, kde=True)
plt.title("Distribution of Delivery Person Ratings")
plt.xlabel("Ratings")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("visualizations/ratings_distribution.png")
plt.show()

# 8. Orders by hour
time_values = df["Time_Orderd"].astype(str)

df["Order_Hour"] = pd.to_numeric(
    time_values.str.extract(r"(\d{1,2}):")[0],
    errors="coerce"
)

print("\n8. ORDERS BY HOUR")
print(df["Order_Hour"].value_counts().sort_index())

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="Order_Hour", order=sorted(
    df["Order_Hour"].dropna().unique()
))
plt.title("Number of Orders by Hour")
plt.xlabel("Order Hour")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.savefig("visualizations/orders_by_hour.png")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("All graphs saved in the visualizations folder.")
