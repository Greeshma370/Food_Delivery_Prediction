
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Loading dataset...")

# 1. Load cleaned dataset
df = pd.read_csv("dataset/cleaned_train.csv")

# 2. Calculate distance using latitude and longitude
def calculate_distance(row):
    lat1 = np.radians(row["Restaurant_latitude"])
    lon1 = np.radians(row["Restaurant_longitude"])
    lat2 = np.radians(row["Delivery_location_latitude"])
    lon2 = np.radians(row["Delivery_location_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 6371 * 2 * np.arcsin(np.sqrt(a))

df["Distance_km"] = df.apply(calculate_distance, axis=1)

# 3. Extract order hour and pickup hour
df["Order_Hour"] = pd.to_numeric(
    df["Time_Orderd"].str.extract(r"(\d{1,2}):")[0],
    errors="coerce"
)

df["Pickup_Hour"] = pd.to_numeric(
    df["Time_Order_picked"].str.extract(r"(\d{1,2}):")[0],
    errors="coerce"
)

# 4. Select input features
numeric_features = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Vehicle_condition",
    "multiple_deliveries",
    "Distance_km",
    "Order_Hour",
    "Pickup_Hour"
]

categorical_features = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City"
]

features = numeric_features + categorical_features

X = df[features]
y = df["Time_taken(min)"]

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 6. Prepare numerical and categorical data
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 7. Create Random Forest model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# 8. Train the model
print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

# 9. Make predictions
predictions = model.predict(X_test)

# 10. Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n========== MODEL PERFORMANCE ==========")
print("Mean Absolute Error (MAE):", round(mae, 2), "minutes")
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# 11. Display sample predictions
results = pd.DataFrame({
    "Actual Delivery Time": y_test.values,
    "Predicted Delivery Time": predictions.round(2)
})

print("\n========== SAMPLE PREDICTIONS ==========")
print(results.head(10))
# Save the trained machine learning model
joblib.dump(model, "delivery_time_model.pkl")

print("Model saved successfully!")

print("\nMACHINE LEARNING COMPLETED SUCCESSFULLY!")
