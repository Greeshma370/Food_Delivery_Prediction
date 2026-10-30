
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Food Delivery Analytics",
    page_icon="🍔",
    layout="wide"
)

# --------------------------------
# 2. LOAD DATA AND MODEL
# --------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("dataset/cleaned_train.csv")

@st.cache_resource
def load_model():
    return joblib.load("delivery_time_model_compressed.pkl")

df = load_data()
model = load_model()

# --------------------------------
# 3. PROJECT TITLE
# --------------------------------

st.title("🍔 Food Delivery Analytics and Order Pattern Prediction")

st.write(
    "This project analyses food delivery data and predicts "
    "delivery time using Machine Learning."
)

st.divider()

# --------------------------------
# 4. DATASET OVERVIEW
# --------------------------------

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Deliveries", f"{len(df):,}")
col2.metric(
    "Average Delivery Time",
    f"{df['Time_taken(min)'].mean():.2f} min"
)
col3.metric(
    "Average Delivery Rating",
    f"{df['Delivery_person_Ratings'].mean():.2f}"
)

st.divider()

# --------------------------------
# 5. DELIVERY ANALYSIS
# --------------------------------

st.header("📈 Delivery Pattern Analysis")

chart1, chart2 = st.columns(2)

with chart1:
    st.subheader("Deliveries by City")
    city_counts = df["City"].value_counts()
    st.bar_chart(city_counts)

with chart2:
    st.subheader("Average Delivery Time by Traffic")
    traffic_time = df.groupby(
        "Road_traffic_density"
    )["Time_taken(min)"].mean()
    st.bar_chart(traffic_time)

chart3, chart4 = st.columns(2)

with chart3:
    st.subheader("Average Delivery Time by Weather")
    weather_time = df.groupby(
        "Weatherconditions"
    )["Time_taken(min)"].mean()
    st.bar_chart(weather_time)

with chart4:
    st.subheader("Average Delivery Time by Vehicle Condition")
    vehicle_time = df.groupby(
        "Vehicle_condition"
    )["Time_taken(min)"].mean()
    st.bar_chart(vehicle_time)

st.divider()

# --------------------------------
# 6. DELIVERY TIME PREDICTION
# --------------------------------

st.header("🚚 Predict Food Delivery Time")

st.write(
    "Enter the delivery details below to estimate "
    "how long the delivery may take."
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Delivery Person Age",
        min_value=18,
        max_value=60,
        value=25
    )

    rating = st.number_input(
        "Delivery Person Rating",
        min_value=1.0,
        max_value=6.0,
        value=4.5,
        step=0.1
    )

    vehicle_condition = st.selectbox(
        "Vehicle Condition",
        [0, 1, 2, 3],
        index=2
    )

    multiple_deliveries = st.selectbox(
        "Number of Multiple Deliveries",
        [0, 1, 2, 3],
        index=0
    )

    city = st.selectbox(
        "City",
        ["Metropolitian", "Urban", "Semi-Urban", "Unknown"]
    )

    festival = st.selectbox(
        "Festival",
        ["No", "Yes", "Unknown"]
    )

with col2:
    weather = st.selectbox(
        "Weather Conditions",
        ["Sunny", "Cloudy", "Fog", "Stormy",
         "Sandstorms", "Windy", "Unknown"]
    )

    traffic = st.selectbox(
        "Road Traffic Density",
        ["Low", "Medium", "High", "Jam", "Unknown"]
    )

    order_type = st.selectbox(
        "Type of Order",
        ["Snack", "Meal", "Drinks", "Buffet"]
    )

    vehicle_type = st.selectbox(
        "Type of Vehicle",
        ["motorcycle", "scooter", "electric_scooter", "bicycle"]
    )

    order_hour = st.slider(
        "Order Hour",
        min_value=0,
        max_value=23,
        value=19
    )

    pickup_hour = st.slider(
        "Pickup Hour",
        min_value=0,
        max_value=23,
        value=19
    )

    distance = st.number_input(
        "Delivery Distance (km)",
        min_value=0.1,
        max_value=100.0,
        value=5.0,
        step=0.5
    )

# --------------------------------
# 7. PREDICTION BUTTON
# --------------------------------

if st.button("Predict Delivery Time", type="primary"):

    # Example coordinates are used only to fill
    # the model's coordinate input fields.
    # The entered distance is used for Distance_km.

    input_data = pd.DataFrame([{
        "Delivery_person_Age": age,
        "Delivery_person_Ratings": rating,
        "Restaurant_latitude": 12.9716,
        "Restaurant_longitude": 77.5946,
        "Delivery_location_latitude": 12.9716,
        "Delivery_location_longitude": 77.5946,
        "Vehicle_condition": vehicle_condition,
        "multiple_deliveries": multiple_deliveries,
        "Distance_km": distance,
        "Order_Hour": order_hour,
        "Pickup_Hour": pickup_hour,
        "Weatherconditions": weather,
        "Road_traffic_density": traffic,
        "Type_of_order": order_type,
        "Type_of_vehicle": vehicle_type,
        "Festival": festival,
        "City": city
    }])

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated Delivery Time: {prediction:.1f} minutes"
    )

    st.caption(
        "This is a machine learning estimate, not a guaranteed delivery time."
    )

# --------------------------------
# 8. FOOTER
# --------------------------------

st.divider()

st.write(
    "Big Data Analytics Project | Food Delivery Analytics "
    "and Order Pattern Prediction"
)
