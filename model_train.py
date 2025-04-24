import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load dataset
df = pd.read_csv("Food_Delivery_Times.csv")

# Manually map categorical values (same as in app.py)
weather_map = {"Clear": 0, "Rainy": 1, "Windy": 2, "Foggy": 3}
traffic_map = {"Low": 0, "Medium": 1, "High": 2}
time_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
vehicle_map = {"Bike": 0, "Car": 1, "Van": 2}

df["Weather"] = df["Weather"].map(weather_map)
df["Traffic_Level"] = df["Traffic_Level"].map(traffic_map)
df["Time_of_Day"] = df["Time_of_Day"].map(time_map)
df["Vehicle_Type"] = df["Vehicle_Type"].map(vehicle_map)

# Features and target
X = df[['Distance_km', 'Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type', 'Preparation_Time_min', 'Courier_Experience_yrs']]
y = df['Delivery_Time_min']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"✅ Model trained successfully. MSE: {mse:.2f}")

# Save model
joblib.dump(model, "order_delivery_model.pkl")
print("📦 Model saved as 'order_delivery_model.pkl'")
