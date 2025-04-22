import joblib
from sklearn.ensemble import RandomForestRegressor
import numpy as np

X_train = np.random.randint(0, 3, size=(100, 3))
y_train = np.random.uniform(1.0, 10.0, size=100)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "order_delivery_model.pkl")
print("Model trained and saved.")
