import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load training data
df = pd.read_csv("fake_sleep_training_data_varied.csv")

# Features and target
features = [
    "total_sleep_hrs",
    "light_sleep_hrs",
    "deep_sleep_hrs",
    "rem_sleep_hrs",
    "awake_hrs",
    "latency_mins",
    "interruptions",
    "consistency_score"
]
target = "sleep_score"

X = df[features]
y = df[target]

# Split into train/test (for eval purposes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate (optional)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model trained. Test MSE: {mse:.2f}")

# Save model
joblib.dump(model, "sleep_model.pkl")
print("Model saved as sleep_model.pkl")
