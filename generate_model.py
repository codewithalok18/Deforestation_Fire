from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import joblib

# Generate data with 6 features instead of 4
X, y = make_classification(n_samples=200, n_features=6, random_state=42)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier()
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, "best_fire_detection_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved with 6 features.")
