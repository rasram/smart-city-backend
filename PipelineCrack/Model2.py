from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("hybrid_synthetic_ultrasonic_data_with_labels.csv")  # Update with the actual filename

# Split features and target
X = df.drop(columns=["Cracked"])  # Features
y = df["Cracked"]  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Save the trained model
joblib.dump(rf_model, "random_forest_model.pkl")
print("Model saved successfully!")

# Evaluate
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
