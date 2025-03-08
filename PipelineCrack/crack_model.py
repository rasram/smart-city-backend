import numpy as np

def run_model(xgb_model, scaler, signal):
    # Convert to NumPy array
    new_data = np.array([list(signal.values())])

    # Print to verify
    print(new_data)

    # Scale the new data
    new_data_scaled = scaler.transform(new_data)

    # Predict
    prediction = xgb_model.predict(new_data_scaled)

    # Output the prediction
    print("Predicted Class:", prediction[0])  # 0 = Not Cracked, 1 = Cracked

    return prediction[0]