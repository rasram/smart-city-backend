import numpy as np

def run_model(xgb_model, scaler, signal):
    # Convert to NumPy array
    print(signal)
    temp_data = list(signal.values())
    new_data = np.array(temp_data)
    
    # Print to verify
    #print(new_data)

    # Scale the new data
    new_data_scaled = scaler.transform(new_data.reshape(-1, 1))

    # Predict
    prediction = xgb_model.predict(new_data_scaled)

    # Output the prediction
    print("Predicted Class:", prediction[0])  # 0 = Not Cracked, 1 = Cracked

    return prediction[0]