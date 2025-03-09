import numpy as np

def run_model(xgb_model, scaler, features):
    # Convert to NumPy array
    print(f"testing features print: {features}")
    new_data = np.array(list(features.values()))
    
    # Print to verify
    #print(new_data)
    #print(type(new_data))

    print(new_data.shape)
    # Scale the new data
    new_data_scaled = scaler.transform(new_data.reshape(1,14))
    

    # Predict
    prediction = xgb_model.predict(new_data_scaled)

    # Output the prediction
    print("Predicted Class:", prediction[0])  # 0 = Not Cracked, 1 = Cracked

    return prediction[0]