import numpy as np

def run_model(model, scaler, signal):
    # Convert to NumPy array
    print(f"testing features print: {features}")
    new_data = np.array(list(features.values()))
    
    # Print to verify
    #print(new_data)
    #print(type(new_data))

    # Scale the new data
    new_data_scaled = scaler.transform(new_data.reshape(-1, 1))

    # # Predict
    # prediction = xgb_model.predict(new_data_scaled)
    
    prediction = model.predict(new_data)
    print("Predicted Class:", prediction[0])


    return prediction[0]