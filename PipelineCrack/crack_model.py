import numpy as np

def run_model(model, scaler, signal):
    # Convert to NumPy array
    print(f"testing features print: {signal}")
    new_data = np.array(list(signal.values()))
    
    # Print to verify
    #print(new_data)
    #print(type(new_data))

    print(new_data.shape)
    # Scale the new data
    #new_data_scaled = scaler.transform(new_data.reshape(1,14))
    #print(new_data.reshape(1,14))
    prediction = model.predict(new_data.reshape(1,14))

    return prediction[0]