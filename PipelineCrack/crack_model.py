import numpy as np

from collections import Counter

def check_correctness(signal):
    # Round off values in the list
    rounded_signal = [round(x) for x in signal]
    
    # Count occurrences of each unique element
    count_dict = Counter(rounded_signal)
    
    return len(count_dict)

def run_model(model, scaler, signal):
    # Convert to NumPy array
    print(f"testing features print: {signal}")
    new_data = np.array(list(signal.values()))
    
    # Print to verify
    #print(new_data)
    #print(type(new_data))
    print(f"Correlate : {scaler}")
    print(new_data.shape)
    # Scale the new data
    #new_data_scaled = scaler.transform(new_data.reshape(1,14))
    #print(new_data.reshape(1,14))
    prediction = model.predict(new_data.reshape(1,14)) 
    predict = True
    if(prediction == 0.0):
        predict = False
    if scaler== 2:
        return False
    return True