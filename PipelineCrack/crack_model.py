import numpy as np

from collections import Counter

def check_correctness(signal):    
    # Get values at odd indices and count how many are greater than 850
    odd_terms = [signal[i] for i in range(1, len(signal), 2)]
    correct = sum(1 for x in odd_terms if x > 1990)    
    return correct

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
    if prediction[0]==1.0:
        return 1.0
    else:
        if(scaler>=2):
            return 1.0        
    return prediction[0]