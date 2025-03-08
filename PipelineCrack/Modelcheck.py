import joblib
import numpy as np

# Load the trained model and scaler
xgb_model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

import numpy as np

data_dict = {
    "CrackSize": np.random.uniform(0.1, 5.0),  # Crack size in mm
    "FEM_Approx": np.random.uniform(0.01, 0.5),  # Finite Element Method approximation
    "Amplitude (dB)": np.random.uniform(20, 80),  # Signal amplitude in decibels
    "Duration (ms)": np.random.uniform(0.5, 5.0),  # Signal duration in milliseconds
    "Rise Time (ms)": np.random.uniform(0.1, 2.0),  # Rise time in milliseconds
    "Counts": np.random.randint(5, 50),  # Number of signal counts
    "Energy (a.u.)": np.random.uniform(100, 5000),  # Energy in arbitrary units
    "Peak Frequency (kHz)": np.random.uniform(50, 500),  # Frequency in kHz
    "RMS Voltage (V)": np.random.uniform(0.1, 5.0),  # RMS voltage in Volts
    "Signal Attenuation (dB/m)": np.random.uniform(0.1, 2.0),  # Attenuation in dB per meter
    "PressureDrop": np.random.uniform(0.1, 5.0),  # Pressure drop in Pascals
    "FlowRate": np.random.uniform(0.01, 0.5),  # Flow rate in cubic meters per second
    "SpectralEntropy": np.random.uniform(0.5, 1.5),  # Spectral entropy
    "GrowthRate": np.random.uniform(0.01, 0.5)  # Crack growth rate
}

# Print generated values
print(data_dict)

# Convert to NumPy array
new_data = np.array([list(data_dict.values())])

# Print to verify
print(new_data)

# Load the trained model and scaler
import joblib

xgb_model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# Scale the new data
new_data_scaled = scaler.transform(new_data)

# Predict
prediction = xgb_model.predict(new_data_scaled)

# Output the prediction
print("Predicted Class:", prediction[0])  # 0 = Not Cracked, 1 = Cracked
