import serial
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fft
from scipy.stats import entropy
# Replace with your Arduino's port (e.g. 'COM3', '/dev/ttyACM0', etc.)
ser = serial.Serial('/dev/cu.usbmodem11101', 9600, timeout=1)

# Allow time for Arduino reset
time.sleep(2)
time_list = []
values = []
# Open a CSV file for writing
with open("distance_data.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    # Write a header row
    writer.writerow(["Time (s)", "Distance (cm)"])

    print("Logging distance for 10 seconds...")
    start_time = time.time()
    log_duration = 10  # seconds

    while True:
        current_time = time.time() - start_time
        if current_time > log_duration:
            break
        
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            
            # Expecting just a numeric value from Arduino
            # If you see "Distance: 1196.93 cm", parse out the number
            dist_value = float(line)
            time_list.append(current_time)
            time_list.append(current_time+0.01)
            values.append(2000-dist_value)
            values.append(0)
            # Write the data (time + distance) to CSV
            writer.writerow([round(current_time, 2), 2000-dist_value])
            print(f"Time: {round(current_time, 2)}s | Distance: {dist_value} cm")

print("Done logging. CSV file saved as distance_data.csv")
ser.close()
print(time_list)
print(values)




time_data = []
distance_data = []

# 1) Open the CSV file
with open("distance_data.csv", mode="r") as f:
    reader = csv.reader(f)
    
    # 2) Skip the header row if you have one
    header = next(reader)  # e.g., ["Time (s)", "Distance (cm)"]

    # 3) Read each row and parse the time and distance
    for row in reader:
        # row[0] should be the time, row[1] the distance
        time_value = float(row[0])
        dist_value = float(row[1])
        
        time_data.append(time_value)
        distance_data.append(dist_value)

# 4) Plot the data
plt.plot(time_data, distance_data)
plt.xlabel("Time (s)")
plt.ylabel("Distance (cm)")
plt.title("Distance vs Time from CSV")
plt.show()



def preprocess_signal(signal_data, fs):
    """Apply bandpass filter to remove noise."""
    lowcut, highcut = 1000, fs / 2 - 1000  
    sos = signal.butter(4, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return signal.sosfilt(sos, signal_data)

def extract_features(signal_data, fs):
    """Extract features from the ultrasonic signal."""
    filtered_signal = preprocess_signal(signal_data, fs)
    
   
    amplitude = np.max(np.abs(filtered_signal))
    
   
    threshold = 0.1 * amplitude
    indices = np.where(np.abs(filtered_signal) > threshold)[0]
    duration = (indices[-1] - indices[0]) / fs if len(indices) > 0 else 0
   
    rise_start, rise_end = int(0.1 * len(filtered_signal)), int(0.9 * len(filtered_signal))
    rise_time = (rise_end - rise_start) / fs
    
   
    energy = np.sum(filtered_signal ** 2)
    
    rms_voltage = np.sqrt(np.mean(filtered_signal ** 2))
    
    freq_spectrum = np.abs(fft.fft(filtered_signal))
    freqs = fft.fftfreq(len(filtered_signal), d=1/fs)
    peak_frequency = freqs[np.argmax(freq_spectrum[:len(freq_spectrum)//2])]
    
    hist, bin_edges = np.histogram(filtered_signal, bins=50, density=True)
    signal_entropy = entropy(hist + 1e-9)  # Small constant to avoid log(0)
 
    A_in, A_out = amplitude, np.max(np.abs(filtered_signal[len(filtered_signal)//2:]))
    signal_attenuation = 20 * np.log10(A_in / (A_out + 1e-9))
    
    material_factor = 1.0

    
    crack_direction_factor = np.random.uniform(0.5, 1.5) 
    return {
        "Amplitude": amplitude,
        "Duration": duration,
        "Rise Time": rise_time,
        "Energy": energy,
        "RMS Voltage": rms_voltage,
        "Peak Frequency": peak_frequency,
        "Signal Entropy": signal_entropy,
        "Signal Attenuation": signal_attenuation,
        "Material Factor": material_factor,
        "Crack Direction Factor": crack_direction_factor,
    }

## Testing 

fs = 100000  
signal_data = np.sin(2 * np.pi * 5000 * np.linspace(0, 0.01, fs)) + 0.2 * np.random.randn(fs)  
features = extract_features(signal_data, fs)
print(features)