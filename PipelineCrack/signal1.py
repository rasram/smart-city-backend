import serial
import time
import csv
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fft
from scipy.stats import entropy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Initialize serial connection (replace with your Arduino's port)


def collect_signal():
    ser = serial.Serial('COM10', 9600, timeout=1)

# Allow time for Arduino reset
    time.sleep(2)

    time_list = []
    values = []
# Open a CSV file for writing
    with open("distance_data.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time (s)", "Distance (cm)"])  # CSV Header

        print("Logging distance for 10 seconds...")
        start_time = time.time()
        log_duration = 10  # seconds

        while True:
            current_time = time.time() - start_time
            if current_time > log_duration:
                break
            
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                try:
                    dist_value = float(line)  # Convert to float
                    time_list.append(current_time)
                    time_list.append(current_time + 0.01)
                    values.append(2000 - dist_value)
                    values.append(0)
                    
                    # Write data to CSV
                    writer.writerow([round(current_time, 2), 2000 - dist_value])
                    print(f"Time: {round(current_time, 2)}s | Distance: {dist_value} cm")
                
                except ValueError:
                    print("Invalid data received, skipping...")
    print("Done logging. CSV file saved as distance_data.csv")
    ser.close()
    return values,time_list

def crack_growth_with_factors(t, a, C0=1e-10, m=3.0, sigma=200, T=298, H2S=0, C_threshold=1e-4, alpha=0.5, beta=2, mu=0.5, pi=np.pi):
    C_T = C0 * np.exp(-mu / (8.314 * T))
    SCC_factor = alpha * (H2S / C_threshold) ** beta
    C_x = np.random.normal(1.0, 0.1)
    return C_T * (sigma * np.sqrt(pi * a)) ** m * (1 + SCC_factor) * C_x

def simulate_crack_growth(a0=0.005, time_span=(0, 1000), time_eval=np.linspace(0, 1000, 500)):
    sol = solve_ivp(crack_growth_with_factors, time_span, [a0], t_eval=time_eval, method='RK45')
    t_values = sol.t
    crack_sizes = sol.y[0]
    growth_rates = np.zeros_like(crack_sizes)
    growth_rates[1:] = crack_sizes[1:] - crack_sizes[:-1]
    growth_rates[0] = growth_rates[1]
    return t_values, crack_sizes, growth_rates

def generate_burst_noise(N, probability=0.1, strength=10):
    noise = np.random.normal(0, 1, N)
    bursts = np.random.rand(N) < probability
    noise[bursts] += np.random.normal(0, strength, np.sum(bursts))
    return noise

def generate_fem_data(samples=50):
    crack_sizes = np.linspace(0.001, 0.02, samples)
    fem_wave_responses = np.sin(2 * np.pi * 50 * crack_sizes)
    return crack_sizes.reshape(-1, 1), fem_wave_responses

fem_inputs, fem_outputs = generate_fem_data()
scaler = StandardScaler()
fem_inputs_scaled = scaler.fit_transform(fem_inputs)
gp_model = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
gp_model.fit(fem_inputs_scaled, fem_outputs)

def predict_fem(crack_size):
    crack_size_scaled = scaler.transform([[crack_size]])
    return gp_model.predict(crack_size_scaled)[0]

def leakage_pressure_drop(crack_size, P1=1e5, P2=9e4, rho=1000, Cd=0.62, A_ref=1e-4):
    A = A_ref * (crack_size / 0.01)
    velocity = np.sqrt(2 * (P1 - P2) / rho)
    flow_rate = Cd * A * velocity
    delta_P = rho * (velocity ** 2) / 2
    return delta_P, flow_rate

def compute_spectral_entropy(signal, fs=500):
    _, psd = welch(signal, fs=fs)
    total_psd = np.sum(psd)
    if total_psd > 0:
        psd_norm = psd / total_psd
        mask = psd_norm > 0
        entropy = -np.sum(psd_norm[mask] * np.log2(psd_norm[mask]))
        return entropy
    else:
        return 0.0

time_values, crack_sizes, growth_rates = simulate_crack_growth()


# Feature Extraction Functions
def preprocess_signal(signal_data, fs):
    """Apply bandpass filter to remove noise."""
    lowcut, highcut = 10, fs / 2 - 1  # Ensure highcut is positive
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
    signal_entropy = entropy(hist + 1e-9)  
    
    A_in, A_out = amplitude, np.max(np.abs(filtered_signal[len(filtered_signal)//2:]))
    signal_attenuation = 20 * np.log10(A_in / (A_out + 1e-9))
    

    n_samples = len(crack_sizes)
    
    for i in range(100):
        idx = i % n_samples
        a = crack_sizes[idx]
        growth_rate = growth_rates[idx]
        
        if i >= n_samples:
            a += np.random.normal(0, 0.0005)
            a = max(0.001, a)
            
        fem_approx = predict_fem(a)
    delta_P, flow_rate = leakage_pressure_drop(a)

    return {
        "CrackSize":a,
        "FEM_Approx":fem_approx,
        "Amplitude (dB)": amplitude,
        "Duration (ms)": duration,
        "Rise Time (ms)": rise_time,
        "Counts": max(5, min(50, int(flow_rate * 1e6))),
        "Energy (a.u.)": energy,
        "Peak Frequency (kHz)": peak_frequency,
        "RMS Voltage (V)": rms_voltage,
        "Signal Attenuation (dB/m)": signal_attenuation,
        "PressureDrop":delta_P,
        "FlowRate":flow_rate,
        "SpectralEntropy": signal_entropy,
        "GrowthRate":growth_rate
    }

fs = 100 


def get_signal(values):
    if len(values) > 0:
        normal_features = extract_features(values, fs)
        print("\nNormal Pipe Signal Features:", normal_features)
        return normal_features
    else:
        print("No valid signal data collected for feature extraction.")
