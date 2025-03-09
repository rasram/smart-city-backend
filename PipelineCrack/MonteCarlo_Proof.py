# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import plotly.express as px
# from scipy.integrate import solve_ivp
# from scipy.signal import welch
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPRegressor
# from joblib import Parallel, delayed

# # Constants
# PI = np.pi
# R = 8.314  # Universal Gas Constant

# # Crack Growth Function
# def crack_growth_with_factors(t, a, C0=1e-10, m=3.0, sigma=200, T=298, H2S=0, C_threshold=1e-4, alpha=0.5, beta=2, mu=0.5):
#     C_T = C0 * np.exp(-mu / (R * T))  # Temperature dependence
#     SCC_factor = alpha * (H2S / C_threshold) ** beta if H2S > 0 else 0
#     C_x = 1.0  # Remove randomness to maintain consistency
#     return C_T * (sigma * np.sqrt(PI * a)) ** m * (1 + SCC_factor) * C_x

# # Simulate Crack Growth
# def simulate_crack_growth(a0=0.005, time_span=(0, 1000), time_eval=np.linspace(0, 1000, 500)):
#     sol = solve_ivp(lambda t, a: crack_growth_with_factors(t, a), time_span, [a0], t_eval=time_eval, method='RK45')
#     t_values = sol.t
#     crack_sizes = sol.y[0]
#     growth_rates = np.diff(crack_sizes, prepend=crack_sizes[0])
#     return t_values, crack_sizes, growth_rates

# # Generate FEM Data
# def generate_fem_data(samples=50):
#     crack_sizes = np.linspace(0.001, 0.02, samples)
#     fem_wave_responses = np.sin(2 * np.pi * 50 * crack_sizes)
#     return crack_sizes.reshape(-1, 1), fem_wave_responses

# # Train MLP for FEM Approximation
# fem_inputs, fem_outputs = generate_fem_data()
# scaler = StandardScaler()
# fem_inputs_scaled = scaler.fit_transform(fem_inputs)
# gp_model = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
# gp_model.fit(fem_inputs_scaled, fem_outputs)

# # FEM Prediction Function
# def predict_fem(crack_size):
#     crack_size_scaled = scaler.transform([[crack_size]])
#     return gp_model.predict(crack_size_scaled)[0]

# # Leakage Pressure Drop
# def leakage_pressure_drop(crack_size, P1=1e5, P2=9e4, rho=1000, Cd=0.62, A_ref=1e-4):
#     A = A_ref * (crack_size / 0.01)
#     velocity = np.sqrt(2 * (P1 - P2) / rho)
#     flow_rate = Cd * A * velocity
#     delta_P = rho * (velocity ** 2) / 2
#     return delta_P, flow_rate

# # Compute Spectral Entropy
# def compute_spectral_entropy(signal, fs=500):
#     _, psd = welch(signal, fs=fs)
#     total_psd = np.sum(psd)
#     if total_psd > 0:
#         psd_norm = psd / total_psd
#         entropy = -np.sum(psd_norm[psd_norm > 0] * np.log2(psd_norm[psd_norm > 0] + 1e-10))  # Numerical stability
#         return entropy
#     return 0.0

# # Generate Burst Noise
# def generate_burst_noise(N, probability=0.1, strength=10):
#     noise = np.random.normal(0, 1, N)
#     bursts = np.random.rand(N) < probability
#     noise[bursts] += np.random.normal(0, strength, np.sum(bursts))
#     return noise

# # Generate Synthetic Data
# def generate_synthetic_data(N=500):
#     time_values, crack_sizes, growth_rates = simulate_crack_growth()
#     data = []
#     n_samples = len(crack_sizes)

#     for i in range(N):
#         idx = i % n_samples
#         a = crack_sizes[idx] + np.random.normal(0, 0.0005) if i >= n_samples else crack_sizes[idx]
#         a = max(0.001, a)
        
#         fem_approx = predict_fem(a)
#         burst_noise = generate_burst_noise(500)
#         wave_signal = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 500)) + 0.5 * burst_noise
#         delta_P, flow_rate = leakage_pressure_drop(a)
#         entropy = compute_spectral_entropy(wave_signal)
        
#         amplitude = np.max(np.abs(wave_signal)) * 100
#         duration = max(0.5, min(5, a * 1000))
#         rise_time = duration / 10
#         counts = max(5, min(50, int(flow_rate * 1e6)))
#         energy = 0.5 * amplitude * amplitude * duration
#         peak_frequency = max(10, min(100, 50 + a * 1000))
#         rms_voltage = amplitude / 10
#         material_factor = np.random.normal(1.0, 0.1)
#         crack_direction_factor = np.cos(np.random.uniform(0, np.pi))
#         signal_attenuation = max(0.1, min(5, a * 100 * material_factor * crack_direction_factor))
        
#         data.append([time_values[i % len(time_values)], a, fem_approx, amplitude, duration, rise_time, counts, energy, peak_frequency, rms_voltage, signal_attenuation, delta_P, flow_rate, entropy, growth_rates[idx]])
    
#     return np.array(data)

# # Monte Carlo Simulation Function
# def monte_carlo_simulation():
#     _, crack_sizes, growth_rates = simulate_crack_growth()
#     crack_size = np.random.choice(crack_sizes)
    
#     delta_P, flow_rate = leakage_pressure_drop(crack_size)
#     burst_noise = generate_burst_noise(500)
#     wave_signal = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 500)) + 0.5 * burst_noise
#     entropy = compute_spectral_entropy(wave_signal)
    
#     return growth_rates[-1], flow_rate, delta_P, entropy

# # Run Monte Carlo Simulations in Parallel
# num_simulations = 1000
# results = Parallel(n_jobs=-1)(delayed(monte_carlo_simulation)() for _ in range(num_simulations))

# # Convert Results to Numpy Arrays
# crack_growth_rates, flow_rates, pressure_drops, entropy_values = map(np.array, zip(*results))

# # Save Results to CSV
# df = pd.DataFrame({
#     "Crack Growth Rate": crack_growth_rates,
#     "Flow Rate": flow_rates,
#     "Pressure Drop": pressure_drops,
#     "Spectral Entropy": entropy_values
# })
# df.to_csv("monte_carlo_results.csv", index=False)

# # Plot Distributions
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# sns.histplot(crack_growth_rates, bins=30, kde=True, ax=axes[0, 0])
# axes[0, 0].set_title("Crack Growth Rate Distribution")

# sns.histplot(flow_rates, bins=30, kde=True, ax=axes[0, 1])
# axes[0, 1].set_title("Flow Rate Distribution")

# sns.histplot(pressure_drops, bins=30, kde=True, ax=axes[1, 0])
# axes[1, 0].set_title("Pressure Drop Distribution")

# sns.histplot(entropy_values, bins=30, kde=True, ax=axes[1, 1])
# axes[1, 1].set_title("Spectral Entropy Distribution")

# plt.tight_layout()
# plt.show()

# # Interactive Plot
# fig = px.histogram(df, x="Crack Growth Rate", nbins=30, title="Crack Growth Rate Distribution", marginal="rug")
# fig.show()

# # Compute Mean & Std Dev
# print(df.describe().loc[['mean', 'std']])

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

np.random.seed(42)

# Crack growth function with corrections
def crack_growth_with_factors(t, a, C0=1e-10, m=3.0, sigma=200, T=298, H2S=0, C_threshold=1e-4, alpha=0.5, beta=2, mu=0.5, pi=np.pi):
    if a < 1e-6:  # Ensure crack size remains positive
        a = 1e-6
    C_T = C0 * np.exp(-mu / (8.314 * T))
    SCC_factor = alpha * (H2S / C_threshold) ** beta
    C_x = np.random.normal(1.0, 0.1)
    return C_T * (sigma * np.sqrt(pi * a)) ** m * (1 + SCC_factor) * C_x

# Simulate crack growth over time
def simulate_crack_growth(a0=0.005, time_span=(0, 1000), time_eval=np.linspace(0, 1000, 500)):
    sol = solve_ivp(crack_growth_with_factors, time_span, [a0], t_eval=time_eval, method='RK45')
    t_values = sol.t
    crack_sizes = sol.y[0]
    crack_sizes[crack_sizes < 0] = 1e-6  # Prevent negative crack sizes
    growth_rates = np.diff(crack_sizes, prepend=crack_sizes[0])
    return t_values, crack_sizes, growth_rates

# Generate FEM wave responses
def generate_fem_data(samples=50):
    crack_sizes = np.linspace(0.001, 0.02, samples)
    fem_wave_responses = np.sin(2 * np.pi * 50 * crack_sizes)
    return crack_sizes.reshape(-1, 1), fem_wave_responses

# Train FEM approximation model
fem_inputs, fem_outputs = generate_fem_data()
scaler = StandardScaler()
fem_inputs_scaled = scaler.fit_transform(fem_inputs)
gp_model = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
gp_model.fit(fem_inputs_scaled, fem_outputs)

# Predict FEM response for a given crack size
def predict_fem(crack_size):
    crack_size_scaled = scaler.transform([[crack_size]])
    return gp_model.predict(crack_size_scaled)[0]

# Leakage pressure drop model
def leakage_pressure_drop(crack_size, P1=1e5, P2=9e4, rho=1000, Cd=0.62, A_ref=1e-4):
    A = A_ref * (crack_size / 0.01)
    velocity = np.sqrt(2 * (P1 - P2) / rho)
    flow_rate = Cd * A * velocity
    delta_P = rho * (velocity ** 2) / 2
    return delta_P, flow_rate

# Compute spectral entropy
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

# Generate burst noise
def generate_burst_noise(N, probability=0.1, strength=10):
    noise = np.random.normal(0, 1, N)
    bursts = np.random.rand(N) < probability
    noise[bursts] += np.random.normal(0, strength, np.sum(bursts))
    return noise

# Simulate crack growth data
time_values, crack_sizes, growth_rates = simulate_crack_growth()

# Generate synthetic dataset
def generate_synthetic_data(N=500):
    data = []
    for i in range(N):
        idx = i % len(crack_sizes)
        a = max(0.001, crack_sizes[idx] + np.random.normal(0, 0.0005))
        growth_rate = growth_rates[idx]

        # FEM Approximation
        fem_approx = predict_fem(a)

        # Generate burst noise and wave signal
        burst_noise = generate_burst_noise(500)
        wave_signal = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 500)) + 0.5 * burst_noise

        # Leakage pressure drop model
        delta_P, flow_rate = leakage_pressure_drop(a)

        # Extract waveform features
        entropy = compute_spectral_entropy(wave_signal)
        amplitude = np.max(np.abs(wave_signal)) * 100
        duration = max(0.5, min(5, a * 1000))
        rise_time = duration / 10
        counts = max(5, min(50, int(flow_rate * 1e6)))
        energy = 0.5 * amplitude * amplitude * duration
        peak_frequency = max(10, min(100, 50 + a * 1000))
        rms_voltage = amplitude / 10
        material_factor = np.random.normal(1.0, 0.1)
        crack_direction_factor = np.cos(np.random.uniform(0, np.pi))
        signal_attenuation = max(0.1, min(5, a * 100 * material_factor * crack_direction_factor))

        # Store data
        data.append([time_values[idx], a, fem_approx, amplitude, duration, rise_time, counts, energy, peak_frequency, rms_voltage, signal_attenuation, delta_P, flow_rate, entropy, growth_rate])
    
    return np.array(data)

# Generate and save dataset
data = generate_synthetic_data()
dataset = pd.DataFrame(data, columns=["Time (hours)", "CrackSize", "FEM_Approx", "Amplitude (dB)", "Duration (ms)", "Rise Time (ms)", "Counts", "Energy (a.u.)", "Peak Frequency (kHz)", "RMS Voltage (V)", "Signal Attenuation (dB/m)", "PressureDrop", "FlowRate", "SpectralEntropy", "GrowthRate"])
dataset.to_csv("hybrid_synthetic_ultrasonic_data.csv", index=False)

print("Hybrid Synthetic Data Generation Complete with Crack Growth Simulation and GrowthRate Restoration!")
