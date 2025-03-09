# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPRegressor

# def crack_growth_with_factors(t, a, C0=1e-10, m=3.0, sigma=200, T=298, H2S=0, C_threshold=1e-4, alpha=0.5, beta=2, mu=0.5, pi=np.pi):
#     """ Crack growth rate based on Paris’ Law with SCC effects. """
#     C_T = C0 * np.exp(-mu / (8.314 * T))  # Temperature-dependent coefficient
#     SCC_factor = alpha * (H2S / C_threshold) ** beta  # Stress corrosion cracking effect
#     return C_T * (sigma * np.sqrt(pi * a)) ** m * (1 + SCC_factor)

# def simulate_crack_growth(a0=0.005, time_span=(0, 1000), time_eval=np.linspace(0, 1000, 500)):
#     """ Solves the crack growth differential equation using RK45. """
#     sol = solve_ivp(crack_growth_with_factors, time_span, [a0], t_eval=time_eval, method='RK45')
#     t_values = sol.t
#     crack_sizes = sol.y[0]
#     growth_rates = np.gradient(crack_sizes, t_values)
#     return t_values, crack_sizes, growth_rates

# def predict_fem(crack_size):
#     """ Placeholder FEM approximation model. """
#     return 0.1 * crack_size ** 2 + 0.05 * crack_size + 0.02

# def generate_burst_noise(size):
#     """ Generate random burst noise for simulation. """
#     return np.random.normal(0, 0.1, size)

# def leakage_pressure_drop(a):
#     """ Compute leakage pressure drop and flow rate. """
#     delta_P = 0.2 * a + np.random.normal(0, 0.01)
#     flow_rate = 0.1 * a ** 2 + np.random.normal(0, 0.005)
#     return delta_P, flow_rate

# def compute_spectral_entropy(signal):
#     """ Compute spectral entropy from signal. """
#     power_spectrum = np.abs(np.fft.fft(signal))**2
#     prob_distribution = power_spectrum / np.sum(power_spectrum)
#     entropy = -np.sum(prob_distribution * np.log2(prob_distribution + 1e-10))
#     return entropy

# def generate_synthetic_data(N=500):
#     """ Generate synthetic data with FEM approximation and ultrasonic parameters. """
#     time_values, crack_sizes, growth_rates = simulate_crack_growth()
#     n_samples = len(crack_sizes)
#     data = []
    
#     for i in range(N):
#         idx = i % n_samples
#         a = crack_sizes[idx]
#         growth_rate = growth_rates[idx]
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
        
#         data.append([
#             time_values[idx], a, fem_approx, amplitude, duration, rise_time, counts,
#             energy, peak_frequency, rms_voltage, signal_attenuation, delta_P, flow_rate, entropy, growth_rate
#         ])
    
#     return np.array(data)

# # Generate and scale synthetic data
# data = generate_synthetic_data()
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(data)

# # Train FEM approximation model
# fem_inputs = scaled_data[:, 1:4]
# fem_outputs = scaled_data[:, 4]
# gp_model = MLPRegressor(hidden_layer_sizes=(50, 25, 10), max_iter=2000, random_state=42)
# gp_model.fit(fem_inputs, fem_outputs)

# # Plot simulation results
# time_values, crack_sizes, growth_rates = simulate_crack_growth()
# plt.figure(figsize=(10, 5))
# plt.plot(time_values, crack_sizes, label="Crack Size (m)")
# plt.xlabel("Time (s)")
# plt.ylabel("Crack Size (m)")
# plt.legend()
# plt.title("Crack Growth Over Time")
# plt.show()











# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp

# # Seed for reproducibility
# np.random.seed(42)

# # Paris' law parameters
# C0 = 1e-10
# m = 3.0
# sigma = 200
# pi = np.pi

# # Monte Carlo parameters
# num_simulations = 1000  # Number of MC runs
# time_span = (0, 1000)
# time_eval = np.linspace(0, 1000, 500)

# # Function for crack growth with Monte Carlo variability
# def crack_growth_mc(t, a, C, m_exp):
#     return C * (sigma * np.sqrt(pi * a)) ** m_exp

# # Storage for results
# mc_results = np.zeros((num_simulations, len(time_eval)))

# # Run Monte Carlo simulations
# for i in range(num_simulations):
#     C_mc = np.random.lognormal(mean=np.log(C0), sigma=0.2)  # Vary C
#     m_mc = np.random.normal(m, 0.1)  # Vary exponent m
#     a0_mc = np.random.normal(0.005, 0.001)  # Vary initial crack size

#     sol = solve_ivp(crack_growth_mc, time_span, [a0_mc], t_eval=time_eval, args=(C_mc, m_mc), method='RK45')
#     mc_results[i, :] = sol.y[0]

# # Compute mean and variance
# mean_crack_size = np.mean(mc_results, axis=0)
# std_crack_size = np.std(mc_results, axis=0)

# # Plot comparison
# plt.figure(figsize=(10, 5))
# plt.plot(time_eval, mean_crack_size, label="Monte Carlo Mean Crack Size", color='blue')
# plt.fill_between(time_eval, mean_crack_size - std_crack_size, mean_crack_size + std_crack_size, alpha=0.3, color='blue', label="±1 Std Dev")
# plt.xlabel("Time (hours)")
# plt.ylabel("Crack Size (m)")
# plt.title("Crack Growth: Deterministic vs. Monte Carlo Simulation")
# plt.legend()
# plt.grid()
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # Given parameters
# time_hours = np.linspace(0, 1000, 100)  # Time from 0 to 1000 hours
# initial_crack_size = 0.005  # Initial crack size in meters (5 mm)
# growth_rate = 1e-6  # Crack growth rate per hour (example deterministic rate)

# # Deterministic crack growth model (linear example)
# deterministic_crack_size = initial_crack_size + growth_rate * time_hours

# # Monte Carlo simulation (mean and standard deviation from assumed data)
# mc_mean = initial_crack_size + growth_rate * time_hours * (1 + 0.1 * np.sin(time_hours / 200))  # Example variation
# mc_std = 0.001 * (1 + np.exp((time_hours - 800) / 200))  # Increasing uncertainty over time

# # Define critical crack size for failure
# critical_crack_size = 0.01  # 10 mm (example threshold)

# # Probability of failure estimation (assuming normal distribution)
# prob_failure = 1 - np.exp(-((critical_crack_size - mc_mean) / (mc_std + 1e-6))**2)

# # Plot results
# plt.figure(figsize=(10, 5))
# plt.plot(time_hours, deterministic_crack_size, 'r-', label="Deterministic Crack Size")
# plt.plot(time_hours, mc_mean, 'b-', label="Monte Carlo Mean Crack Size")
# plt.fill_between(time_hours, mc_mean - mc_std, mc_mean + mc_std, color='blue', alpha=0.3, label="±1 Std Dev")
# plt.axhline(y=critical_crack_size, color='k', linestyle='--', label="Critical Crack Size")
# plt.xlabel("Time (hours)")
# plt.ylabel("Crack Size (m)")
# plt.title("Crack Growth: Deterministic vs. Monte Carlo Simulation")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Find probability of failure at 1000 hours
# failure_prob_at_1000h = prob_failure[-1]
# failure_prob_at_1000h










import numpy as np
import matplotlib.pyplot as plt

# Given Parameters
t = np.linspace(0, 1000, 100)  # Time from 0 to 1000 hours
a0 = 5e-3  # Initial crack size (5 mm)
C = 1e-10  # Growth rate coefficient
m = 2.5  # Exponent

# Deterministic Crack Growth Model
a_deterministic = a0 + C * t**m

# Monte Carlo Simulation
np.random.seed(42)
n_simulations = 10000
noise_std = 0.001  # Standard deviation for stochastic variations
a_monte_carlo = np.array([a0 + C * t**m + np.random.normal(0, noise_std, len(t)) for _ in range(n_simulations)])

# Compute Monte Carlo Mean and Standard Deviation
a_mc_mean = np.mean(a_monte_carlo, axis=0)
a_mc_std = np.std(a_monte_carlo, axis=0)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(t, a_deterministic, 'r-', label='Deterministic Model')
plt.plot(t, a_mc_mean, 'b-', label='Monte Carlo Mean')
plt.fill_between(t, a_mc_mean - a_mc_std, a_mc_mean + a_mc_std, color='blue', alpha=0.3, label='±1 Std Dev')
plt.xlabel('Time (hours)')
plt.ylabel('Crack Size (m)')
plt.title('Comparison of Deterministic vs. Monte Carlo Model')
plt.legend()
plt.grid()
plt.show()

# Validation Step
max_error = np.max(np.abs(a_deterministic - a_mc_mean))
if max_error < 2 * np.mean(a_mc_std):
    print(f"✅ Model Validation Passed! Max error: {max_error:.6f} is within expected range.")
else:
    print(f"⚠️ Model Validation Warning! Max error: {max_error:.6f} exceeds expected range.")