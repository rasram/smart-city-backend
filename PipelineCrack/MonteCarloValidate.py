import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

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

def theoretical_crack_growth(t_values, a0=0.005, C=1e-10, m=3.0, sigma=200, pi=np.pi):
    return a0 * (1 + C * (sigma * np.sqrt(pi * a0)) ** m * t_values)

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

def validate_model():
    t_values, crack_sizes, _ = simulate_crack_growth()
    theoretical_sizes = theoretical_crack_growth(t_values)
    rmse = np.sqrt(mean_squared_error(theoretical_sizes, crack_sizes))
    r2 = r2_score(theoretical_sizes, crack_sizes)
    print(f"Validation Metrics: RMSE={rmse:.6f}, R²={r2:.6f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(t_values, crack_sizes, label='Simulated Crack Growth', linestyle='dashed')
    plt.plot(t_values, theoretical_sizes, label='Theoretical Prediction', linestyle='solid')
    plt.xlabel('Time (hours)')
    plt.ylabel('Crack Size (m)')
    plt.title('Crack Growth Validation')
    plt.legend()
    plt.show()

validate_model()

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

def generate_synthetic_data(N=500):
    data = []
    time_values, crack_sizes, growth_rates = simulate_crack_growth()
    
    for i in range(N):
        idx = i % len(crack_sizes)
        a = crack_sizes[idx]
        fem_approx = predict_fem(a)
        delta_P, flow_rate = leakage_pressure_drop(a)
        entropy = np.random.uniform(0.1, 1.0)
        data.append([time_values[idx], a, fem_approx, delta_P, flow_rate, entropy, growth_rates[idx]])
    
    return np.array(data)

data = generate_synthetic_data()
dataset = pd.DataFrame(data, columns=["Time (hours)", "CrackSize", "FEM_Approx", "PressureDrop", "FlowRate", "SpectralEntropy", "GrowthRate"])
dataset.to_csv("validated_crack_growth_data.csv", index=False)

print("Validation and Data Generation Complete!")