#problem4A
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# simulate an MA process
def simulate_MA(q, theta, n=1000):
    np.random.seed(42)
    e = np.random.normal(0, 1, n + q)  # Generate white noise
    y = np.array([sum(theta[j] * e[i - (j + 1)] for j in range(q)) + e[i] for i in range(q, n + q)])
    return y

# plot ACF and PACF
def plot_acf_pacf(series, title):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    axes[0].plot(series)
    axes[0].set_title(f"Time Series: {title}")
    
    sm.graphics.tsa.plot_acf(series, lags=30, ax=axes[1])
    axes[1].set_title(f"ACF: {title}")
    
    sm.graphics.tsa.plot_pacf(series, lags=30, ax=axes[2])
    axes[2].set_title(f"PACF: {title}")
    
    plt.tight_layout()
    plt.show()

# Simulate and plot MA(1), MA(2), and MA(3)
theta_values = {
    "MA(1)": [0.5],
    "MA(2)": [0.5, -0.3],
    "MA(3)": [0.5, -0.3, 0.2]
}

for ma_type, theta in theta_values.items():
    plot_acf_pacf(simulate_MA(len(theta), theta), ma_type)

