#problem4B
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

filepath = "C:/Users/Eric Liu/OneDrive/Documents/Fintech545Project1/problem4.csv"
data = pd.read_csv(filepath)
y_series = data['y'].values

#plot ACF and PACF
def plot_acf_pacf(series, title):
    fig, axes = plt.subplots(3, 1, figsize=(8, 6))
    axes[0].plot(series)
    axes[0].set_title(title)
    sm.graphics.tsa.plot_acf(series, lags=20, ax=axes[1])
    sm.graphics.tsa.plot_pacf(series, lags=20, ax=axes[2])
    plt.tight_layout()
    plt.show()

plot_acf_pacf(y_series, "Given Data ACF & PACF")

#AR(1), AR(2), and AR(3) 
from statsmodels.tsa.arima.model import ARIMA
ar_orders = [1, 2, 3]
simulated_series = {}
for order in ar_orders:
    model = ARIMA(y_series, order=(order, 0, 0))
    fitted_model = model.fit()
    simulated_series[order] = fitted_model.simulate(nsimulations=len(y_series))

    # Plot ACF and PACF for the simulated series
    plot_acf_pacf(simulated_series[order], f"Simulated AR({order}) Process")