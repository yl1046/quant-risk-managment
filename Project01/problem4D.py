from statsmodels.tsa.arima.model import ARIMA
import pandas as pd 
results = []
filepath = "C:/Users/Eric Liu/OneDrive/Documents/Fintech545Project1/problem4.csv"
data = pd.read_csv(filepath)
y_series = data['y'].values
for p in range(4):
    for q in range(4):
        try:
            model = ARIMA(y_series, order=(p, 0, q)).fit()
            k, n = len(model.params), len(y_series)
            aicc = model.aic + (2 * k * (k + 1)) / (n - k - 1)
            results.append(((p, q), aicc))
        except:
            continue

# Find the best model
best_order, best_aicc = min(results, key=lambda x: x[1])
aicc_df = pd.DataFrame(results, columns=["(p, q)", "AICc"]).sort_values(by="AICc")
best_order, best_aicc
