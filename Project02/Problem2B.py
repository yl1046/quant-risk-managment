import numpy as np
import pandas as pd
from scipy.stats import norm, t

# Extract price and the three assets
prices = pd.read_csv("C:/Users/Eric Liu/OneDrive/Documents/Fintech545Project2/DailyPrices.csv", parse_dates=["Date"]).set_index("Date")
assets = ["SPY", "AAPL", "EQIX"]
prices = prices[assets]

# Calculate daily arithmetic returns
returns = prices.pct_change().dropna()

# Portfolio positions (number of shares)
shares = {"SPY": 100, "AAPL": 200, "EQIX": 150}
current_prices = prices.iloc[-1]
total_value_for_each_asset = {asset: shares[asset] * current_prices[asset] for asset in assets}

# Exponentially weighted weights with lambda=0.97
lam = 0.97
T = len(returns)
weights = lam ** np.arange(T - 1, -1, -1)
weights = weights / weights.sum()

# compute EW volatility (sigma) for an asset
def ew_sigma(asset_returns):
    return np.sqrt(np.sum(weights * asset_returns**2))

# Method (a): Normal Model
z = norm.ppf(0.05) 
norm_results = {}
for asset in assets:
    sigma = ew_sigma(returns[asset])
    # Dollar VaR: current price * sigma 
    VaR = -z * sigma * current_prices[asset]
    ES = sigma * norm.pdf(z) / 0.05 * current_prices[asset]
    norm_results[asset] = {"VaR": VaR, "ES": ES}

# For the portfolio, compute an EW covariance matrix
cov_ew = pd.DataFrame(
    {a: [np.sum(weights * returns[a] * returns[b]) for b in assets] for a in assets},
    index=assets
)
exposures = np.array([total_value_for_each_asset[a] for a in assets])
port_sigma = np.sqrt(exposures @ cov_ew.values @ exposures)
norm_port = {"VaR": -z * port_sigma, "ES": port_sigma * norm.pdf(z) / 0.05}

# Method (b): T Distribution via Gaussian Copula 
nu = 4
t_results = {}
for asset in assets:
    sigma = ew_sigma(returns[asset])
    scale = sigma * np.sqrt((nu - 2) / nu)
    t_q = t.ppf(0.05, nu)
    VaR = -t_q * scale * current_prices[asset]
    ES = scale * ((nu + t_q**2) / (nu - 1)) * (t.pdf(t_q, nu) / 0.05) * current_prices[asset]
    t_results[asset] = {"VaR": VaR, "ES": ES}

# Portfolio for T model:
port_sigma_t = np.sqrt(exposures @ cov_ew.values @ exposures)
scale_port = port_sigma_t * np.sqrt((nu - 2) / nu)
t_q_port = t.ppf(0.05, nu)
t_port = {"VaR": -t_q_port * scale_port,
          "ES": scale_port * ((nu + t_q_port**2) / (nu - 1)) * (t.pdf(t_q_port, nu) / 0.05)}

# Method (c): Historical Simulation 
hist_results = {}
for asset in assets:
    pnl = returns[asset] * current_prices[asset]
    var_hist = -np.percentile(pnl, 5)
    es_hist = -pnl[pnl <= np.percentile(pnl, 5)].mean()
    hist_results[asset] = {"VaR": var_hist, "ES": es_hist}

# Portfolio historical simulation: sum individual dollar pnl's
pnl_port = returns.multiply(pd.Series(current_prices))\
                   .multiply(pd.Series(shares))\
                   .sum(axis=1)
hist_port = {"VaR": -np.percentile(pnl_port, 5),
             "ES": -pnl_port[pnl_port <= np.percentile(pnl_port, 5)].mean()}

# --- Print Results ---
def print_results(method, ind, port):
    print(f"--- {method} ---")
    for asset in assets:
        print(f"{asset}: VaR = {ind[asset]['VaR']:.2f}, ES = {ind[asset]['ES']:.2f}")
    print(f"Portfolio: VaR = {port['VaR']:.2f}, ES = {port['ES']:.2f}\n")

print_results("Normal EW", norm_results, norm_port)
print_results("T Copula (nu=4)", t_results, t_port)
print_results("Historical", hist_results, hist_port)
