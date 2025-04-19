import pandas as pd
import numpy as np
from scipy.stats import norm, multivariate_normal

# === Load Data ===
prices = pd.read_csv("C:/Users/Eric Liu/OneDrive/Documents/Fintech545Final Project/DailyPrices Final Project.csv", parse_dates=["Date"], index_col="Date")
portfolio = pd.read_csv("C:/Users/Eric Liu/OneDrive/Documents/Fintech545Final Project/initial_portfolio.csv")
rf = pd.read_csv("C:/Users/Eric Liu/OneDrive/Documents/Fintech545Final Project/rf.csv", parse_dates=["Date"], index_col="Date")

# === Compute Returns and Excess Returns ===
returns = prices.pct_change().dropna()
rf_aligned = rf.loc[returns.index].squeeze()
excess_returns = returns.sub(rf_aligned, axis=0)

# === Get Portfolio Weights at the Cutoff Date ===
cutoff_date = prices[prices.index.year == 2023].index.max()
start_prices = prices.loc[cutoff_date]

def get_weights(portfolio_group):
    symbols = portfolio_group['Symbol']
    holdings = portfolio_group['Holding'].values
    start_values = start_prices[symbols].values * holdings
    weights = start_values / np.sum(start_values)
    return symbols.tolist(), weights

# === Generate Weights for All Portfolios ===
portfolio_weights = {}
for name, group in portfolio.groupby('Portfolio'):
    symbols, weights = get_weights(group)
    portfolio_weights[name] = {'symbols': symbols, 'weights': weights}

# Add total portfolio
symbols, weights = get_weights(portfolio)
portfolio_weights['Total'] = {'symbols': symbols, 'weights': weights}

# === Gaussian Copula Simulation (Manual) ===
def gaussian_copula_simulation(data, weights, alpha=0.05, samples=10000):
    u = data.rank(method="average") / (len(data) + 1)
    u = u.fillna(0)
    z = pd.DataFrame(norm.ppf(u), index=u.index, columns=u.columns)
    z = z.fillna(0).values
    
    corr = np.corrcoef(z.T)
    mvn_samples = np.random.multivariate_normal(mean=np.zeros(z.shape[1]), cov=corr, size=samples)

    sorted_data = np.sort(data.values, axis=0)
    idx = (norm.cdf(mvn_samples) * (len(data) - 1)).astype(int)
    synthetic = np.take_along_axis(sorted_data, idx, axis=0)

    portfolio_samples = synthetic.dot(weights)
    var = np.quantile(portfolio_samples, alpha)
    es = portfolio_samples[portfolio_samples <= var].mean()
    return var, es

# === Compute Portfolio Returns ===
def compute_portfolio_returns(symbols, weights):
    return excess_returns[symbols].dot(weights)

# === Run Analysis ===
copula_results = []
for name, info in portfolio_weights.items():
    symbols, weights = info['symbols'], info['weights']
    port_returns = compute_portfolio_returns(symbols, weights)

    # MVN
    mean_vec = excess_returns[symbols].mean().values
    cov_mat = excess_returns[symbols].cov().values
    mvn_samples = multivariate_normal.rvs(mean=mean_vec, cov=cov_mat, size=10000)
    mvn_port_returns = mvn_samples.dot(weights)
    mvn_var = np.quantile(mvn_port_returns, 0.05)
    mvn_es = mvn_port_returns[mvn_port_returns <= mvn_var].mean()

    # Copula
    copula_var, copula_es = gaussian_copula_simulation(excess_returns[symbols], weights)

    copula_results.append({
        "Portfolio": name,
        "VaR_MVN (5%)": -mvn_var,
        "ES_MVN (5%)": -mvn_es,
        "VaR_Copula (5%)": -copula_var,
        "ES_Copula (5%)": -copula_es
    })

copula_results_df = pd.DataFrame(copula_results)
print("\n=== 1-Day VaR and ES (MVN vs Approx. Copula) ===")
print(copula_results_df)