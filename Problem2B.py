import pandas as pd
import numpy as np
import math
from scipy.stats import norm, t

# Load daily prices from CSV 
df = pd.read_csv("C:/Users/Eric Liu/OneDrive/Documents/Fintech545Project2/DailyPrices.csv", parse_dates=["Date"], index_col="Date")
returns = df.pct_change().dropna()

# Define Portfolio Positions

shares = {'SPY': 100, 'AAPL': 200, 'EQIX': 150}
latest_prices = df.iloc[-1]
position_values = {stock: shares[stock] * latest_prices[stock] for stock in shares}
portfolio_value = sum(position_values.values())

alpha = 0.05
lambda_val = 0.97

# Method (a): Normal Distribution with EWMA Covariance

def ewma_std(series, lambda_val=0.97):
    T = len(series)
    weights = np.array([ (1 - lambda_val) * (lambda_val ** (T - i - 1)) for i in range(T) ])
    var = np.sum(weights * series**2) / np.sum(weights)
    return math.sqrt(var)


ewma_std_dict = {}
for stock in shares:
    series = returns[stock].values
    ewma_std_dict[stock] = ewma_std(series, lambda_val)


z = norm.ppf(alpha)  
# Compute individual VaR and ES (in dollars) for each stock
VaR_normal = {}
ES_normal = {}
for stock in shares:
    sigma = ewma_std_dict[stock]
    pos_val = position_values[stock]
    VaR_normal[stock] = - pos_val * sigma * z
    ES_normal[stock] = pos_val * sigma * norm.pdf(z) / alpha

# For the portfolio, here is EWMA covariance matrix.
def ewma_cov_matrix(df_returns, lambda_val=0.97):
    T = df_returns.shape[0]
    weights = np.array([ (1 - lambda_val) * (lambda_val ** (T - i - 1)) for i in range(T) ])
    weights = weights / weights.sum()  
    demeaned = df_returns - 0
    cov = np.dot((demeaned * weights[:, None]).T, demeaned)
    return cov

# Compute EWMA covariance matrix for our three stocks
cov_matrix = ewma_cov_matrix(returns[['SPY', 'AAPL', 'EQIX']], lambda_val)

# In dollar terms, the portfolio’s daily P&L is given by:
#   P&L = diag(position_values) * returns   (i.e. each stock’s P&L = position * return)
# Thus, portfolio variance = P^T * Cov * P, where P is the vector of dollar positions.
P_vec = np.array([position_values['SPY'], position_values['AAPL'], position_values['EQIX']])
portfolio_std_normal = math.sqrt(np.dot(P_vec, np.dot(cov_matrix, P_vec)))
VaR_normal_portfolio = - portfolio_std_normal * z
ES_normal_portfolio = portfolio_std_normal * norm.pdf(z) / alpha


# Method (b): T Distribution Using a Gaussian Copula

nu = 4  

# For a t distribution, to match the standard deviation sigma, we adjust the quantiles.
def t_quantile(alpha, nu):
    return t.ppf(alpha, nu) / math.sqrt(nu/(nu-2))

def t_ES(alpha, nu):
    # Expected Shortfall for the standardized t distribution:
    tq = t.ppf(alpha, nu)
    # The formula for ES (for a t distribution) is:
    return t.pdf(tq, nu) * (nu + tq**2) / ((nu - 1) * alpha) / math.sqrt(nu/(nu-2))

# Compute adjusted quantile and ES factors for the t distribution
t_q = t_quantile(alpha, nu)  # This is negative (e.g., around -2.1 for nu=4)
t_es_factor = t_ES(alpha, nu)

# Compute individual VaR and ES for each stock using t distribution
VaR_t = {}
ES_t = {}
for stock in shares:
    sigma = ewma_std_dict[stock]
    pos_val = position_values[stock]
    VaR_t[stock] = - pos_val * sigma * t_q
    ES_t[stock] = pos_val * sigma * t_es_factor

# For the portfolio, use the same portfolio_std_normal computed earlier.
VaR_t_portfolio = - portfolio_std_normal * t_q
ES_t_portfolio = portfolio_std_normal * t_es_factor

# Method (c): Historical Simulation Using Full History

# For historical simulation we use the actual historical daily returns.
hist_VaR = {}
hist_ES = {}
for stock in shares:
    pnl = position_values[stock] * returns[stock]
    # 5th percentile loss:
    var_level = np.percentile(pnl, 5)
    hist_VaR[stock] = - var_level
    # ES is the average loss for days where loss exceeds the VaR threshold.
    hist_ES[stock] = - pnl[pnl <= var_level].mean()

# For the portfolio, compute daily portfolio P&L as the sum across stocks.
pnl_portfolio = (returns[['SPY','AAPL','EQIX']] * pd.Series(position_values)).sum(axis=1)
var_level_port = np.percentile(pnl_portfolio, 5)
hist_VaR_portfolio = - var_level_port
hist_ES_portfolio = - pnl_portfolio[pnl_portfolio <= var_level_port].mean()

# Display the Results

print("Method (a): Normal Distribution with EWMA")
print("Individual Stocks:")
for stock in shares:
    print(f" {stock}: VaR = ${VaR_normal[stock]:.2f}, ES = ${ES_normal[stock]:.2f}")
print("Portfolio:")
print(f" VaR = ${VaR_normal_portfolio:.2f}, ES = ${ES_normal_portfolio:.2f}")

print("\nMethod (b): T Distribution (nu={nu}) via Gaussian Copula".format(nu=nu))
print("Individual Stocks:")
for stock in shares:
    print(f" {stock}: VaR = ${VaR_t[stock]:.2f}, ES = ${ES_t[stock]:.2f}")
print("Portfolio:")
print(f" VaR = ${VaR_t_portfolio:.2f}, ES = ${ES_t_portfolio:.2f}")

print("\nMethod (c): Historical Simulation")
print("Individual Stocks:")
for stock in shares:
    print(f" {stock}: VaR = ${hist_VaR[stock]:.2f}, ES = ${hist_ES[stock]:.2f}")
print("Portfolio:")
print(f" VaR = ${hist_VaR_portfolio:.2f}, ES = ${hist_ES_portfolio:.2f}")