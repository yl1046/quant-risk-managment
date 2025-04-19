import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t

# === Load data ===
def read_input_data():
    prices = pd.read_csv("C:/Users/Eric Liu/OneDrive/Documents/Fintech545Final Project/DailyPrices Final Project.csv", 
                         parse_dates=["Date"], index_col="Date")
    portfolio = pd.read_csv("C:/Users/Eric Liu/OneDrive/Documents/Fintech545Final Project/initial_portfolio.csv")
    rf = pd.read_csv("C:/Users/Eric Liu/OneDrive/Documents/Fintech545Final Project/rf.csv", 
                     parse_dates=["Date"], index_col="Date")
    return prices, portfolio, rf

# === Compute excess returns before cutoff ===
def compute_excess_returns(prices, rf, cutoff_date):
    train = prices[prices.index <= cutoff_date].pct_change().dropna()
    test = prices[prices.index > cutoff_date].pct_change().dropna()
    train_rf = rf.loc[train.index].squeeze()
    test_rf = rf.loc[test.index].squeeze()
    train_excess = train.sub(train_rf, axis=0)
    return train_excess, test, test_rf

# === Fit CAPM model (alpha, beta) ===
def fit_capm_model(excess_returns):
    market = excess_returns["SPY"]
    coefficients = {}
    for symbol in excess_returns.columns:
        if symbol == "SPY":
            coefficients[symbol] = {"beta": 1.0, "alpha": 0.0}
            continue
        data = pd.concat([market, excess_returns[symbol]], axis=1).dropna()
        if len(data) > 1:
            slope, intercept = stats.linregress(data.iloc[:, 0], data.iloc[:, 1])[:2]
            coefficients[symbol] = {"beta": slope, "alpha": intercept}
    return coefficients

# === Compute portfolio total return ===
def compute_total_return(portfolio, start_prices, end_prices):
    initial_value = final_value = 0.0
    for _, row in portfolio.iterrows():
        symbol = row["Symbol"]
        if symbol in start_prices and symbol in end_prices:
            quantity = row["Holding"]
            initial_value += quantity * start_prices[symbol]
            final_value += quantity * end_prices[symbol]
    return (final_value - initial_value) / initial_value if initial_value > 0 else 0

# === t-distribution ES estimation ===
def t_distribution_es(series, alpha=0.05):
    x = series.dropna()
    try:
        df, loc, scale = t.fit(x)
        var = t.ppf(alpha, df, loc=loc, scale=scale)
        es = t.expect(lambda r: r, args=(df,), loc=loc, scale=scale, lb=-np.inf, ub=var) / alpha
        return -es
    except:
        return np.nan

# === Risk parity weighting ===
def risk_parity_weights(risks):
    inv_risk = 1.0 / risks
    weights = inv_risk / inv_risk.sum()
    return weights

# === Portfolio weights based on prices and holdings ===
def get_portfolio_weights(portfolio, prices):
    symbols = portfolio["Symbol"]
    holdings = portfolio["Holding"].values
    values = prices[symbols].values * holdings
    return values / values.sum()

# === Return Attribution ===
def attribute_return(weights, returns, market_returns, betas):
    n = len(returns)
    p_return = np.zeros(n)
    factor_exposure = np.zeros(n)
    residuals = np.zeros(n)
    current_weights = weights.copy()
    for t in range(n):
        factor_exposure[t] = np.dot(betas, current_weights)
        current_weights *= (1 + returns.iloc[t])
        total_value = current_weights.sum()
        current_weights /= total_value
        p_return[t] = total_value - 1
        residuals[t] = p_return[t] - factor_exposure[t] * market_returns.iloc[t]
    total_ret = np.exp(np.sum(np.log(p_return + 1))) - 1
    k = np.log1p(total_ret) / total_ret
    adjusted_weights = np.log1p(p_return) / p_return / k
    market_component = np.sum(market_returns * factor_exposure * adjusted_weights)
    alpha_component = np.sum(residuals * adjusted_weights)
    return market_component, alpha_component

# === Volatility Attribution ===
def attribute_volatility(weights, returns, market_returns, betas):
    n = len(returns)
    p_return = np.zeros(n)
    factor_exposure = np.zeros(n)
    residuals = np.zeros(n)
    current_weights = weights.copy()
    for t in range(n):
        factor_exposure[t] = np.dot(betas, current_weights)
        current_weights *= (1 + returns.iloc[t])
        total_value = current_weights.sum()
        current_weights /= total_value
        p_return[t] = total_value - 1
        residuals[t] = p_return[t] - factor_exposure[t] * market_returns.iloc[t]
    return np.std(factor_exposure * market_returns), np.std(residuals), np.std(p_return)

# === Print Attribution Output ===
def _analyze_group(pf, test_returns, start_prices, end_prices, capm, spy_return):
    total_ret = compute_total_return(pf, start_prices, end_prices)
    weights = get_portfolio_weights(pf, start_prices)
    symbols = pf["Symbol"]
    betas = np.array([capm[s]["beta"] for s in symbols])
    mkt_attr, alpha_attr = attribute_return(weights, test_returns[symbols], test_returns["SPY"], betas)
    mkt_vol, alpha_vol, port_vol = attribute_volatility(weights, test_returns[symbols], test_returns["SPY"], betas)
    print("# 3x4 DataFrame")
    print("#", "-" * 70)
    print(f"#  Row | Value               {'SPY':>15}    {'Alpha':>10}    {'Portfolio':>10}")
    print("#", "-" * 70)
    print(f"#  1   | TotalReturn         {spy_return:15.8f}    {total_ret - spy_return:10.8f}    {total_ret:10.8f}")
    print(f"#  2   | Return Attribution  {mkt_attr:15.8f}    {alpha_attr:10.8f}    {total_ret:10.8f}")
    print(f"#  3   | Vol Attribution     {mkt_vol:15.8f}    {alpha_vol:10.8f}    {port_vol:10.8f}")

# === Full Analysis ===
def analyze_portfolio(prices, portfolio, rf):
    cutoff_date = prices[prices.index.year == 2023].index.max()
    train_excess, test_returns, test_rf = compute_excess_returns(prices, rf, cutoff_date)
    capm = fit_capm_model(train_excess)
    start_prices = prices.loc[cutoff_date]
    end_prices = prices.loc[test_returns.index[-1]]
    spy_return = (end_prices["SPY"] - start_prices["SPY"]) / start_prices["SPY"]
    print("\n=== Total Portfolio Attribution ===")
    _analyze_group(portfolio, test_returns, start_prices, end_prices, capm, spy_return)
    for group_name, group_data in portfolio.groupby("Portfolio"):
        print(f"\n=== {group_name} Portfolio Attribution ===")
        _analyze_group(group_data, test_returns, start_prices, end_prices, capm, spy_return)

# === MAIN EXECUTION ===
if __name__ == "__main__":
    prices, portfolio, rf = read_input_data()
    
    # Compute excess returns before 2024
    cutoff_date = prices[prices.index.year == 2023].index.max()
    returns = prices.pct_change().dropna()
    train_returns = returns[returns.index <= cutoff_date]
    train_rf = rf.loc[train_returns.index].squeeze()
    excess_returns = train_returns.sub(train_rf, axis=0)

    # Replace holdings using risk-parity weights based on ES
    new_portfolio_rows = []
    for portfolio_name, group in portfolio.groupby("Portfolio"):
        symbols = group["Symbol"]
        es_values = {
            symbol: t_distribution_es(excess_returns[symbol])
            for symbol in symbols if symbol in excess_returns.columns
        }
        es_series = pd.Series(es_values).dropna()
        if not es_series.empty:
            weights = risk_parity_weights(es_series)
            for symbol, weight in weights.items():
                new_portfolio_rows.append({
                    "Symbol": symbol,
                    "Portfolio": portfolio_name,
                    "Holding": weight
                })
    portfolio = pd.DataFrame(new_portfolio_rows)

    # Run attribution with updated portfolio
    analyze_portfolio(prices, portfolio, rf)
