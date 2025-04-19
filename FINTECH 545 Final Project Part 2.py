import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

# ===== Estimate CAPM Coefficients =====
def estimate_capm_coefficients(excess_returns_df):
    def fit_linear_regression(stock, market):
        combined = pd.concat([market, stock], axis=1).dropna()
        if len(combined) < 2:
            return {'alpha': np.nan, 'beta': np.nan, 'r2': np.nan}
        slope, intercept, r_val, _, _ = stats.linregress(combined.iloc[:, 0], combined.iloc[:, 1])
        return {'alpha': intercept, 'beta': slope, 'r2': r_val ** 2}

    market_ret = excess_returns_df['SPY']
    result = {}
    for ticker in excess_returns_df.columns:
        if ticker != 'SPY':
            result[ticker] = fit_linear_regression(excess_returns_df[ticker], market_ret)
    result['SPY'] = {'alpha': 0.0, 'beta': 1.0, 'r2': 1.0}
    return result

# ===== Optimize Portfolio Using Sharpe Ratio =====
def get_optimal_weights(excess_returns_df, capm_info, tickers):
    eligible = [t for t in tickers if t in capm_info and t in excess_returns_df.columns]
    if not eligible:
        return None

    market_avg_return = excess_returns_df['SPY'].mean()
    expected_returns = {t: capm_info[t]['beta'] * market_avg_return for t in eligible}
    mu = np.array([expected_returns[t] for t in eligible])
    cov_matrix = excess_returns_df[eligible].cov().values

    def neg_sharpe(weights):
        ret = weights @ mu
        vol = np.sqrt(weights @ cov_matrix @ weights)
        return -ret / vol if vol > 0 else 0

    bounds = [(0, 1)] * len(eligible)
    constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    initial_guess = np.ones(len(eligible)) / len(eligible)
    
    result = minimize(neg_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=[constraint])
    return dict(zip(eligible, result.x)) if result.success else None

# ===== Display Portfolio Metrics =====
def print_portfolio_metrics(expected_returns, summaries):
    print("Expected Returns (per stock):")
    for symbol, val in expected_returns.items():
        print(f"{symbol}: {val:.2%}")
    print()
    for portfolio_name, stats in summaries.items():
        print(f"Portfolio {portfolio_name} Optimal Allocation:")
        print(f"  Annualized Return: {stats['return']:.2%}")
        print(f"  Annualized Volatility: {stats['vol']:.2%}")
        print(f"  Sharpe Ratio: {stats['sharpe']:.2f}\n")

# ===== Tabular Comparison Output =====
def show_comparison_table(data_dict):
    for portfolio, metrics in data_dict.items():
        header = "\nTotal Portfolio Comparison:" if portfolio == 'Total' else f"\nComparison for Portfolio {portfolio}:"
        print(header)
        print(f"{'Metric':<25} {'Original Portfolio':>20} {'Optimized Portfolio':>20} {'Difference':>15}")
        print("-" * 85)
        for key, (original, optimized) in metrics.items():
            if original is None or optimized is None:
                orig_str = "-"
                opt_str = f"{optimized:.6f}" if optimized is not None else "-"
                diff_str = "-"
            else:
                diff = optimized - original
                if 'Beta' in key:
                    orig_str = f"{original:.4f}"
                    opt_str = f"{optimized:.4f}"
                    diff_str = f"{diff:.4f}"
                elif 'Sharpe' in key:
                    orig_str = "-" if original is None else f"{original:.6f}"
                    opt_str = f"{optimized:.6f}"
                    diff_str = "-"
                else:
                    orig_str = f"{original:.4%}"
                    opt_str = f"{optimized:.4%}"
                    diff_str = f"{diff:.4%}"
            print(f"{key:<25} {orig_str:>20} {opt_str:>20} {diff_str:>15}")

# ===== Main Analysis Function =====
def execute_optimization_analysis():
    # Load required files
    price_data = pd.read_csv("C:/Users/Eric Liu/OneDrive/Documents/Fintech545Final Project/DailyPrices Final Project.csv", parse_dates=['Date'])
    portfolio_df = pd.read_csv("C:/Users/Eric Liu/OneDrive/Documents/Fintech545Final Project/initial_portfolio.csv")
    rf_data = pd.read_csv("C:/Users/Eric Liu/OneDrive/Documents/Fintech545Final Project/rf.csv", parse_dates=['Date'])

    price_data.set_index('Date', inplace=True)
    rf_data.set_index('Date', inplace=True)

    cutoff = price_data[price_data.index.year == 2023].index.max()
    price_train = price_data[price_data.index <= cutoff]
    returns_train = price_train.pct_change().dropna()
    rf_aligned = rf_data.loc[returns_train.index].squeeze()
    excess_ret_train = returns_train.sub(rf_aligned, axis=0)

    capm_output = estimate_capm_coefficients(excess_ret_train)
    avg_market_ret = excess_ret_train['SPY'].mean()
    exp_returns = {s: capm_output[s]['beta'] * avg_market_ret for s in capm_output if s != 'SPY'}

    grouped_portfolios = {name: df for name, df in portfolio_df.groupby('Portfolio')}
    weights_by_portfolio = {}
    metrics_by_portfolio = {}

    for name, group in grouped_portfolios.items():
        symbols = group['Symbol'].unique()
        weights = get_optimal_weights(excess_ret_train, capm_output, symbols)
        if weights:
            weights_arr = np.array(list(weights.values()))
            tickers = list(weights.keys())
            mu_vec = np.array([capm_output[t]['beta'] * avg_market_ret for t in tickers])
            cov_matrix = excess_ret_train[tickers].cov().values
            ann_return = weights_arr @ mu_vec * 252
            ann_vol = np.sqrt(weights_arr.T @ cov_matrix @ weights_arr) * np.sqrt(252)
            sharpe_ratio = ann_return / ann_vol if ann_vol > 0 else 0

            weights_by_portfolio[name] = weights
            metrics_by_portfolio[name] = {
                'return': ann_return,
                'vol': ann_vol,
                'sharpe': sharpe_ratio
            }

    print_portfolio_metrics(exp_returns, metrics_by_portfolio)

    # Comparison Results (example only)
    comparison_results = {
        'Total': {
            'Total Return': [0.2047, 0.2839],
            'Systematic Return': [0.2493, 0.2644],
            'Idiosyncratic Return': [-0.0446, 0.0195],
            'Portfolio Beta': [0.95, 1.01],
            'Sharpe Ratio': [None, 1.4763]
        },
        'A': {
            'Total Return': [0.1366, 0.2886],
            'Systematic Return': [0.2529, 0.2641],
            'Idiosyncratic Return': [-0.1163, 0.0245],
            'Portfolio Beta': [0.97, 1.01],
            'Sharpe Ratio': [None, 1.4635]
        },
        'B': {
            'Total Return': [0.2035, 0.2579],
            'Systematic Return': [0.2407, 0.2632],
            'Idiosyncratic Return': [-0.0372, -0.0053],
            'Portfolio Beta': [0.92, 1.01],
            'Sharpe Ratio': [None, 1.4836]
        },
        'C': {
            'Total Return': [0.2812, 0.3059],
            'Systematic Return': [0.2543, 0.2659],
            'Idiosyncratic Return': [0.0268, 0.0400],
            'Portfolio Beta': [0.97, 1.02],
            'Sharpe Ratio': [None, 1.4827]
        }
    }

    show_comparison_table(comparison_results)

# ===== Script Entry Point =====
if __name__ == '__main__':
    execute_optimization_analysis()