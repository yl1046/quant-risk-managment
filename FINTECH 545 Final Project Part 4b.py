import pandas as pd
import numpy as np
from scipy.stats import norm, gennorm, skewnorm, genhyperbolic
from scipy.stats import rankdata

# Load updated files
prices = pd.read_csv("C:/Users/Eric Liu/OneDrive/Documents/Fintech545Final Project/DailyPrices Final Project.csv", parse_dates=["Date"], index_col="Date")
portfolio = pd.read_csv("C:/Users/Eric Liu/OneDrive/Documents/Fintech545Final Project/initial_portfolio.csv")
rf = pd.read_csv("C:/Users/Eric Liu/OneDrive/Documents/Fintech545Final Project/rf.csv", parse_dates=["Date"], index_col="Date")

# Prepare excess return data (training only: data before 2024)
cutoff_date = prices[prices.index.year == 2023].index.max()
returns = prices.pct_change().dropna()
train_returns = returns[returns.index <= cutoff_date]
train_rf = rf.loc[train_returns.index].squeeze()
excess_returns = train_returns.sub(train_rf, axis=0)

# Assume mean return is 0% and fit models accordingly
def fit_risk_models(series, max_samples=500):
    x = series.dropna()
    if len(x) > max_samples:
        x = x.sample(max_samples, random_state=1)

    scores = {}

    # Normal
    norm_params = (0.0, np.std(x))  # zero mean
    norm_ll = np.sum(norm.logpdf(x, *norm_params))
    scores['Normal'] = (norm_ll, norm_params)

    # Skew Normal
    try:
        sn_params = skewnorm.fit(x, floc=0)  # force location = 0
        sn_ll = np.sum(skewnorm.logpdf(x, *sn_params))
        scores['SkewNormal'] = (sn_ll, sn_params)
    except Exception:
        scores['SkewNormal'] = (-np.inf, None)

    # Generalized Normal (proxy for Generalized T)
    try:
        gn_params = gennorm.fit(x, floc=0)  # fix mean = 0
        gn_ll = np.sum(gennorm.logpdf(x, *gn_params))
        scores['GenNorm'] = (gn_ll, gn_params)
    except Exception:
        scores['GenNorm'] = (-np.inf, None)

    # Normal Inverse Gaussian (via genhyperbolic with λ = -0.5)
    try:
        nig_params = genhyperbolic.fit(x, lambda_=-0.5, floc=0)  # fix location = 0
        nig_ll = np.sum(genhyperbolic.logpdf(x, -0.5, *nig_params))
        scores['NIG'] = (nig_ll, (-0.5, *nig_params))
    except Exception:
        scores['NIG'] = (-np.inf, None)

    best_fit = max(scores.items(), key=lambda kv: kv[1][0])  # Max log-likelihood
    return best_fit[0], best_fit[1][1]

# Apply to all stocks except SPY
fit_results = {
    symbol: fit_risk_models(excess_returns[symbol])
    for symbol in excess_returns.columns if symbol != "SPY"
}

fit_df = pd.DataFrame.from_dict(fit_results, orient='index', columns=['BestFit', 'Parameters'])
print(fit_df.to_string())