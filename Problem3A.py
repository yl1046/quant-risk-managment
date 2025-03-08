import math
from scipy.stats import norm
from scipy.optimize import brentq

def bs_call(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

S = 31            # Stock price
K = 30            # Strike price
T = 0.25          # Time to maturity (3 months)
r = 0.10          # Risk-free rate
market_price = 3.0  # Market call option price

# Find the implied volatility that makes the BS call price equal to the market price
implied_vol = brentq(lambda sigma: bs_call(S, K, T, r, sigma) - market_price, 0.01, 2.0)

print("Implied Volatility: {:.2%}".format(implied_vol))
