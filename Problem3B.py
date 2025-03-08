import math
from math import log, sqrt, exp
from scipy.stats import norm

S = 31            # Stock price
K = 30            # Strike price
T = 0.25          # Time to maturity (in years)
r = 0.10          # Risk-free rate
sigma = 0.33      # Implied volatility

# Calculate d1 and d2 for the Black-Scholes formula
d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)

# Compute the Greeks
Delta = norm.cdf(d1)                                
Vega = S * sqrt(T) * norm.pdf(d1)                   
Theta = -(S * norm.pdf(d1) * sigma) / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(d2)

# For a 1% increase in volatility, the price change is approximately:
price_change = Vega * 0.01

print("Delta: {:.4f}".format(Delta))
print("Vega: {:.4f}".format(Vega))
print("Theta (per year): {:.4f}".format(Theta))
print("If implied volatility increases by 1% (0.01), the option price increases by about ${:.4f}".format(price_change))