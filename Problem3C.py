import math
from math import log, sqrt, exp
from scipy.stats import norm

S = 31             # Stock price
K = 30             # Strike price
T = 0.25           # Time to maturity (in years)
r = 0.10           # Risk-free rate
C_market = 3.00    # Market call option price
sigma = 0.33 

# Compute d1 and d2 for the Black-Scholes formulas
d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)

# Calculate the put price using the Black-Scholes formula for a put: P = K * exp(-r*T) * N(-d2) - S * N(-d1)
P_BS = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Calculate the put price using Put-Call parity: P = C - S + K * exp(-r*T)
P_PC = C_market - S + K * exp(-r * T)

print("Put price using Black-Scholes (GBSM): ${:.2f}".format(P_BS))
print("Put price using Put-Call Parity:   ${:.2f}".format(P_PC))