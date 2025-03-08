import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes call and put pricing
def bs_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    # Use put-call parity: P = C - S + K*exp(-rT)
    return bs_call(S, K, T, r, sigma) - S + K*math.exp(-r*T)


S0 = 31.0        # Stock price
K = 30.0         # Strike price
T0 = 0.25        # Time to maturity 
r = 0.10         # Risk-free rate
call_price = 3.00
# Compute put price from put-call parity
put_price = call_price - S0 + K * math.exp(-r*T0)

# Portfolio: 1 call + 1 put + 1 share
initial_value = call_price + put_price + S0

sigma_implied = 0.33  
sigma_stock = 0.25

# Compute d1 and d2, Greeks
d1 = (math.log(S0/K) + (r + 0.5*sigma_implied**2)*T0) / (sigma_implied * math.sqrt(T0))
d2 = d1 - sigma_implied * math.sqrt(T0)
call_delta = norm.cdf(d1)
call_theta = -(S0 * norm.pdf(d1) * sigma_implied)/(2 * math.sqrt(T0)) - r * K * math.exp(-r*T0) * norm.cdf(d2)

# For the put: delta = call_delta - 1, theta from put-call parity
put_delta = call_delta - 1
put_theta = call_theta + r*(S0 - K*math.exp(-r*T0))

# Portfolio Greeks: add the share (delta = 1, theta = 0)
port_delta = call_delta + put_delta + 1
port_theta = call_theta + put_theta  
hold_days = 20
T_hold = hold_days / 255.0
theta_decay = port_theta * T_hold  

# Delta effect: stock's standard deviation over 20 days
stock_std = S0 * sigma_stock * math.sqrt(T_hold)
delta_std = port_delta * stock_std

# Delta-Normal approximation: 
mean_PL = theta_decay
std_PL = delta_std
z = norm.ppf(0.05)
PL_quantile = mean_PL + z * std_PL
VaR_delta = -PL_quantile
ES_delta = - (mean_PL - std_PL * norm.pdf(z) / 0.05)

print("    Delta-Normal Approximation   ")
print("Portfolio initial value: ${:.2f}".format(initial_value))
print("VaR (5%): ${:.2f}".format(VaR_delta))
print("ES (5%): ${:.2f}".format(ES_delta))

# Monte Carlo Simulation
num_sim = 100000
np.random.seed(0)
Z = np.random.randn(num_sim)
S_end = S0 * np.exp(-0.5 * sigma_stock**2 * T_hold + sigma_stock * math.sqrt(T_hold) * Z)
T_new = T0 - T_hold

# Reprice options at the end of 20 days
call_end = np.array([bs_call(s, K, T_new, r, sigma_implied) for s in S_end])
put_end = np.array([bs_put(s, K, T_new, r, sigma_implied) for s in S_end])
port_value_end = call_end + put_end + S_end
PL = port_value_end - initial_value

VaR_MC = -np.percentile(PL, 5)
ES_MC = -np.mean(PL[PL <= np.percentile(PL, 5)])

print("     Monte Carlo Simulation    ")
print("VaR (5%): ${:.2f}".format(VaR_MC))
print("ES (5%): ${:.2f}".format(ES_MC))

# Graph: Portfolio Value vs. Stock Price

def portfolio_value_at_S(S, T, r, sigma):
    """Compute the portfolio value (1 call + 1 put + 1 share) given stock price S."""
    return bs_call(S, K, T, r, sigma) + bs_put(S, K, T, r, sigma) + S

# Create a range of stock prices at the end of the holding period
S_range = np.linspace(20, 40, 200)
actual_values = np.array([portfolio_value_at_S(s, T_new, r, sigma_implied) for s in S_range])

# Delta-Normal (linear) approximation of portfolio value:
d1_hold = (math.log(S0/K) + (r + 0.5 * sigma_implied**2) * T_hold) / (sigma_implied * math.sqrt(T_hold))
call_delta_hold = norm.cdf(d1_hold)
put_delta_hold = call_delta_hold - 1
port_delta_hold = call_delta_hold + put_delta_hold + 1  # local portfolio delta over T_hold

initial_portfolio_at_S0 = portfolio_value_at_S(S0, T_new, r, sigma_implied)
linear_approx = initial_portfolio_at_S0 + port_delta_hold * (S_range - S0) + theta_decay

# Plot the actual (non-linear) portfolio value and its Delta-Normal linear approximation.
plt.figure(figsize=(8, 6))
plt.plot(S_range, actual_values, label='Actual Portfolio Value')
plt.plot(S_range, linear_approx, '--', label='Delta-Normal Approximation')
plt.xlabel('Stock Price at End of Holding Period')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value vs. Stock Price')
plt.legend()
plt.grid(True)
plt.show()
