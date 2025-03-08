import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

#read data
filepath = "C:/Users/Eric Liu/OneDrive/Documents/Fintech545Project1/problem1.csv"
data = pd.read_csv(filepath)

# Define Normal Distribution
mu, sigma = 0.049238, 0.010321
normal_dist = stats.norm(mu, sigma)

# PDF Example
x = data.iloc[:, 0]
df = pd.DataFrame({'x': x})
df['pdf'] = normal_dist.pdf(x)
print(df.head())

# Plot PDF
plt.figure()
plt.plot(df['x'], df['pdf'], label='PDF')
plt.legend()
plt.title('Probability Density Function')
plt.show()
def first_four_moments(sample):
    n = len(sample)
    mu_hat = np.mean(sample)
    sim_corrected = sample - mu_hat
    cm2 = np.mean(sim_corrected ** 2)
    sigma2_hat = np.var(sample, ddof=1)
    skew_hat = np.mean(sim_corrected ** 3) / (cm2 ** (3/2))
    kurt_hat = np.mean(sim_corrected ** 4) / (cm2 ** 2) - 3
    return mu_hat, sigma2_hat, skew_hat, kurt_hat

#Fit both normal and t-distribution
data_values = data.iloc[:, 0]
normal_stat = stats.norm.fit(data_values)
t_stat = stats.t.fit(data_values)

#Getting simulated data from normal distributiona and t-distribution
simulated_normal = stats.norm.rvs(*normal_stat, size=len(data_values))
simulated_t = stats.t.rvs(*t_stat, size=len(data_values))

#first_four_moments
original_moments = first_four_moments(data_values)
normal_moments = first_four_moments(simulated_normal)
t_moments = first_four_moments(simulated_t)

moment_results = pd.DataFrame({
    "Moment": ["Mean", "Variance", "Skewness", "Kurtosis"],
    "Original data": original_moments,
    "Normal fit": normal_moments,
    "T_distribution fit": t_moments
})

print(moment_results)
