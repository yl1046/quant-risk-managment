import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt

#problem3A
# Read the data
filepath = "C:/Users/Eric Liu/OneDrive/Documents/Fintech545Project1/problem3.csv"
df = pd.read_csv(filepath)

#The mean and covariance matrix of the data
mean_para = df.mean().values
covMatrix = df.cov().values

# Fit the multivariate normal distribution
multivari_dist = multivariate_normal(mean=mean_para, cov=covMatrix)

print(mean_para, covMatrix)

#problem3C
# the mean and the standard deviation and known X1
mu, sigma = [0.04600157, 0.09991502], [[0.0101622, 0.00492354], 
[0.00492354, 0.02028441]]
X1 = 0.6  

# bivariate normal samples
X = np.random.multivariate_normal(mu, sigma, 10000)

# conditional distribution by the first method
mu_condi = mu[1] + sigma[0][1] / sigma[0][0] * (X1 - mu[0])
sigma_condi = np.sqrt(sigma[1][1] - (sigma[0][1] ** 2) / sigma[0][0])

# samples for X2 | X1 = 0.6
X2_given_X1 = np.random.normal(mu_condi, sigma_condi, 10000)

# Plot histogram with theoretical distribution
plt.hist(X2_given_X1, bins=50, density=True, alpha=0.6, color='b', label="Simulation X2 | X1=0.6")
x_vals = np.linspace(min(X2_given_X1), max(X2_given_X1), 100)
plt.plot(x_vals, norm.pdf(x_vals, mu_condi, sigma_condi), 'r-', label="PDF")
plt.legend()
plt.show()

