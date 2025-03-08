import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the data
file_path = "C:/Users/Eric Liu/OneDrive/Documents/Fintech545Project1/DailyReturn.csv"
df = pd.read_csv(file_path)

# The index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

# Define λ values for the exponential weighting
lambda_values = np.linspace(0.12, 0.95, 5)  

plt.figure(figsize=(9, 6))

for lambda_ in lambda_values:
    alpha = 1 - lambda_  
    ewm_cov_matrix = df.ewm(alpha=alpha).cov(pairwise=True)
    latest_cov_matrix = ewm_cov_matrix.loc[df.index.max()]

    # Perform PCA
    pca = PCA()
    pca.fit(latest_cov_matrix)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, label=f'λ={lambda_:.1f}')

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Variance Explained")
plt.title("Cumulative Variance Explained by PCA for Different λ Values")
plt.legend()
plt.grid()
plt.show()
