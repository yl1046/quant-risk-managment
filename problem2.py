#problem2A
import pandas as pd
import numpy as np

filepath = "C:/Users/Eric Liu/OneDrive/Documents/Fintech545Project1/problem2.csv"
df = pd.read_csv(filepath)

# Compute the pairwise covariance matrix
cov_matrix = df.cov()
print(cov_matrix)

#problem2B
# Compute the eigenvalues of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
print(eigenvalues)

#probelm2C
from scipy.linalg import sqrtm

def HighamNearestPsd(A, tol=1e-10):
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = V.T @ np.diag(np.maximum(s, tol)) @ V
    A_nearest = (B + H) / 2
    return (A_nearest + A_nearest.T) / 2

def NearestPsdRebonatoJackel(A):
    B = (A + A.T) / 2  
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals[eigvals < 0] = 0 
    A_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return A_psd

# Compute nearest PSD matrices
higham_psd_matrix = HighamNearestPsd(cov_matrix)
rebonatoANDjackel_psd_matrix = NearestPsdRebonatoJackel(cov_matrix)

# Display the results
print(higham_psd_matrix)
print(rebonatoANDjackel_psd_matrix)

#problem2D
# Exclude rows with any missing values
df_overlap = df.dropna()

# Compute the pairwise covariance matrix
cov_matrix_overlap = df_overlap.cov()
print(cov_matrix_overlap)
