# Problem 1 question A
import pandas as pd
filepath = "C:/Users/Eric Liu/OneDrive/Documents/Fintech545Project1/problem1.csv"
df = pd.read_csv(filepath)

# Calculate the mean, variance, skewness, and kurtosis
mean = df['X'].mean()
variance = df['X'].var()
skewness = df['X'].skew()
kurtosis = df['X'].kurtosis()

print(mean, variance, skewness, kurtosis)
