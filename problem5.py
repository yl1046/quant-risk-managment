import pandas as pd

filepath = "C:/Users/Eric Liu/OneDrive/Documents/Fintech545Project1/DailyReturn.csv"
df = pd.read_csv(filepath)

# set index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# exponentially weighted covariance matrix
span = 35
exwem_cov_matrix = df.ewm(span=span).cov(pairwise=True)

latest_cov_matrix = exwem_cov_matrix.loc[df.index.max()]
print(latest_cov_matrix)
