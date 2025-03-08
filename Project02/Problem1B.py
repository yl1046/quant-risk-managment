import pandas as pd
import numpy as np

# Load the dataset from the csv.
df = pd.read_csv("C:/Users/Eric Liu/OneDrive/Documents/Fintech545Project2/DailyPrices.csv", parse_dates=True, index_col=0)

# Selecting SPY, AAPL, EQIX out of the list
stocks = ['SPY', 'AAPL', 'EQIX']
df = df[stocks]

# Calculate log returns and remove the mean
log_returns = np.log(df / df.shift(1))
demeaned_log_returns = log_returns - log_returns.mean()

# Present the last 5 rows
last_five_rows = demeaned_log_returns.tail(5)

# Calculate the total standard deviation
total_std = demeaned_log_returns.stack().std()

# Display results.
print("Last 5 rows of de-meaned log returns:")
print(last_five_rows)
print("Total standard deviation (across all stocks and days):", total_std)
