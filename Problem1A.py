import pandas as pd

# Load the dataset from the csv.
df = pd.read_csv('C:\Users\Eric Liu\OneDrive\桌面\FINTECH 545 Quant Risk\Fintech545Project2\DailyPrices.csv', parse_dates=True, index_col=0)

# Selecting SPY, AAPL, EQIX out of the list
stocks = ['SPY', 'AAPL', 'EQIX']
df = df[stocks]

# Calculate arithmetic returns and remove the mean
returns = df.pct_change()
demeaned_returns = returns - returns.mean()

# Present the last 5 rows
last_five_rows = demeaned_returns.tail(5)

# Calculate the total standard deviation
total_std_dev = demeaned_returns.stack().std()

# Display results:
print("Last 5 rows of de-meaned arithmetic returns:")
print(last_five_rows)
print("Total standard deviation:", total_std_dev)