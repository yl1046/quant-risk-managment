import pandas as pd

# Load the dataset from the csv.
df = pd.read_csv("C:/Users/Eric Liu/OneDrive/Documents/Fintech545Project2/DailyPrices.csv", parse_dates=True, index_col=0)
df.index = pd.to_datetime(df.index)

# Clarify the portfolio shares
shares = {'SPY': 100, 'AAPL': 200, 'EQIX': 150}

# Define the date: January 3, 2025
date_of_20250103 = pd.Timestamp('2025-01-03')

# Pick out the prices for SPY, AAPL, EQIX on 2025-01-03
prices_on_date = df.loc[date_of_20250103, ['SPY', 'AAPL', 'EQIX']]
prices_on_date = prices_on_date.round(2)

# Calculate the current value of the portfolio:
portfolio_value = (prices_on_date * pd.Series(shares)).sum()
portfolio_value = round(portfolio_value, 2)

print("Prices on", date_of_20250103.date(), "for SPY, AAPL, EQIX:")
print(prices_on_date)
print("Portfolio value on", date_of_20250103.date(), ":", portfolio_value)
