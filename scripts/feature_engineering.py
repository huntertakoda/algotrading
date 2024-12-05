import pandas as pd

# loading

btc_data = pd.read_csv('BTC_Cleaned.csv')
eth_data = pd.read_csv('ETH_Cleaned.csv')

btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'])
eth_data['timestamp'] = pd.to_datetime(eth_data['timestamp'])

btc_data.sort_values('timestamp', inplace=True)
eth_data.sort_values('timestamp', inplace=True)

# daily returns calculation

btc_data['daily_return'] = btc_data['BTC_price'].pct_change()
eth_data['daily_return'] = eth_data['ETH_price'].pct_change()

# moving averages

btc_data['MA7'] = btc_data['BTC_price'].rolling(window=7).mean()
btc_data['MA30'] = btc_data['BTC_price'].rolling(window=30).mean()
eth_data['MA7'] = eth_data['ETH_price'].rolling(window=7).mean()
eth_data['MA30'] = eth_data['ETH_price'].rolling(window=30).mean()

# rolling volatility viz standard deviation of returns

btc_data['volatility'] = btc_data['daily_return'].rolling(window=7).std()
eth_data['volatility'] = eth_data['daily_return'].rolling(window=7).std()

btc_data['day_of_week'] = btc_data['timestamp'].dt.day_name()
eth_data['day_of_week'] = eth_data['timestamp'].dt.day_name()

# saving
btc_data.to_csv('BTC_Featured.csv', index=False)
eth_data.to_csv('ETH_Featured.csv', index=False)

print("feature engineering completed")
