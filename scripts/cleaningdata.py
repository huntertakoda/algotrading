import pandas as pd

# loading data
btc_data = pd.read_csv('BTC_Coingecko.csv')
eth_data = pd.read_csv('ETH_Coingecko.csv')

# data structure

print("BTC Data Info:")
print(btc_data.info())
print("\nETH Data Info:")
print(eth_data.info())

# transformation2datetime

btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'])
eth_data['timestamp'] = pd.to_datetime(eth_data['timestamp'])

btc_data.rename(columns={'price': 'BTC_price'}, inplace=True)
eth_data.rename(columns={'price': 'ETH_price'}, inplace=True)

# check for missing values

print("Missing Values in BTC Data:\n", btc_data.isnull().sum())
print("Missing Values in ETH Data:\n", eth_data.isnull().sum())

# handling / dropping missing data

btc_data = btc_data.dropna()
eth_data = eth_data.dropna()

# saving

btc_data.to_csv('BTC_Cleaned.csv', index=False)
eth_data.to_csv('ETH_Cleaned.csv', index=False)

print("Data preprocessing completed. Cleaned files saved!")
