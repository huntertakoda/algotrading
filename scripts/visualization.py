import pandas as pd
import matplotlib.pyplot as plt

# loading

btc_data = pd.read_csv('BTC_Cleaned.csv')
eth_data = pd.read_csv('ETH_Cleaned.csv')

btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'])
eth_data['timestamp'] = pd.to_datetime(eth_data['timestamp'])

# btc price trend visualization

plt.figure(figsize=(12, 6))
plt.plot(btc_data['timestamp'], btc_data['BTC_price'], label='BTC Price', linewidth=2)
plt.title('BTC Price Trend', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.legend()
plt.grid()
plt.show()

# eth price trend visualization

plt.figure(figsize=(12, 6))
plt.plot(eth_data['timestamp'], eth_data['ETH_price'], label='ETH Price', linewidth=2, color='orange')
plt.title('ETH Price Trend', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.legend()
plt.grid()
plt.show()

# btc / eth comparison visualization

plt.figure(figsize=(12, 6))
plt.plot(btc_data['timestamp'], btc_data['BTC_price'], label='BTC Price', linewidth=2)
plt.plot(eth_data['timestamp'], eth_data['ETH_price'], label='ETH Price', linewidth=2, color='orange')
plt.title('BTC vs ETH Price Trends', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.legend()
plt.grid()
plt.show()
