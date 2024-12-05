import requests
import pandas as pd
from datetime import datetime

# coingecko fetching

def fetch_coingecko_data(coin_id, vs_currency, from_date, to_date):
    base_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    params = {
        'vs_currency': vs_currency,
        'from': int(datetime.strptime(from_date, '%Y-%m-%d').timestamp()),
        'to': int(datetime.strptime(to_date, '%Y-%m-%d').timestamp())
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print(f"Error: {response.json()}")
        return None

    data = response.json()
    prices = data['prices']
    ohlc = pd.DataFrame(prices, columns=['timestamp', 'price'])
    ohlc['timestamp'] = pd.to_datetime(ohlc['timestamp'], unit='ms')
    return ohlc

# fetching btc / eth data

btc_data = fetch_coingecko_data('bitcoin', 'usd', '2020-01-01', '2023-12-01')
eth_data = fetch_coingecko_data('ethereum', 'usd', '2020-01-01', '2023-12-01')

# saving 

if btc_data is not None:
    btc_data.to_csv('BTC_Coingecko.csv', index=False)
if eth_data is not None:
    eth_data.to_csv('ETH_Coingecko.csv', index=False)

print("Data for BTC and ETH has been saved using CoinGecko!")

