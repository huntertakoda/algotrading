import pandas as pd

# loading

btc_data = pd.read_csv('BTC_Featured.csv')

# relative strength index (rsi)

def calculate_rsi(data, window=14):
    delta = data['BTC_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

btc_data['RSI'] = calculate_rsi(btc_data)

# bollinger bands

btc_data['BB_Upper'] = btc_data['MA30'] + 2 * btc_data['BTC_price'].rolling(window=30).std()
btc_data['BB_Lower'] = btc_data['MA30'] - 2 * btc_data['BTC_price'].rolling(window=30).std()

# lagged ftres

btc_data['lag_1'] = btc_data['BTC_price'].shift(1)
btc_data['lag_2'] = btc_data['BTC_price'].shift(2)
btc_data['lag_3'] = btc_data['BTC_price'].shift(3)

# drop nan value rows

btc_data.dropna(inplace=True)

# saving

btc_data.to_csv('BTC_Featured_Enhanced.csv', index=False)

print("New features added and dataset saved!")
