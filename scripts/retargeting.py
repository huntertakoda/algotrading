import pandas as pd

# loading

btc_data = pd.read_csv('BTC_Featured_Enhanced.csv')

prediction_window = 3  
btc_data['future_price'] = btc_data['BTC_price'].shift(-prediction_window)
btc_data['target'] = (btc_data['future_price'] > btc_data['BTC_price']).astype(int)

btc_data.dropna(inplace=True)

# saving

btc_data.to_csv('BTC_Featured_NewTarget.csv', index=False)

print("New target defined and dataset updated!")
