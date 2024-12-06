import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# loading

btc_data = pd.read_csv(r'C:/Users/yanfr/algorithmictrading/BTC_Featured_NewTarget.csv')

# ftres and target definition

features = ['BTC_price', 'daily_return', 'MA7', 'MA30', 'volatility', 'RSI', 'BB_Upper', 'BB_Lower', 'lag_1', 'lag_2', 'lag_3']
X = btc_data[features]
y = btc_data['target']

# train / test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# predictions

best_xgb_model = XGBClassifier(
    random_state=42,
    learning_rate=0.2,
    max_depth=10,
    min_child_weight=1,
    n_estimators=200,
    subsample=0.8
)
best_xgb_model.fit(X_train, y_train)
y_pred = best_xgb_model.predict(X_test)

btc_data_test = btc_data.iloc[X_test.index].copy()
btc_data_test['prediction'] = y_pred

# actual price changes calculation

btc_data_test['actual_change'] = btc_data_test['BTC_price'].shift(-1) - btc_data_test['BTC_price']
btc_data_test['signal'] = np.where(btc_data_test['prediction'] == 1, 'Buy', 'Sell')

# returns simulation

btc_data_test['return'] = np.where(
    btc_data_test['prediction'] == 1,  
    btc_data_test['actual_change'],    
    -btc_data_test['actual_change']    
)

print(btc_data_test[['BTC_price', 'actual_change', 'return']].isna().sum())
btc_data_test.dropna(subset=['return'], inplace=True)

# cumulative returns calculation

btc_data_test['cumulative_return'] = btc_data_test['return'].cumsum()

# cumulative returns visualization

plt.figure(figsize=(10, 6))
plt.plot(btc_data_test['cumulative_return'], label='Cumulative Return')
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title('Backtesting: Cumulative Returns')
plt.xlabel('Trades')
plt.ylabel('Cumulative Return (USD)')
plt.legend()
plt.show()

# summary stats

total_return = btc_data_test['cumulative_return'].iloc[-1]
win_rate = (btc_data_test['return'] > 0).mean() * 100
max_drawdown = (btc_data_test['cumulative_return'].cummax() - btc_data_test['cumulative_return']).max()

print(f"Total Return: ${total_return:.2f}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Maximum Drawdown: ${max_drawdown:.2f}")
