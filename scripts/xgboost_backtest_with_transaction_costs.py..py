import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# loading

btc_data = pd.read_csv(r'BTC_Featured_NewTarget.csv')

# ftres and target definition

features = ['BTC_price', 'daily_return', 'MA7', 'MA30', 'volatility', 'RSI', 'BB_Upper', 'BB_Lower', 'lag_1', 'lag_2', 'lag_3']
X = btc_data[features]
y = btc_data['target']

# train / test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# xgboost training

from xgboost import XGBClassifier
best_xgb_model = XGBClassifier(
    random_state=42,
    learning_rate=0.2,
    max_depth=10,
    min_child_weight=1,
    n_estimators=200,
    subsample=0.8
)
best_xgb_model.fit(X_train, y_train)

# prediction probabilities and optimal threshold

y_probs = best_xgb_model.predict_proba(X_test)[:, 1]
optimal_threshold = 0.65
y_pred = (y_probs >= optimal_threshold).astype(int)

# align data for backtesting

btc_data_test = btc_data.iloc[X_test.index].copy()
btc_data_test['prediction'] = y_pred
btc_data_test['actual_change'] = btc_data_test['BTC_price'].shift(-1) - btc_data_test['BTC_price']

# transaction cost definition
transaction_cost = 0.001

# simulate returns with transaction costs
btc_data_test['return'] = np.where(
    btc_data_test['prediction'] == 1,
    btc_data_test['actual_change'] - (btc_data_test['BTC_price'] * transaction_cost),
    -btc_data_test['actual_change'] - (btc_data_test['BTC_price'] * transaction_cost)
)
btc_data_test.dropna(subset=['return'], inplace=True)
btc_data_test['cumulative_return'] = btc_data_test['return'].cumsum()

# plot cumulative returns

plt.figure(figsize=(10, 6))
plt.plot(btc_data_test['cumulative_return'], label='Cumulative Return with Transaction Costs')
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title('Backtesting: Cumulative Returns with Transaction Costs (Threshold 0.65)')
plt.xlabel('Trades')
plt.ylabel('Cumulative Return (USD)')
plt.legend()
plt.show()

# summary statistics

total_return = btc_data_test['cumulative_return'].iloc[-1]
win_rate = (btc_data_test['return'] > 0).mean() * 100
max_drawdown = (btc_data_test['cumulative_return'].cummax() - btc_data_test['cumulative_return']).max()

print(f"Optimal Threshold: {optimal_threshold}")
print(f"Total Return (with Transaction Costs): ${total_return:.2f}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Maximum Drawdown: ${max_drawdown:.2f}")
