import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

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

# prediction probabilities

y_probs = best_xgb_model.predict_proba(X_test)[:, 1]

# evaluating thresholds

thresholds = np.arange(0.3, 0.71, 0.05)
results = []

for threshold in thresholds:

    y_pred = (y_probs >= threshold).astype(int)
    
    btc_data_test = btc_data.iloc[X_test.index].copy()
    btc_data_test['prediction'] = y_pred
    btc_data_test['actual_change'] = btc_data_test['BTC_price'].shift(-1) - btc_data_test['BTC_price']
    btc_data_test['return'] = np.where(
        btc_data_test['prediction'] == 1,
        btc_data_test['actual_change'],
        -btc_data_test['actual_change']
    )
    btc_data_test.dropna(subset=['return'], inplace=True)
    btc_data_test['cumulative_return'] = btc_data_test['return'].cumsum()
    
    total_return = btc_data_test['cumulative_return'].iloc[-1]
    win_rate = (btc_data_test['return'] > 0).mean() * 100
    max_drawdown = (btc_data_test['cumulative_return'].cummax() - btc_data_test['cumulative_return']).max()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    results.append({
        'Threshold': threshold,
        'Total Return': total_return,
        'Win Rate': win_rate,
        'Max Drawdown': max_drawdown,
        'Precision': precision,
        'Recall': recall
    })

# results conversion to dataframe

results_df = pd.DataFrame(results)

# calculate risk-adjusted score

results_df['Risk-Adjusted Score'] = results_df['Total Return'] / results_df['Max Drawdown']

# find optimal threshold

optimal_threshold = results_df.sort_values(by='Risk-Adjusted Score', ascending=False).iloc[0]

# display optimal threshold

print("Optimal Threshold for Balanced Returns and Risk:")
print(optimal_threshold)

# save updated results

results_df.to_csv(r'C:/Users/yanfr/algorithmictrading/threshold_optimization_with_risk.csv', index=False)
print("Updated threshold optimization results saved to 'threshold_optimization_with_risk.csv'.")
