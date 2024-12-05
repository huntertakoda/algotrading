import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# loading

btc_data = pd.read_csv('BTC_Featured.csv')

# feature/target defining

features = ['BTC_price', 'daily_return', 'MA7', 'MA30', 'volatility']
X = btc_data[features]
y = btc_data['target']

# intialize+train

model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X, y)

# extracting ftre importance
importance = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# ftre importance

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', hue=None, legend=False)
plt.title('Feature Importance', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.show()

