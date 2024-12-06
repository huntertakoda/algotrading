import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# loading

btc_data = pd.read_csv(r'BTC_Featured_NewTarget.csv')

# ftres and target definition
features = ['BTC_price', 'daily_return', 'MA7', 'MA30', 'volatility', 'RSI', 'BB_Upper', 'BB_Lower', 'lag_1', 'lag_2', 'lag_3']
X = btc_data[features]
y = btc_data['target']

# train / test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# xgboost model training

best_xgb_model = XGBClassifier(
    random_state=42,
    learning_rate=0.2,
    max_depth=10,
    min_child_weight=1,
    n_estimators=200,
    subsample=0.8
)
best_xgb_model.fit(X_train, y_train)

# predictions

y_pred = best_xgb_model.predict(X_test)

# evaluation

test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {test_accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# confusion matrix visualization

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.title('Confusion Matrix (Tuned XGBoost)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
