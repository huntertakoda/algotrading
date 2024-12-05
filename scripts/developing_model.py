import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# loading

btc_data = pd.read_csv('BTC_Featured.csv')

btc_data['target'] = (btc_data['BTC_price'].shift(-1) > btc_data['BTC_price']).astype(int)

btc_data.dropna(inplace=True)

features = ['BTC_price', 'daily_return', 'MA7', 'MA30', 'volatility']
X = btc_data[features]
y = btc_data['target']

# training / testing set split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# random forest

model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# predicting

y_pred = model.predict(X_test)

# evaluation

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# confusion matrix visualization

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

