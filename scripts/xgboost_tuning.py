import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# loading

btc_data = pd.read_csv(r'BTC_Featured_NewTarget.csv')

# ftres and target definition

features = ['BTC_price', 'daily_return', 'MA7', 'MA30', 'volatility', 'RSI', 'BB_Upper', 'BB_Lower', 'lag_1', 'lag_2', 'lag_3']
X = btc_data[features]
y = btc_data['target']

# train / test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# xgboost model definition

xgb_model = XGBClassifier(random_state=42)

# hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.8, 1.0]
}

# grid search

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# best parameters and cross-validation accuracy

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)
