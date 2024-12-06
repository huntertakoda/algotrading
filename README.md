# "algotrading"

## Algorithmic Trading with Machine Learning

## Introduction
This project applies machine learning (ML) techniques to develop an algorithmic trading strategy for cryptocurrency markets. The pipeline involves data preprocessing, feature engineering, model training, threshold optimization, and backtesting.

## Key Features
1. **Data Collection**: Pulling historical price data using the Coingecko API.
2. **Feature Engineering**: Adding technical indicators such as RSI, Moving Averages, Bollinger Bands, and lagged returns.
3. **Model Training**: Using XGBoost to predict price movements.
4. **Threshold Optimization**: Tuning the prediction threshold for better returns.
5. **Backtesting**: Evaluating strategy performance, including transaction costs.

## Folder Structure
- **`scripts/`**:
  - All Python scripts for each phase of the project (data processing, modeling, backtesting).
- **`visualizations/`**:
  - Visualizations generated during the project, including feature importance, confusion matrices, and cumulative returns.

## Key Results
- **Best Threshold**: 0.65
  - Showed the best balance between risk-adjusted returns and drawdowns.
- **Cumulative Returns**: Highlighted the importance of threshold tuning for profitability.
