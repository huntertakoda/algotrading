# algotrading

## Algorithmic Trading with Machine Learning

### Introduction
This project applies machine learning (ML) techniques to develop an algorithmic trading strategy for cryptocurrency markets. The pipeline involves data preprocessing, feature engineering, model training, threshold optimization, and backtesting.

### Key Features
1. **Data Collection**: Pulling historical price data using the Coingecko API.
2. **Feature Engineering**: Adding technical indicators such as RSI, Moving Averages, Bollinger Bands, and lagged returns.
3. **Model Training**: Using XGBoost to predict price movements.
4. **Threshold Optimization**: Tuning the prediction threshold for better returns.
5. **Backtesting**: Evaluating strategy performance, including transaction costs.

### Folder Structure
- **`scripts/`**:
  - All Python scripts for each phase of the project (data processing, modeling, backtesting).
- **`visualizations/`**:
  - Visualizations generated during the project, including feature importance, confusion matrices, and cumulative returns.

### Key Results
- **Best Threshold**: `0.65`
  - Achieved the best balance between risk-adjusted returns and drawdowns.
- **Cumulative Returns**: Highlighted the importance of threshold tuning for profitability.

---

## Analysis & Insights

1. **Profitability**:
   - The best-performing threshold (`0.65`) yielded a **cumulative return of $430,636** in simulated trading over the selected time period.
   - Accounting for transaction costs, the return slightly decreased to **$423,007**.
   - **Win Rate**: 51.6% of trades were profitable.

2. **Feature Importance**:
   - The model identified key features driving its predictions:
     - **MA30 (30-day Moving Average)**: Captures medium-term price trends.
     - **Bollinger Bands**: Identify overbought/oversold conditions.
     - **RSI (Relative Strength Index)**: Measures price momentum.
   - These features align with traditional technical analysis, demonstrating that the model’s predictions have practical relevance.

3. **Threshold Optimization**:
   - Lower thresholds (e.g., `0.3`) resulted in frequent trades with higher transaction costs, reducing profitability.
   - A higher threshold (e.g., `0.65`) balanced precision and trade frequency, improving profitability and reducing drawdowns.

4. **Risk Management**:
   - Incorporating **stop-loss (-2%) and take-profit (+3%)** rules helped manage risk during volatile markets.
   - While these measures reduced extreme losses, they also highlighted the need for proper calibration to avoid limiting overall returns.

5. **Market Sensitivity**:
   - The strategy performed well in **bull markets**, capitalizing on trends, but struggled during **sideways/volatile markets** due to unpredictability.

6. **Real-World Considerations**:
   - The strategy assumes no latency in data or trade execution, which may differ in live trading environments.
   - Slippage and low liquidity in certain markets could impact the real-world applicability of these results.

---

### What's Next
This project demonstrates the potential of machine learning in algorithmic trading, but there are opportunities for further refinement:
- **Expand Asset Coverage**: Test the strategy on additional cryptocurrencies or asset classes like stocks or forex.
