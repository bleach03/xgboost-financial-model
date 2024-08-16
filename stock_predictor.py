import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from finta import TA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# Download data for AAPL
df = yf.download('AAPL', start='2010-01-01', end='2024-08-11')

# Technical indicators
df['SMA200'] = TA.SMA(df, 200)
df['RSI'] = TA.RSI(df)
df['ATR'] = TA.ATR(df)
df['BBWidth'] = TA.BBWIDTH(df)
df['Williams'] = TA.WILLIAMS(df)

# Add 'target' to dataframe
df['target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# Define features and target
features = ['SMA200', 'RSI', 'ATR', 'BBWidth', 'Williams']
X = df[features]
y = df['target']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid Search for hyperparameters
params = {
    'max_depth': [3, 6],
    'learning_rate': [0.05],
    'n_estimators': [700, 1000],
    'colsample_bytree': [0.3, 0.7]
}
xgbr = XGBRegressor(seed=20)
model = GridSearchCV(estimator=xgbr, param_grid=params, scoring='neg_mean_squared_error', verbose=1)
model.fit(X_train, y_train)

print("Best Parameters: ", model.best_params_)
print("Lowest RMSE: ", (-model.best_score_) ** (1/2.0))

# Predict on the entire dataset
y_pred_all = model.predict(X)
df['Predicted'] = y_pred_all

# Calculate RMSE for the test set (optional, for evaluation purposes)
y_pred_test = model.predict(X_test)
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
print("Test RMSE: ", test_rmse)

print(len(y_test))
print(len(y_pred_test))
print(len(df))

# Plot actual values and predicted
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['Close'], label='Actual Close Price', color='blue', alpha=0.6)
plt.plot(df.index, df['Predicted'], label='Predicted Close Price', color='red', alpha=0.6)
plt.xlabel('Date')
plt.ylabel('Close Price $')
plt.legend()
plt.title('AAPL Actual vs Predicted Close Prices')
plt.show()