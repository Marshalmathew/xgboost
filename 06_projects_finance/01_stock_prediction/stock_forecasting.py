# %% [markdown]
# # Project 1: Stock Prediction / Financial Forecasting
# 
# Using `yfinance`, we'll pull historical data for the S&P 500 (SPY) and try to predict if the next day's return will be positive or negative.
# 
# **Warning**: Stock market prediction is incredibly hard. This notebook focuses on the MLOps pipeline (time-series splitting) rather than a guaranteed winning trading strategy!

# %%
import yfinance as yf
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Fetching Data

# %%
ticker = "SPY"
data = yf.download(ticker, start="2010-01-01", end="2023-01-01")

# Create Features (Lagged returns, moving averages)
data['Returns'] = data['Close'].pct_change()
data['Lag_1'] = data['Returns'].shift(1)
data['Lag_2'] = data['Returns'].shift(2)
data['MA_10'] = data['Close'].rolling(window=10).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()

# Target: 1 if Returns > 0 else 0
data['Target'] = (data['Returns'] > 0).astype(int)

# Drop NaNs created by lagging/rolling
data = data.dropna()

features = ['Lag_1', 'Lag_2', 'MA_10', 'MA_50']
X = data[features]
y = data['Target']

# %% [markdown]
# ## 2. Time-Series Train/Test Split
# **CRITICAL**: Never use `train_test_split` with `shuffle=True` (or K-Fold CV) on time series. You will leak future data into the past. Always split chronologically.

# %%
split_idx = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# %% [markdown]
# ## 3. Training the Model

# %%
params = {
    'objective': 'binary:logistic',
    'max_depth': 4,
    'eta': 0.05,
    'subsample': 0.8, # Good for noisy financial data
    'colsample_bytree': 0.8
}

bst = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtrain, 'train'), (dtest, 'test')], verbose_eval=10)

# %% [markdown]
# ## 4. Evaluation

# %%
preds_prob = bst.predict(dtest)
preds_binary = (preds_prob > 0.5).astype(int)

print(f"Accuracy: {accuracy_score(y_test, preds_binary):.4f}")
print(classification_report(y_test, preds_binary))

# Feature Importance
xgb.plot_importance(bst, importance_type='gain')
plt.show()
