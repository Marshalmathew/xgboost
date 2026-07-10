# %% [markdown]
# # Project 4: Marketing Propensity
# 
# **Dataset**: [Banking Dataset - Marketing Targets](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets)
# 
# In this project, we want to predict whether a customer will subscribe to a term deposit based on their demographic and previous marketing campaign interaction data. 
# 
# ## Goals
# - Handle categorical features natively.
# - Optimize for the right business metric (e.g., F1-Score or custom cost function balancing marketing spend vs conversion value).
# - Use SHAP to identify which customer segments are most likely to convert.

# %%
import xgboost as xgb
import pandas as pd
import numpy as np

# %% [markdown]
# ## 1. Load Data
# *Note: Download `train.csv` and `test.csv` from Kaggle and place them in this folder.*

# %%
try:
    df_train = pd.read_csv("train.csv", sep=";") # Adjust separator if needed based on CSV format
    print("Data loaded successfully!")
    
    # Preprocessing hints:
    # 1. Convert categorical columns to `category` dtype for XGBoost.
    # 2. Map target 'y' (yes/no) to 1/0.
    
except FileNotFoundError:
    print("Please download the dataset from Kaggle and place it in this directory.")

# %% [markdown]
# ## 2. Next Steps
# 1. Exploratory Data Analysis (EDA) on age, job, and marital status.
# 2. Train an XGBoost model `xgb.train()`.
# 3. Analyze SHAP values to present marketing insights to the business team!
