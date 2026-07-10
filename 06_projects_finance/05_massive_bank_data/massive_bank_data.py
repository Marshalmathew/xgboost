# %% [markdown]
# # Project 5: Massive Data Scaling
# 
# **Dataset**: [Massive Bank Dataset - 1 Million Rows](https://www.kaggle.com/datasets/ksabishek/massive-bank-dataset-1-million-rows)
# 
# This dataset is large enough that inefficient code might crash your memory, but small enough that XGBoost can handle it beautifully on a single machine if configured correctly.
# 
# ## Goals
# - Use efficient data loading techniques.
# - Leverage `tree_method='hist'` for blazing-fast CPU performance.
# - (Optional) If you have a GPU, test `tree_method='gpu_hist'` or `device='cuda'` and benchmark the training time.

# %%
import xgboost as xgb
import pandas as pd
import time

# %% [markdown]
# ## 1. Memory-Efficient Loading
# *Note: Download the dataset from Kaggle and place it in this folder.*

# %%
try:
    start_time = time.time()
    
    # Tip: Define dtypes beforehand to save memory during pandas read_csv
    df = pd.read_csv("bank_dataset.csv") 
    
    print(f"Loaded {len(df)} rows in {time.time() - start_time:.2f} seconds.")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    
except FileNotFoundError:
    print("Please download the massive dataset from Kaggle and place it in this directory.")

# %% [markdown]
# ## 2. Performance Tips
# 
# 1. Create your `DMatrix` and then `del df` to free up RAM.
# 2. Train the model using the histogram method.
# 
# ```python
# params = {
#     'tree_method': 'hist',
#     'max_bin': 256, # Default is 256, controls memory footprint of histograms
#     'objective': 'binary:logistic'
# }
# ```
