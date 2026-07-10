# Distributed XGBoost using Dask
# Dask is the most Pythonic way to scale XGBoost across a cluster.
# This script demonstrates how to set up a LocalCluster and train an XGBoost model.

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import xgboost as xgb
from sklearn.datasets import make_classification
import pandas as pd

def main():
    print("Setting up Dask Cluster...")
    # In a real environment, you might connect to a remote cluster here.
    # We use a LocalCluster for demonstration.
    cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    client = Client(cluster)
    print(f"Dashboard available at: {client.dashboard_link}")

    print("Generating synthetic data...")
    # Generate data locally, then distribute it (for demo purposes)
    X, y = make_classification(n_samples=100000, n_features=20, random_state=42)
    pdf = pd.DataFrame(X)
    pdf['target'] = y
    
    # Convert to Dask DataFrame (partitioned across workers)
    ddf = dd.from_pandas(pdf, npartitions=8)
    
    X_ddf = ddf.drop('target', axis=1)
    y_ddf = ddf['target']

    # Convert to DaskDMatrix
    print("Creating DaskDMatrix...")
    dtrain = xgb.dask.DaskDMatrix(client, X_ddf, y_ddf)

    # Train
    print("Training XGBoost on Dask Cluster...")
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist', # Use 'hist' for distributed CPU training
        'max_depth': 4
    }
    
    output = xgb.dask.train(
        client,
        params,
        dtrain,
        num_boost_round=100
    )

    bst = output['booster']
    print("Training complete! Model type:", type(bst))
    
    client.close()
    cluster.close()

if __name__ == "__main__":
    main()
