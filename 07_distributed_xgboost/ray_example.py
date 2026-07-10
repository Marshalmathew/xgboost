# Distributed XGBoost using Ray
# Ray is ideal for heterogeneous clusters and deep learning integration.
# This script demonstrates xgboost_ray usage.

import ray
from xgboost_ray import RayDMatrix, RayParams, train
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def main():
    print("Initializing Ray...")
    # Initialize a local Ray cluster
    ray.init(ignore_reinit_error=True)

    print("Generating dummy data...")
    X, y = make_classification(n_samples=50000, n_features=10, random_state=42)
    pdf = pd.DataFrame(X)
    pdf['label'] = y

    print("Creating RayDMatrix...")
    # RayDMatrix is the distributed equivalent of xgb.DMatrix
    dtrain = RayDMatrix(pdf, label="label", num_actors=2) # Shard data across 2 actors

    print("Training XGBoost on Ray...")
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 4,
        'eta': 0.1
    }

    # RayParams defines cluster-level settings (num workers, CPUs/GPUs per worker)
    ray_params = RayParams(
        num_actors=2,
        cpus_per_actor=1,
        gpus_per_actor=0 # Set to >0 if you have GPUs!
    )

    evals_result = {}
    
    # Use xgboost_ray.train instead of xgb.train
    bst = train(
        params,
        dtrain,
        num_boost_round=50,
        evals=[(dtrain, 'train')],
        evals_result=evals_result,
        ray_params=ray_params,
        verbose_eval=10
    )

    print("Training complete! Final logloss:", evals_result['train']['logloss'][-1])
    
    # Save model locally
    bst.save_model("ray_xgboost.json")
    ray.shutdown()

if __name__ == "__main__":
    main()
