# 07 - Distributed XGBoost

XGBoost on a single machine is incredibly fast, especially with GPUs. However, when your dataset exceeds RAM/GPU memory, or when you are operating within a massive ETL pipeline, you need distributed computing.

XGBoost provides native integration with three major distributed frameworks:

## 1. Dask
- **Best For**: Python-native data scientists migrating from Pandas/Scikit-Learn.
- **Why**: Dask DataFrames parallelize Pandas. Dask-XGBoost allows you to train an XGBoost model across a Dask cluster seamlessly. It's often the easiest entry point for local cluster scaling.

## 2. PySpark
- **Best For**: Enterprise environments with existing massive Hadoop/Spark infrastructure.
- **Why**: PySpark is the industry standard for big data ETL. The `xgboost.spark` module allows you to integrate XGBoost directly into your Spark MLlib pipelines, avoiding the need to move terabytes of data between a data lake and a separate compute node.

## 3. Ray
- **Best For**: Modern MLOps, deep learning ecosystems, and heterogeneous clusters.
- **Why**: Ray is designed for highly scalable distributed execution (used heavily at OpenAI). `xgboost_ray` handles fault tolerance, elastic training (adding/removing nodes mid-training), and multi-node multi-GPU setups beautifully.
