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

---

## 4. Production Realities & Distributed Failure Modes

While local simulations (`LocalCluster`, `local[*]`, `num_actors=1`) demonstrate API usage, production distributed clusters face distinct operational challenges:

### A. AllReduce vs. Centralized Aggregation (Rabit Architecture)
XGBoost coordinates distributed workers using **Rabit** (Reliable Adaptive Bias/Variance Training). Unlike architectures with a bottlenecked parameter server, Rabit implements an AllReduce ring/tree topology:
- Workers build local histograms of gradients ($g$) and hessians ($h$) on their partitions.
- AllReduce synchronizes histogram bins across all workers with optimal $\mathcal{O}(\log P)$ communication overhead.

### B. Network & Bandwidth Saturation
- In distributed tree training, network bandwidth—not CPU compute—is usually the primary bottleneck.
- **Mitigation**: Use `tree_method='hist'`, reduce `max_bin` (e.g., 256), and ensure high-throughput inter-node interconnects (10GbE / InfiniBand).

### C. Partition Skew & Straggler Problem
- An AllReduce iteration is only as fast as its slowest worker. If partition sizes or memory usage vary significantly across executors, fast workers idle waiting at synchronization barriers.
- **Mitigation**: Repartition datasets evenly before feeding into `DMatrix` or Spark/Dask/Ray estimators.

### D. Memory Headroom & OOM Prevention
- Converting raw data partitions (e.g. Spark DataFrames or Arrow tables) into native DMatrix structures creates temporary memory spikes of $2\times$ to $3\times$.
- **Mitigation**: Size executor RAM with sufficient headroom, stream partitions via iterator interfaces, or leverage memory-mapped external memory caching.

