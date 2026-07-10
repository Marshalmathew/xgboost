# 03 - Basic Usage and Hyperparameter Tuning

Now that we understand the math, it's time to map it to the API and practical usage.

## 1. DMatrix: The Secret Sauce
XGBoost uses an internal data structure called `DMatrix` that optimizes both memory consumption and training speed.
- It is highly recommended to explicitly convert your Pandas DataFrames or Numpy arrays into `xgb.DMatrix` before training.
- Missing values can be specified during conversion (e.g., `missing=np.nan`), and XGBoost will handle them optimally via the Sparsity-Aware algorithm discussed in Module 2.

## 2. The Core Hyperparameters

XGBoost is powerful but prone to overfitting if not tuned properly.

### Controlling Tree Complexity (The $\Omega$ in our Objective Function)
- `max_depth` (default=6): Maximum depth of a tree. Higher = more complex.
- `min_child_weight` (default=1): Minimum sum of Instance Weight (Hessian) needed in a child. In regression, this is just the minimum number of instances. In classification, it's the sum of $p(1-p)$. Higher = more conservative.
- `gamma` / `min_split_loss` (default=0): Minimum loss reduction (Gain) required to make a further partition.

### Robustness to Noise (Stochastic Gradient Boosting)
- `subsample` (default=1): Subsample ratio of the training instances. Setting it to 0.5 means XGBoost will randomly sample half of the training data prior to growing trees.
- `colsample_bytree` (default=1): Subsample ratio of columns (features) when constructing each tree. Similar to Random Forests.

### Step Size Shrinkage
- `eta` / `learning_rate` (default=0.3): Step size shrinkage used in update to prevents overfitting. After each boosting step, we multiply the new tree's weights by `eta`.

## 3. Tuning Strategies

Never tune everything at once. A good sequence:
1. Fix `eta` to a relatively high value (e.g., 0.1) and find the optimal number of boosting rounds (`num_boost_round`) using Early Stopping.
2. Tune tree-specific parameters: `max_depth` and `min_child_weight`.
3. Tune stochastic parameters: `subsample` and `colsample_bytree`.
4. Tune regularization parameters: `gamma`, `alpha` (L1), `lambda` (L2).
5. Lower `eta` (e.g., 0.01) and proportionally increase `num_boost_round` for the final model.

We will use **Optuna** for Bayesian optimization in our practical notebook.
