# 02 - XGBoost Core Mechanics: The Mathematics of Authority

To truly master XGBoost, you must understand the mathematical formulations that make it distinct from traditional Gradient Boosting. This module covers the math under the hood.

## 1. The Objective Function

In standard machine learning, we optimize a loss function $L(\theta)$. XGBoost optimizes a regularized objective function:

\[ Obj(\theta) = L(\theta) + \Omega(\theta) \]

- **Training Loss $L(\theta)$**: Measures how well the model fits the training data (e.g., Mean Squared Error, Log Loss).
- **Regularization $\Omega(\theta)$**: Penalizes the complexity of the model to prevent overfitting.
  - $\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$
  - Where $T$ is the number of leaves in the tree, $w_j$ are the leaf weights (scores), $\gamma$ controls the penalty for adding a new leaf (pruning), and $\lambda$ is L2 regularization on leaf weights.

## 2. Taylor Expansion (Second-Order Gradients)

Standard GBM uses the first-order gradient of the loss function (the residuals). XGBoost uses a second-order Taylor expansion to approximate the loss function, which converges much faster and allows for custom loss functions natively.

For a given prediction at step $t$, $\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)$, the objective becomes:

\[ Obj^{(t)} \approx \sum_{i=1}^n \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t) \]

Where:
- $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ (First-order derivative, **Gradient**)
- $h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ (Second-order derivative, **Hessian**)

Because $l(y_i, \hat{y}_i^{(t-1)})$ is a constant at step $t$, we can remove it. The final simplified objective for a tree is:

\[ Obj^{(t)} = \sum_{j=1}^T \left[ (\sum_{i \in I_j} g_i) w_j + \frac{1}{2} (\sum_{i \in I_j} h_i + \lambda) w_j^2 \right] + \gamma T \]

From this, the optimal weight $w_j^*$ for leaf $j$ is analytically solved as:

\[ w_j^* = - \frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda} \]

## 3. The Gain (Tree Splitting Criterion)

When building a tree, XGBoost evaluates a potential split by calculating the reduction in the objective function. This is called the **Gain**:

\[ Gain = \frac{1}{2} \left[ \frac{(\sum_{L} g_i)^2}{\sum_{L} h_i + \lambda} + \frac{(\sum_{R} g_i)^2}{\sum_{R} h_i + \lambda} - \frac{(\sum_{I} g_i)^2}{\sum_{I} h_i + \lambda} \right] - \gamma \]

- The first term represents the score of the Left child.
- The second term is the Right child.
- The third term is the original parent node.
- $\gamma$ is the penalty for adding an additional split.
If $Gain > 0$ (or technically, if it improves the objective), the split is kept. Otherwise, it is pruned.

## 4. Engineering Marvels

XGBoost isn't just math; it's systems engineering.
1. **Approximate Algorithm (Histograms)**: Instead of sorting all data to find exact splits, it groups features into discrete bins (histograms). This takes $O(bins)$ instead of $O(n \log n)$ per split.
2. **Sparsity-Aware Split Finding**: Missing values are natively handled by routing them to a "default direction" (left or right child) that maximizes the Gain.
3. **Column Block for Parallel Learning**: Data is sorted once in-memory and stored in blocks to allow parallel split finding.
