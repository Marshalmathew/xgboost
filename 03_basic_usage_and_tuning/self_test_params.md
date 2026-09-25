# Self-Test Answer Sheet — Day 2: Hyperparameter Failure Modes

> **Protocol**: This answer sheet represents the *active retrieval practice* phase recommended in Step 3 of the Day 2 curriculum. 
> Fill in your hypotheses **before** checking the official reference documentation or the empirical ablation results. 
> The disagreements identified between your intuitive mental model and real engine mechanics become your highest-yield study signals.

---

## 📝 Part 1: Pre-Check Intuition & Failure Mode Hypotheses

### Group 1: Tree Structure & Complexity
| Parameter | What happens when Too HIGH? | What happens when Too LOW? |
|---|---|---|
| `max_depth` (default=6) | **Overfitting / High Variance**: Trees grow deeply into sample noise; leaves isolate single data points or spurious feature combinations; inference latency and memory balloon. | **Underfitting / High Bias**: Inability to capture multi-feature interaction effects; model behaves like shallow linear decision stumps. |
| `min_child_weight` (default=1) | **Underfitting**: Requires enormous statistical evidence (sum of Hessians) to permit any split; tree stops growing prematurely, missing localized non-linear boundaries. | **Overfitting**: Splits on arbitrary noise or single outlier rows (leaves with near-zero Hessian); extreme instability on noisy leaves. |
| `gamma` / `min_split_loss` (default=0) | **Underfitting**: Demands excessively high score gain $\Delta \text{Gain} > \gamma$ to justify node splitting; leads to heavily pruned, shallow, conservative trees. | **Overfitting**: Any marginal positive gain (even $\epsilon = 10^{-7}$) creates splits, capturing dataset artifacts and spurious variance. |

---

### Group 2: Regularization (Leaf Weight Penalties)
| Parameter | What happens when Too HIGH? | What happens when Too LOW? |
|---|---|---|
| `reg_lambda` (L2, default=1) | **Underfitting / Zero-Shrinkage**: Leaf weights shrink asymptotically to zero: $w^* = -\frac{G}{H + \lambda} \to 0$. Predictions remain anchored to `base_score`, suppressing model learning. | **Exploding Leaf Weights**: When Hessian sum $H$ is small, leaf weights $w^* \approx -G/H$ oscillate wildly, amplifying sensitivity to noisy training samples. |
| `reg_alpha` (L1, default=0) | **Over-Sparsification / Inactive Leaves**: High L1 drives leaf weights exactly to zero when $|G| \le \alpha$, converting informative leaves into non-contributing dead nodes. | **Dense Non-Zero Weights**: No L1 feature or weight sparsity; all leaves contribute non-zero residuals regardless of marginal contribution. |

---

### Group 3: Sampling & Stochastic Randomness
| Parameter | What happens when Too HIGH? | What happens when Too LOW? |
|---|---|---|
| `subsample` (default=1.0) | **Ensemble Correlation**: Each tree trains on identical data rows; lacks bootstrap diversity; higher variance on noisy datasets. | **High Variance across Trees / Underfitting**: Each tree sees too few instances; gradient/hessian estimates become noisy and inaccurate. |
| `colsample_bytree` (default=1.0) | **Dominant Feature Greed**: Dominant predictive features are chosen at the root of every tree, preventing weaker but complementary features from being explored. | **Feature Starvation**: Trees may not have access to sufficient informative features at split time, degrading split quality. |
| `colsample_bylevel` (default=1.0) | Every tree level evaluates the same features that survived `colsample_bytree`, potentially maintaining intra-level correlation. | Aggressive feature restriction at each depth level can starve sub-branches of necessary splitting features. |
| `colsample_bynode` (default=1.0) | Every node split searches across the entire level feature set. | Extreme randomness at individual node level; can destabilize greedy split finding if informative features are repeatedly excluded. |

---

### Group 4: Boosting Dynamics & Optimization Control
| Parameter | What happens when Too HIGH? | What happens when Too LOW? |
|---|---|---|
| `learning_rate` / `eta` (default=0.3) | **Overshooting & Instability**: Steps along the gradient are too large; optimization oscillates or diverges; validation loss spikes early. | **Slow Convergence & Underfitting**: Enormous number of trees required to reach optimal loss; training runtime balloons; risk of stopping before convergence. |
| `n_estimators` / `num_boost_round` | **Overfitting (without early stopping)**: Model continues learning idiosyncratic residuals and memorizing training labels. | **Underfitting**: Optimization terminates before the ensemble reaches the error plateau. |
| `early_stopping_rounds` | **Delayed Halt**: Overfits for many unnecessary rounds before declaring stopping; wastes compute. | **Premature Stopping**: A momentary plateau or noisy validation fluctuation trips the stopping criterion before the model begins converging. |

---

### Group 5: Class Imbalance Control
| Parameter | What happens when Too HIGH? | What happens when Too LOW? |
|---|---|---|
| `scale_pos_weight` (default=1.0) | **High False Positive Surge / Destroyed Precision**: Model aggressively shifts probability mass upward; flags ordinary transactions as fraud, overwhelming ops teams. | **Low Positive Recall**: In severe imbalance (e.g., 98:2), model ignores the minority class because default logloss is dominated by the majority class gradients. |

---

## 🔍 Part 2: Disagreements & Insights After Cross-Checking with Docs & Ablation

### 1. Hessian Guard (`min_child_weight`) Misconception
- **Common Mental Trap**: Believing `min_child_weight` is a simple sample count (like `min_samples_leaf` in scikit-learn).
- **Engine Reality**: In classification, $h_i = p_i(1 - p_i)$. Confident samples ($p_i \approx 0.99$ or $0.01$) contribute almost zero Hessian ($h_i \approx 0.0099$). Therefore, 100 confident samples can have a combined Hessian sum $\sum h_i < 1.0$. `min_child_weight` measures **statistical uncertainty**, not raw sample count!

### 2. `scale_pos_weight` vs. Operational Alert Volume (The Banking Trap)
- **Common Mental Trap**: Optimizing `scale_pos_weight` purely to maximize AUROC.
- **Engine Reality**: AUROC is threshold-invariant and ranks pairs of positive and negative samples. However, in fraud and AML operations, investigative capacity is strictly capped (e.g., $\le 5\%$ of incoming volume). Setting `scale_pos_weight = sum(neg)/sum(pos)` calibrates log-odds toward minority sensitivity, but operational deployment requires evaluating **Precision and Recall at the fixed operational alert threshold**, not just global AUC.

### 3. Gamma vs. Depth Pruning Mechanics
- **Common Mental Trap**: Assuming `gamma` and `max_depth` do the same thing because both control tree size.
- **Engine Reality**: `max_depth` is a hard architectural cutoff applied uniformly across all branches. In contrast, `gamma` performs **adaptive, gain-dependent post-pruning**: a branch can grow to depth 8 if a high-gain split exists, while an uninformative split at depth 2 is pruned if its gain $\le \gamma$.

### 4. L2 Regularization ($\lambda$) in the Score Denominator
- **Common Mental Trap**: Believing $\lambda$ only penalizes large weights like in linear ridge regression.
- **Engine Reality**: In XGBoost, $\lambda$ appears directly in the denominator of both the optimal leaf weight $w^* = -\frac{G}{H + \lambda}$ and the split gain formula $\text{Score} = \frac{G^2}{H + \lambda}$. Hence, $\lambda$ simultaneously acts as a leaf shrinkage parameter and a split gain dampener.
