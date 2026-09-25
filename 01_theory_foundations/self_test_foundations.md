# Self-Test Answer Sheet — Module 01: Theoretical Foundations

> **Protocol**: Complete this active retrieval self-check **before** reading through the mathematical derivations in Module 01. 
> Write down your intuition first, then verify against the reference text to expose conceptual gaps.

---

## 📝 Part 1: Core Conceptual Retrieval

### 1. CART vs. Classical Decision Trees (ID3 / C4.5)
| Diagnostic Question | Your Answer / Intuition | Mathematical Ground Truth |
|:---|:---|:---|
| **What does a leaf node actually output in CART vs. ID3?** | | In ID3/C4.5, leaves output a categorical label or discrete probability distribution. In CART (Classification and Regression Trees), **leaves always output a continuous real-valued scalar score $w \in \mathbb{R}$**, even for classification tasks. The final label is produced by passing the sum of leaf scores through a link function (e.g. sigmoid). |
| **Why is the "diagonal decision boundary" expensive for CART?** | | CART splits are strictly axis-aligned ($x_j \le s$). Approximating a smooth linear diagonal boundary (e.g. $x_1 + x_2 > 1$) requires a staircase of many orthogonal splits, demanding exponential tree depth and heavy data density in boundary corners. |

---

### 2. Impurity Measures & Splitting Mechanics
| Impurity Metric | Mathematical Formula | Key Operational Property |
|:---|:---|:---|
| **Variance / MSE** (Regression) | $\frac{1}{\|S\|} \sum_{i \in S} (y_i - \bar{y}_S)^2$ | Optimal leaf prediction is the node sample mean $\hat{w} = \bar{y}_S$, which analytically minimizes squared residual error. |
| **Gini Impurity** (Classification) | $\sum_{k=1}^K p_k (1 - p_k) = 1 - \sum_{k=1}^K p_k^2$ | Measures the probability that a randomly chosen element from the set would be incorrectly labeled if randomly labeled according to the class distribution. Computationally faster than Entropy (no logarithms). |
| **Cross-Entropy / Info Gain** | $-\sum_{k=1}^K p_k \log_2(p_k)$ | Directly measures reduction in information uncertainty (mutual information) between feature split indicator and class distribution. |

---

### 3. Ensemble Paradigms: Bagging vs. Boosting
| Dimension | Bagging (Bootstrap Aggregation / Random Forest) | Boosting (Gradient Boosting / GBM) |
|:---|:---|:---|
| **Base Learner Training** | In **parallel** on independent bootstrap samples. | In **sequence**; each tree fits the residual errors of preceding trees. |
| **Primary Error Reduction** | Primarily reduces **variance** without increasing bias: $\text{Var}(\bar{X}) = \frac{\sigma^2}{B} + \frac{B-1}{B} \rho \sigma^2$. | Primarily reduces **bias** and variance by iteratively minimizing the empirical risk functional. |
| **Base Learner Complexity** | Deep, unpruned, high-capacity trees (low bias, high variance). | Shallow, regularized weak learners (stumps or depth 3–6). |
| **Sensitivity to Outliers** | Highly robust; outlier impact is diluted across parallel averages. | Vulnerable unless regularized; boosting can focus excessive attention on hard-to-fit noisy outliers. |

---

## 🔍 Part 2: The Gradient Descent Connection (Friedman 2001)

### Why is it called *Gradient* Boosting?
- When our loss function is Squared Error $L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$, the negative gradient with respect to prediction $\hat{y}$ is:
  $$-\frac{\partial L}{\partial \hat{y}} = y - \hat{y} = \text{Residual}$$
- Therefore, fitting a regression tree to the residual errors is mathematically equivalent to taking a step in the direction of **steepest descent in function space** (Friedman 2001).
- For arbitrary loss functions (Logistic Loss, Huber Loss, Poisson), pseudo-residuals are computed as the negative gradient evaluated at the current prediction:
  $$r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}$$
