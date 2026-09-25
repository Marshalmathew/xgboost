# Self-Test Answer Sheet — Module 05: Production Deployment & Serving

> **Protocol**: Complete this active retrieval self-check **before** studying Module 05. 
> These questions assess your readiness to engineer, serialize, and govern XGBoost models in high-throughput enterprise environments.

---

## 📝 Part 1: Diagnostic Questions

### Q1: What is the "Raw Margin Space Trap" in custom XGBoost objectives?
* **Your Initial Hypothesis**:
* **Engineering Reality**:
  When defining a custom objective:
  ```python
  def custom_obj(preds: np.ndarray, dtrain: xgb.DMatrix):
      labels = dtrain.get_label()
      # ... compute grad, hess ...
      return grad, hess
  ```
  The argument `preds` is passed as **raw untransformed margin values ($z \in (-\infty, +\infty)$)**, NOT probabilities.
  - If you assume `preds` is a probability $p \in (0, 1)$ and compute $g = \text{preds} - y$, gradient descent diverges catastrophically.
  - You MUST explicitly evaluate the link function:
    $$p = \frac{1}{1 + e^{-z}}$$
  - In addition, if you provide a `custom_metric(preds, dtrain)`, `preds` is also in margin space.

---

### Q2: What three mathematical requirements must any custom loss function satisfy?
* **Your Initial Hypothesis**:
* **Engineering Reality**:
  1. **$C^2$ Differentiability**: The loss function must possess continuous first and second derivatives with respect to the margin ($g_i$ and $h_i$).
  2. **Row Additivity**: The objective must decompose as $\sum_{i=1}^n l(y_i, \hat{y}_i)$ without row-to-row dependencies (e.g. sorting-based metrics like AUC cannot be directly computed as an elementwise objective).
  3. **Strict Positive Convexity**: The second derivative must be strictly positive ($h_i > 0$) for all inputs. If $h_i \le 0$, the quadratic approximation is concave, having no minimum.

---

### Q3: What happens when a custom objective returns a negative Hessian ($h_i < 0$)?
* **Your Initial Hypothesis**:
* **Engineering Reality**:
  XGBoost does **NOT** throw a runtime error or raise an exception. Instead, in the C++ engine (`src/tree/updater_colmaker.cc`), it **silently clips** $h_i \to \epsilon \approx 10^{-16}$.
  - When $h_i \approx 0$, the optimal leaf weight degenerates to $w^* \approx -G / \lambda$.
  - The split evaluation metrics become corrupted without emitting warnings, causing the model to learn suboptimal splits quietly. A non-convex custom loss fails silently, not loudly!

---

### Q4: Why is an Asymmetric Utility Loss mathematically superior to `scale_pos_weight` in AML operations?
* **Your Initial Hypothesis**:
* **Engineering Reality**:
  - `scale_pos_weight = N_neg / N_pos` (e.g. $s = 49.0$) is a **frequency rebalancing correction**. It inflates all positive gradients uniformly, artificially shifting predicted probabilities upward and destroying probability calibration. On borderline cases ($p \approx 0.5$), it pulls gradients $4.9\times$ too hard, causing massive false-alarm surges.
  - A hand-derived Asymmetric Loss ($k = \text{Cost}_{\text{FN}} / \text{Cost}_{\text{FP}} = 10.0$) directly optimizes **decision-theoretic business loss**:
    $$L(y, p) = - [ k \cdot y \ln(p) + (1 - y) \ln(1 - p) ]$$
  - It maintains calibrated log-odds while penalizing false negatives according to actual operational economics (\$5,000 fine vs \$500 review), delivering **+\$254,500 in net cost savings (+14.7%)** by avoiding false alarm floods.

---

### Q5: How does XGBoost handle missing values natively, and when does default routing fail?
* **Your Initial Hypothesis**:
* **Engineering Reality**:
  - During training, for every split candidate on feature $j$, XGBoost evaluates two options for samples where $x_{ij} = \text{NaN}$: (1) route all NaNs to the left child, or (2) route all NaNs to the right child. It permanently bakes the direction yielding higher Gain into the tree structure.
  - **The Production Failure Mode**: If a feature had zero missing values during training, the tree has no learned NaN direction. In test data, if an upstream API emits `NaN` for that feature, XGBoost defaults to the right branch by convention, potentially routing unknown applicants into severe risk leaves. Always explicitly test missingness during training!
