# Self-Test Answer Sheet — Module 02: XGBoost Core Mechanics

> **Protocol**: Complete this active retrieval self-check **before** reviewing the mathematical derivations in Module 02. 
> These 5 questions test your understanding of how XGBoost operates under the hood.

---

## 📝 Part 1: Core Mathematical Diagnostic Questions

### Q1: Why does XGBoost use a 2nd-order Taylor expansion instead of 1st-order gradient descent?
* **Your Initial Hypothesis**:
* **Mathematical Reality**:
  Classical GBM (Friedman 2001) takes a first-order step in the direction of the negative gradient, requiring a separate scalar line search to determine step length. 
  By taking a 2nd-order Taylor expansion around the current margin $\hat{y}^{(t-1)}$:
  $$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$
  XGBoost formulates a quadratic function with a known global minimum. Differentiating yields an exact, analytical closed-form step:
  $$w^* = -\frac{g}{h + \lambda}$$
  This is a regularized **Newton-Raphson update**. The Hessian $h_i$ acts as a per-sample adaptive learning rate: small curvature yields large confident steps, whereas steep curvature scales back the step size, eliminating line search entirely.

---

### Q2: How is the optimal leaf weight $w_j^*$ derived?
* **Your Initial Hypothesis**:
* **Mathematical Reality**:
  For all samples $i$ routed to terminal leaf $j$ (instance set $I_j$), let $G_j = \sum_{i \in I_j} g_i$ and $H_j = \sum_{i \in I_j} h_i$. The objective simplified for leaf $j$ is:
  $$\tilde{\mathcal{L}}_j = G_j w_j + \frac{1}{2} (H_j + \lambda) w_j^2 + \gamma$$
  Setting the derivative with respect to $w_j$ to zero:
  $$\frac{\partial \tilde{\mathcal{L}}_j}{\partial w_j} = G_j + (H_j + \lambda) w_j = 0 \implies w_j^* = -\frac{G_j}{H_j + \lambda}$$
  Substituting $w_j^*$ back gives the maximal score reduction achieved by that leaf structure:
  $$\tilde{\mathcal{L}}_j^* = -\frac{1}{2} \frac{G_j^2}{H_j + \lambda} + \gamma$$

---

### Q3: What does the Exact Split Gain formula measure, and why is $\gamma$ subtracted?
* **Your Initial Hypothesis**:
* **Mathematical Reality**:
  The split Gain evaluates the difference in optimal objective value between the parent node $I$ and its proposed left ($I_L$) and right ($I_R$) children:
  $$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G_I^2}{H_I + \lambda} \right] - \gamma$$
  - The bracketed term is the raw loss reduction.
  - Adding a split increases the total number of leaves by 1 ($T_{\text{new}} = T_{\text{old}} + 1$). The complexity penalty $\Omega(f)$ contains the term $\gamma T$.
  - Therefore, the score reduction must exceed $\gamma$ to justify adding a leaf; otherwise $\text{Gain} \le 0$ and the split is pruned.

---

### Q4: Why must candidate split points be spaced by Hessian mass, rather than row count or feature values?
* **Your Initial Hypothesis**:
* **Mathematical Reality**:
  Rewriting the second-order Taylor objective (Chen & Guestrin 2016, Eq. 3):
  $$\tilde{\mathcal{L}}^{(t)} = \sum_{i=1}^n \frac{1}{2} h_i \left( f_t(x_i) - \left(-\frac{g_i}{h_i}\right) \right)^2 + \Omega(f_t) + \text{const}$$
  The objective is mathematically identical to a **weighted squared error regression** targeting $-\frac{g_i}{h_i}$, with instance weight $h_i$.
  - For logloss, $h_i = p_i(1 - p_i)$. Highly confident samples ($p_i \approx 0$ or $1$) have near-zero Hessian ($h_i \to 0$); uncertain samples ($p_i \approx 0.5$) have maximal Hessian ($h_i = 0.25$).
  - A candidate split in a high-Hessian uncertainty zone shifts the loss function orders of magnitude more than one in a low-Hessian zone. Spacing candidates by Hessian mass concentrates split evaluation capacity where prediction uncertainty is greatest.

---

### Q5: What is the fundamental difference between `tree_method='approx'` and `tree_method='hist'`?
* **Your Initial Hypothesis**:
* **Mathematical Reality**:
  - `approx` (Weighted Quantile Sketch): Re-computes or queries candidate quantile buckets per tree or per node using current sample Hessians.
  - `hist` (Histogram Binned Engine): Pre-computes fixed discrete histograms (typically 256 bins) **once** at the start of training using raw feature values. During tree boosting, it aggregates $G$ and $H$ into these pre-allocated bins in $\mathcal{O}(N \times K)$ time with zero sketching overhead, delivering a **$6.25\times$ speedup** over `approx` on modern CPUs.
