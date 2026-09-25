# Self-Test Answer Sheet — Module 04: Advanced Governance & Interpretability

> **Protocol**: Complete this active retrieval self-check **before** studying Module 04. 
> These questions assess your knowledge of TreeSHAP game theory, monotonic constraints, and regulatory model governance (SR 11-7 / ECOA).

---

## 📝 Part 1: Diagnostic Questions

### Q1: Why is TreeSHAP polynomial time $\mathcal{O}(TLD^2)$, whereas KernelSHAP is exponential $\mathcal{O}(2^M)$?
* **Your Initial Hypothesis**:
* **Mathematical Reality**:
  Classical cooperative game theory calculates Shapley values by summing marginal contributions across all $2^{|F|}$ possible feature coalitions $S \subseteq F \setminus \{i\}$. For $M=50$ features, $2^{50} \approx 10^{15}$ evaluations.
  TreeSHAP (Lundberg et al., 2020) exploits the internal tree structure: by recursively caching the sample proportions flowing down each decision path when feature $i$ is absent from coalition $S$, it computes exact expectations $\mathbb{E}[f(x) \mid S]$ in a single recursive pass down the tree:
  $$\text{Complexity} = \mathcal{O}(T \cdot L \cdot D^2)$$
  Where $T$ is the number of trees, $L$ is the number of leaves, and $D$ is the maximum depth.

---

### Q2: What is the "Correlated Features Pathology" in TreeSHAP (Kumar et al., ICML 2020)?
* **Your Initial Hypothesis**:
* **Mathematical Reality**:
  Shapley formulation assumes features can be marginalized independently. When two features are collinear (e.g. `annual_income` and `loan_amount`, $r = 0.75$):
  1. **Off-Manifold Evaluations**: Evaluating coalitions where `loan_amount` is included but `annual_income` is conditioned out evaluates synthetic points in impossible regions of feature space (e.g., an applicant with \$20,000 income requesting a \$500,000 loan).
  2. **Arbitrary Credit Splitting**: Because tree greedy splitting chooses one collinear feature over another based on slight sample noise at each depth, the ensemble splits credit arbitrarily, generating contradictory or misleading adverse action explanations.

---

### Q3: How do monotonic constraints mechanically modify the split-finding algorithm?
* **Your Initial Hypothesis**:
* **Mathematical Reality**:
  When a positive monotonic constraint ($c_j = +1$) is assigned to feature $j$:
  1. The candidate split evaluates optimal child weights $w_L^* = -\frac{G_L}{H_L + \lambda}$ and $w_R^* = -\frac{G_R}{H_R + \lambda}$.
  2. The candidate split is **only accepted if $w_L^* \le w_R^*$**.
  3. If $w_L^* > w_R^*$, the split is strictly discarded, regardless of how high its score reduction ($\Delta \text{Gain}$) would have been.
  4. In addition, the upper and lower bounds on leaf predictions propagate up and down adjacent branches to guarantee global monotonicity across the entire tree.

---

### Q4: What is the bias-variance trade-off when applying monotonic constraints?
* **Your Initial Hypothesis**:
* **Mathematical Reality**:
  Monotonic constraints inject a strong domain prior ($w_R \ge w_L$).
  - **Variance Reduction**: In data-sparse boundary tails (e.g. very high income), unconstrained models overfit to random noise and outliers. Monotonic constraints prune these spurious reversals, acting as a structural regularizer (yielding a $+0.56$ AUC regularization bonus in our credit audit).
  - **Structural Bias**: If the true physical phenomenon has genuine non-monotonic inflection points (e.g. debt-to-income risk flattening beyond a certain threshold), the constraint enforces monotonicity artificially, potentially increasing bias slightly. It is an intentional trade-off: trading marginal fit for regulatory defensibility and business logic.

---

### Q5: How do interaction constraints prevent illegal proxy discrimination under ECOA Reg B?
* **Your Initial Hypothesis**:
* **Mathematical Reality**:
  Under the Equal Credit Opportunity Act (ECOA), models must not learn indirect proxy interactions that penalize protected classes (e.g. interacting credit inquiries with geographical zip code or income).
  By passing:
  ```python
  params["interaction_constraints"] = [
      ["annual_income", "loan_amount", "debt_to_income"],  # Group 1: Financial Capacity
      ["credit_inquiries_12m", "delinquencies_2yr"]        # Group 2: Bureau Behavior
  ]
  ```
  XGBoost strictly forbids any decision tree branch from splitting on a Group 1 feature if that branch already contains a Group 2 split (and vice versa). This is verified by checking `bst.get_dump()`.
