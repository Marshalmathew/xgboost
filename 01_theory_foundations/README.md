# 01 - Theory Foundations: The Pre-Requisites for XGBoost

Before diving into Extreme Gradient Boosting (XGBoost), it is crucial to understand the foundational algorithms it builds upon: **Decision Trees** and **Ensemble Learning (Bagging vs. Boosting)**. 

## 1. Decision Trees (CART)
XGBoost is an ensemble of Classification and Regression Trees (CART). Unlike standard decision trees that output categorical classes (like "Fraud" or "Not Fraud"), CARTs contain real-valued scores in each of their leaves, regardless of whether the task is classification or regression.

### The Anatomy of a Decision Tree
- **Root Node**: The entire dataset.
- **Internal Nodes**: Splitting rules (e.g., `Age > 30`).
- **Leaf Nodes**: The final output values (scores).

### Splitting Criteria (How does a tree learn?)
Trees learn by finding splits that maximize information gain (or minimize impurity).
- **Gini Impurity** (Classification): Measures the probability of misclassifying a randomly chosen element if it were randomly labeled according to the distribution of labels in the subset.
  \[ Gini = 1 - \sum (p_i)^2 \]
- **Mean Squared Error (MSE)** (Regression): Measures the variance of the target variable in the node.

## 2. Ensemble Learning: The Wisdom of the Crowd
A single decision tree is prone to **overfitting** (high variance, low bias). It memorizes the training data. Ensemble methods combine multiple weak learners to create a strong learner.

### Bagging (Bootstrap Aggregating)
- **Concept**: Train multiple independent trees in *parallel* on random subsets of the data (with replacement).
- **Goal**: Reduce variance.
- **Example**: Random Forest.
- **Analogy**: A committee of experts voting on an outcome.

### Boosting
- **Concept**: Train trees *sequentially*. Each new tree tries to correct the errors (residuals) made by the previous trees.
- **Goal**: Reduce bias (and variance, with regularization).
- **Example**: AdaBoost, Gradient Boosting, XGBoost.
- **Analogy**: A student learning from their mistakes on previous exams.

## 3. Gradient Boosting Machines (GBM)
GBM is the framework that XGBoost optimizes. Instead of tweaking the weights of misclassified instances (like AdaBoost), GBM fits new models to the **residuals** (the errors) of the previous models.

### The Intuition
1. We want to predict house prices, $Y$.
2. Our first model, $F_1(x)$, predicts a baseline price (e.g., the mean).
3. We calculate the error (residual): $h_1(x) = Y - F_1(x)$.
4. We train a new tree, $T_1(x)$, not to predict $Y$, but to predict the residual $h_1(x)$.
5. Our new model is $F_2(x) = F_1(x) + \eta T_1(x)$, where $\eta$ is the learning rate.

By repeatedly adding trees that predict the remaining errors, the ensemble gradually converges to the true target $Y$. The process is called **Gradient** Boosting because fitting a tree to the residuals is mathematically equivalent to taking a step in the direction of the negative gradient of the loss function (gradient descent in function space).

> **Note**: Why does XGBoost exist if GBM already does this? 
> Standard GBM is slow and prone to overfitting. XGBoost introduces Second-Order Taylor Expansion (Hessians), hardware optimization (cache-awareness, out-of-core computing), and robust regularization ($\Omega$) to solve these problems. We cover this in Module 2!
