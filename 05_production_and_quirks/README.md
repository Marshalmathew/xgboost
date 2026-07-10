# 05 - Production Quirks and Hacks

This module transitions from theory to real-world engineering. 

## 1. Monotonic Constraints
Sometimes, domain knowledge dictates a monotonic relationship. For example, all else being equal, a higher credit score should NEVER decrease the probability of a loan approval. 
- You can enforce this using the `monotone_constraints` parameter (1 for increasing, -1 for decreasing, 0 for off).
- XGBoost will mathematically enforce this at every split, heavily regularizing the model and preventing weird edge-case predictions in production.

## 2. Handling Imbalanced Data
In fraud detection, positives might be 0.1% of the data.
- **Option A**: `scale_pos_weight = sum(negative instances) / sum(positive instances)`. This scales the gradient of the positive class.
- **Option B**: Use AUC-PR as your evaluation metric instead of standard AUC-ROC.
- **Option C**: Use `max_delta_step` (e.g., set to 1-10) to help convergence in logistic regression when classes are extremely imbalanced.

## 3. Categorical Features
As of version 1.5, XGBoost natively supports categorical features (similar to LightGBM).
- You no longer need to One-Hot Encode (which inflates the feature space and hurts tree algorithms).
- Set `enable_categorical=True` and ensure Pandas columns are of type `category`.
- Alternatively, Target Encoding works extremely well for high-cardinality categoricals.

## 4. Hardware Acceleration
- `tree_method='hist'` is generally the fastest CPU method.
- `tree_method='gpu_hist'` (or `device='cuda'` in newer versions) leverages GPUs. A must-have for huge datasets.

## 5. Serialization and Model Serving
- DO NOT use Python `pickle`. It is insecure and tied to the Python version.
- Use `bst.save_model('model.json')`.
- For ultra-fast C++ serving, export models to **Treelite** or **ONNX**.
