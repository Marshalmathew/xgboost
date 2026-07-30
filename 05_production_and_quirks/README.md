# 05 - Production Quirks and Hacks

This module transitions from theory to real-world engineering. In production, data is rarely perfectly balanced, models need to be saved securely, and inference latency is measured in milliseconds.

## 1. Handling Extreme Class Imbalance

In many real-world scenarios (like fraud detection or disease diagnosis), the "positive" class you care about might only represent 0.1% of the data. 
A naive model will simply learn to predict "0" for every instance, achieving 99.9% accuracy while failing completely at its actual job.

XGBoost provides built-in mechanisms to handle this:
- **`scale_pos_weight`**: This parameter mathematically scales the gradient (the penalty) of the positive class. A standard heuristic is setting it to `sum(negative instances) / sum(positive instances)`. This forces the tree to pay attention to the minority class.
- **Evaluation Metric (`aucpr`)**: Never use standard Accuracy or even ROC-AUC for extreme imbalance. Use **Precision-Recall AUC (aucpr)**. It strictly evaluates how well the model identifies the minority class without being artificially inflated by True Negatives.
- **`max_delta_step`**: Setting this to a finite number (e.g., 1-10) can help convergence in logistic regression when classes are extremely imbalanced.

## 2. Hardware Acceleration
- `tree_method='hist'` is generally the fastest CPU method and should be your default.
- `tree_method='gpu_hist'` (or `device='cuda'` in newer versions) leverages NVIDIA GPUs. This is an absolute necessity when dealing with millions of rows.

## 3. Serialization (Saving the Model)
- **DO NOT use Python `pickle`**. It is insecure (can execute arbitrary code upon loading) and tightly coupled to specific Python versions.
- **JSON**: Use `bst.save_model('model.json')`. It is human-readable, cross-platform, and backwards compatible.
- **UBJSON**: Universal Binary JSON (`model.ubj`) provides the same cross-platform guarantees as JSON but loads significantly faster because it is a binary format.

## 4. Ultra-Fast Serving (ONNX)
If your production API requires sub-millisecond latency (e.g., high-frequency trading or real-time ad bidding), the native Python XGBoost `predict()` method might be too slow.
- You can export your XGBoost model to **ONNX** (Open Neural Network Exchange).
- You can then serve the ONNX file using `onnxruntime` (a highly optimized C++ engine), often resulting in massive latency speedups compared to native Python execution.

---

## 💻 Module Contents (Code)

1. [production_hacks.ipynb](./production_hacks.ipynb)
   - **The Imbalance Trap**: Generates a 99-to-1 imbalanced dataset. Shows how a naive model achieves 99% accuracy but a 0% Recall (catching zero fraud). Then trains a model using `scale_pos_weight` and `aucpr` to successfully isolate the minority class.
   - **Serialization**: Demonstrates saving and loading models natively via JSON and UBJSON.
   - **ONNX Benchmark**: Converts an XGBoost model to ONNX format and runs a live latency benchmark comparing native Python prediction speed vs ONNX C++ Runtime speed.
