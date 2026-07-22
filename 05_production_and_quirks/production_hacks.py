# %% [markdown]
# # Production Quirks, Serialization, and ONNX Conversion
# 
# Learn how to enforce Monotonic Constraints, use native categorical features, serialize models (JSON/UBJ), and convert models to ONNX runtime for ultra-low latency production serving.

# %%
import xgboost as xgb
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Monotonic Constraints
# Synthetic scenario: risk decreases with age, plus random noise.

# %%
np.random.seed(42)
ages = np.random.uniform(18, 80, 500)
risk = -2.0 * ages + np.random.normal(0, 30, 500)

X = pd.DataFrame({'Age': ages})
y = risk

dtrain = xgb.DMatrix(X, label=y)

params_standard = {'objective': 'reg:squarederror', 'max_depth': 3, 'eta': 0.1}
bst_standard = xgb.train(params_standard, dtrain, num_boost_round=50)

params_constrained = {
    'objective': 'reg:squarederror', 
    'max_depth': 3, 
    'eta': 0.1,
    'monotone_constraints': '(-1)' # Decreasing constraint
}
bst_constrained = xgb.train(params_constrained, dtrain, num_boost_round=50)

# %% [markdown]
# ## 2. Native Categorical Features (XGBoost >= 1.5)

# %%
df = pd.DataFrame({
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'London', 'Paris', 'New York'],
    'Value': [100, 200, 150, 300, 210, 140, 110]
})

df['City'] = df['City'].astype('category')
X_cat = df[['City']]
y_cat = df['Value']

dtrain_cat = xgb.DMatrix(X_cat, label=y_cat, enable_categorical=True)

params_cat = {
    'tree_method': 'hist',
    'max_depth': 2
}
bst_cat = xgb.train(params_cat, dtrain_cat, num_boost_round=10)

# %% [markdown]
# ## 3A. Base Model Serialization (JSON & UBJ)
# `.json` and `.ubj` are modern cross-platform serialization formats supported natively by XGBoost.

# %%
# Save as JSON
bst_cat.save_model("model.json")

# Save as Universal Binary JSON (UBJ - faster binary load)
bst_cat.save_model("model.ubj")

# Load model back
loaded_bst = xgb.Booster()
loaded_bst.load_model("model.json")
print("JSON Model loaded successfully!")

# %% [markdown]
# ## 3B. Advanced Serialization & Serving: ONNX Conversion (`onnxruntime`)
# ONNX (Open Neural Network Exchange) provides a high-performance C++ inference engine (`onnxruntime`) for sub-millisecond serving.

# %%
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnxruntime as rt

    # Train a scikit-learn compatible XGBoost regressor for ONNX conversion
    xgb_reg = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
    xgb_reg.fit(X, y)

    # Define ONNX initial input type
    initial_types = [('float_input', FloatTensorType([None, X.shape[1]]))]

    # Convert model to ONNX format
    onnx_model = convert_sklearn(xgb_reg, initial_types=initial_types)
    with open("model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    print("\nONNX Model exported successfully to 'model.onnx'!")

    # Latency Benchmark: Native XGBoost vs ONNX Runtime
    X_test_arr = X.values.astype(np.float32)

    # Native XGBoost inference benchmark
    start_native = time.time()
    for _ in range(1000):
        _ = xgb_reg.predict(X_test_arr)
    native_duration = (time.time() - start_native) * 1000 / 1000  # ms per batch

    # ONNX Runtime inference benchmark
    sess = rt.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    
    start_onnx = time.time()
    for _ in range(1000):
        _ = sess.run(None, {input_name: X_test_arr})
    onnx_duration = (time.time() - start_onnx) * 1000 / 1000  # ms per batch

    print(f"Native XGBoost Latency: {native_duration:.4f} ms per batch")
    print(f"ONNX Runtime Latency:   {onnx_duration:.4f} ms per batch")
    print(f"ONNX Speedup Factor:    {native_duration / onnx_duration:.2f}x faster")

except ImportError:
    print("\nNote: Install 'skl2onnx' and 'onnxruntime' to run ONNX model conversion benchmark.")

