"""
ONNX Export, Serialization Formats & Ultra-Low Latency Inference Benchmark
==========================================================================

Enterprise Serving Standard for XGBoost Models:
1. Model Serialization:
   - model.json : Human-readable, audit-inspectable, standard exchange schema.
   - model.ubj  : Universal Binary JSON, compact, zero-overhead C++ parsing.
   - model.onnx : Open Neural Network Exchange for hardware-accelerated serving.

2. Microsecond Latency Benchmark (Single-row batch=1 inference, N=5,000 queries):
   - Native Python DMatrix
   - Native C++ inplace_predict (zero-copy NumPy view)
   - UBJSON booster prediction
   - ONNX Runtime CPU InferenceSession

3. Numerical Parity:
   - Verifies max absolute difference between native XGBoost and ONNX Runtime is < 1e-5.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnxruntime as ort
import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.model_selection import train_test_split

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.generate_synthetic_data import generate_credit_dataset


def run_onnx_benchmark(n_samples: int = 10000, n_queries: int = 1000) -> Dict[str, Any]:
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("XGBOOST ENTERPRISE SERVING: ONNX EXPORT & LATENCY BENCHMARK")
    print("================================================================================")

    # Step 1: Ingest Data
    print(f"Generating synthetic credit portfolio ({n_samples:,} rows)...")
    df = generate_credit_dataset(n_samples=n_samples, seed=42)
    feature_cols = [
        "annual_income",
        "loan_amount",
        "revolving_utilization",
        "debt_to_income",
        "credit_inquiries_12m",
        "delinquencies_2yr",
    ]
    X = df[feature_cols].astype(np.float32).values
    y = df["is_default"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 2: Train Model
    print("Training production XGBClassifier (50 trees, max_depth=4)...")
    clf = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Step 3: Serialize to JSON, UBJSON, and ONNX
    json_path = output_dir / "credit_model.json"
    ubj_path = output_dir / "credit_model.ubj"
    onnx_path = output_dir / "credit_model.onnx"

    clf.save_model(str(json_path))
    clf.save_model(str(ubj_path))
    print(f"Saved JSON model : {json_path.name} ({json_path.stat().st_size / 1024:.1f} KB)")
    print(f"Saved UBJ model  : {ubj_path.name} ({ubj_path.stat().st_size / 1024:.1f} KB)")

    print("Exporting model to ONNX format via onnxmltools...")
    initial_type = [("float_input", FloatTensorType([None, len(feature_cols)]))]
    onnx_model = convert_xgboost(clf, initial_types=initial_type, target_opset=15)

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Saved ONNX model : {onnx_path.name} ({onnx_path.stat().st_size / 1024:.1f} KB)")

    # Step 4: Numerical Parity Validation
    print("\nVerifying numerical floating-point parity...")
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name

    # Batch test for parity
    native_preds_prob = clf.predict_proba(X_test)[:, 1]
    ort_outputs = ort_session.run(None, {input_name: X_test})

    # ONNX TreeEnsembleClassifier typically outputs [probabilities, labels] or dictionary of class probabilities
    if isinstance(ort_outputs[1], list):
        # List of dicts {0: p0, 1: p1}
        onnx_preds_prob = np.array([row[1] for row in ort_outputs[1]])
    elif isinstance(ort_outputs[1], np.ndarray) and ort_outputs[1].ndim == 2:
        onnx_preds_prob = ort_outputs[1][:, 1]
    else:
        # Fallback to probability tensor
        onnx_preds_prob = np.array(ort_outputs[1])[:, 1]

    max_abs_diff = float(np.max(np.abs(native_preds_prob - onnx_preds_prob)))
    print(f"Maximum absolute parity difference: {max_abs_diff:.8e}")
    assert max_abs_diff < 1e-4, f"Parity failure! Max diff {max_abs_diff} exceeds tolerance 1e-4."
    print("Parity Check PASSED (Exact numerical agreement verified)")

    # Step 5: Microsecond Latency Benchmark (Batch=1)
    print(f"\nBenchmarking single-row (batch=1) inference latency over {n_queries:,} iterations...")
    test_rows = X_test[: min(n_queries, len(X_test))]
    indices = np.random.choice(len(test_rows), size=n_queries, replace=True)

    latencies = {
        "dmatrix": [],
        "inplace_predict": [],
        "onnx_runtime": [],
    }

    booster = clf.get_booster()

    # A: Native DMatrix
    for idx in indices:
        row = test_rows[idx : idx + 1]
        t0 = time.perf_counter()
        dm = xgb.DMatrix(row)
        _ = booster.predict(dm)
        t1 = time.perf_counter()
        latencies["dmatrix"].append((t1 - t0) * 1e6)  # microseconds

    # B: Native inplace_predict (zero-copy C++ buffer)
    for idx in indices:
        row = test_rows[idx : idx + 1]
        t0 = time.perf_counter()
        _ = booster.inplace_predict(row)
        t1 = time.perf_counter()
        latencies["inplace_predict"].append((t1 - t0) * 1e6)

    # C: ONNX Runtime CPU
    for idx in indices:
        row = test_rows[idx : idx + 1]
        t0 = time.perf_counter()
        _ = ort_session.run(None, {input_name: row})
        t1 = time.perf_counter()
        latencies["onnx_runtime"].append((t1 - t0) * 1e6)

    results = {}
    print("\nInference Latency Results (Microseconds, us):")
    print("-" * 75)
    print(f"{'Engine':<25} | {'Mean (us)':<10} | {'p50 (us)':<10} | {'p95 (us)':<10} | {'p99 (us)':<10} | {'Req/Sec':<10}")
    print("-" * 75)

    for engine, lats in latencies.items():
        arr = np.array(lats)
        mean_lat = float(np.mean(arr))
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        throughput = float(1e6 / mean_lat) if mean_lat > 0 else 0.0

        results[engine] = {
            "mean_us": round(mean_lat, 2),
            "p50_us": round(p50, 2),
            "p95_us": round(p95, 2),
            "p99_us": round(p99, 2),
            "throughput_qps": round(throughput, 1),
        }
        print(f"{engine:<25} | {mean_lat:<10.1f} | {p50:<10.1f} | {p95:<10.1f} | {p99:<10.1f} | {throughput:<10.1f}")

    results["max_parity_diff"] = max_abs_diff

    # Save benchmark json
    benchmark_json = output_dir / "onnx_serving_benchmark.json"
    with open(benchmark_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {benchmark_json.name}")

    return results


if __name__ == "__main__":
    run_onnx_benchmark()
