"""
Hardware-Aware Tree Method Benchmark
Compares exact, approx, and hist on synthetic tabular fraud data.
Measures wall-clock training time, speedup vs exact, and PR-AUC validation parity.
"""

import json
import os
import time

import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split


def generate_benchmark_data(n_samples: int = 10000, random_state: int = 42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=30,
        n_informative=18,
        n_redundant=6,
        n_clusters_per_class=2,
        weights=[0.98, 0.02],
        flip_y=0.01,
        random_state=random_state,
    )
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=random_state)


def run_benchmark(n_samples_list=(10000, 50000), output_json_path=None):
    tree_methods = ["exact", "approx", "hist"]
    results = {}

    for n in n_samples_list:
        results[str(n)] = {}
        X_train, X_val, y_train, y_val = generate_benchmark_data(n_samples=n)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        exact_time = None

        for method in tree_methods:
            params = {
                "tree_method": method,
                "max_depth": 6,
                "learning_rate": 0.1,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": 42,
                "nthread": -1,
            }

            start_t = time.perf_counter()
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                evals=[(dval, "val")],
                verbose_eval=False,
            )
            elapsed = time.perf_counter() - start_t

            if method == "exact":
                exact_time = elapsed

            preds = bst.predict(dval)
            pr_auc = float(average_precision_score(y_val, preds))
            speedup = float(exact_time / elapsed) if exact_time and elapsed > 0 else 1.0

            results[str(n)][method] = {
                "wall_clock_sec": round(elapsed, 4),
                "speedup_vs_exact": round(speedup, 2),
                "pr_auc": round(pr_auc, 4),
            }

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def print_markdown_table(results):
    print("\n### Hardware Tree Method Benchmark Results (Ryzen 7 5700U CPU)\n")
    print("| Dataset Size | Method | Wall Time (s) | Speedup vs Exact | PR-AUC |")
    print("|---|---|---|---|---|")
    for n_str, methods in results.items():
        for method, metrics in methods.items():
            print(
                f"| N={int(n_str):,} | `{method}` | {metrics['wall_clock_sec']:.2f}s | "
                f"{metrics['speedup_vs_exact']:.2f}x | {metrics['pr_auc']:.4f} |"
            )


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, "tree_method_benchmark.json")
    res = run_benchmark(n_samples_list=(10000, 50000), output_json_path=json_path)
    print_markdown_table(res)
    print(f"\nSaved benchmark metrics to {json_path}")
