import os
import time
import tracemalloc
import numpy as np
import pandas as pd
from data.synthetic import make_group_sparse_regression
from methods.coordinate_descent import CoordinateDescent
from methods.admm import ADMM
from methods.block_cd import BlockCoordinateDescent
from benchmarks.utils import create_lambda_sequence


def run_one(method, X, y, lambdas, groups=None):
    tracemalloc.start()
    start = time.time()
    result = method.fit(X, y, lambdas, groups=groups)
    elapsed = time.time() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / 1024**2  # MB


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "tables")
    os.makedirs(output_dir, exist_ok=True)

    # 定义扩展性实验配置
    # 变化样本量
    n_samples_list = [100, 200, 500]
    fixed_p = 100
    fixed_n_lambdas = 50

    results = []

    for n_samples in n_samples_list:
        X, y, beta_true, groups = make_group_sparse_regression(
            n_samples=n_samples, n_features=fixed_p, n_informative=10,
            n_groups=20, rho=0.5, noise=0.1, random_state=42
        )
        X_c = X - X.mean(axis=0)
        y_c = y - y.mean()
        lambdas = create_lambda_sequence(X_c, y_c, n_lambdas=fixed_n_lambdas, ratio=1e-4)

        for name, cls in [("coordinate_descent", CoordinateDescent),
                          ("admm", ADMM),
                          ("block_cd", BlockCoordinateDescent)]:
            if name == "block_cd":
                groups_arg = groups
            else:
                groups_arg = None
            method = cls()
            t, mem = run_one(method, X_c, y_c, lambdas, groups=groups_arg)
            results.append({
                "experiment": "vary_n",
                "n_samples": n_samples,
                "n_features": fixed_p,
                "n_lambdas": fixed_n_lambdas,
                "method": name,
                "time": t,
                "memory_mb": mem,
            })

    # 变化特征维度
    p_list = [50, 100, 200]
    fixed_n = 200
    for p in p_list:
        X, y, beta_true, groups = make_group_sparse_regression(
            n_samples=fixed_n, n_features=p, n_informative=10,
            n_groups=20, rho=0.5, noise=0.1, random_state=42
        )
        X_c = X - X.mean(axis=0)
        y_c = y - y.mean()
        lambdas = create_lambda_sequence(X_c, y_c, n_lambdas=fixed_n_lambdas, ratio=1e-4)

        for name, cls in [("coordinate_descent", CoordinateDescent),
                          ("admm", ADMM),
                          ("block_cd", BlockCoordinateDescent)]:
            if name == "block_cd":
                groups_arg = groups
            else:
                groups_arg = None
            method = cls()
            t, mem = run_one(method, X_c, y_c, lambdas, groups=groups_arg)
            results.append({
                "experiment": "vary_p",
                "n_samples": fixed_n,
                "n_features": p,
                "n_lambdas": fixed_n_lambdas,
                "method": name,
                "time": t,
                "memory_mb": mem,
            })

    # 变化路径点数量
    nl_list = [20, 50, 100]
    for nl in nl_list:
        X, y, beta_true, groups = make_group_sparse_regression(
            n_samples=200, n_features=100, n_informative=10,
            n_groups=20, rho=0.5, noise=0.1, random_state=42
        )
        X_c = X - X.mean(axis=0)
        y_c = y - y.mean()
        lambdas = create_lambda_sequence(X_c, y_c, n_lambdas=nl, ratio=1e-4)

        for name, cls in [("coordinate_descent", CoordinateDescent),
                          ("admm", ADMM),
                          ("block_cd", BlockCoordinateDescent)]:
            if name == "block_cd":
                groups_arg = groups
            else:
                groups_arg = None
            method = cls()
            t, mem = run_one(method, X_c, y_c, lambdas, groups=groups_arg)
            results.append({
                "experiment": "vary_n_lambdas",
                "n_samples": 200,
                "n_features": 100,
                "n_lambdas": nl,
                "method": name,
                "time": t,
                "memory_mb": mem,
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "scaling_results.csv"), index=False)
    print("Scaling results saved.")


if __name__ == "__main__":
    main()