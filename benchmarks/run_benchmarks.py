import os
import sys
import tracemalloc
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.real_datasets import load_real_dataset
from data.synthetic import make_group_sparse_regression
from methods.lars import LarsPath
from methods.coordinate_descent import CoordinateDescent
from methods.admm import ADMM
from methods.block_cd import BlockCoordinateDescent
from methods.celer import Celer
from methods.group_lars import GroupLARS
from methods.fista import FISTA
from methods.screening_cd import ScreeningCoordinateDescent
from methods.anderson_cd import AndersonCD
from methods.elastic_net import ElasticNet
from methods.gap_safe_cd import GapSafeCD
from benchmarks.utils import create_lambda_sequence
from benchmarks.metrics import coefficient_error


METHODS = {
    "lars": LarsPath,
    "coordinate_descent": CoordinateDescent,
    "admm": ADMM,
    "block_cd": BlockCoordinateDescent,
    "celer": Celer,
    "group_lars": GroupLARS,
    "fista": FISTA,
    "screening_cd": ScreeningCoordinateDescent,
    "anderson_cd": AndersonCD,
    "elastic_net": ElasticNet,
    "gap_safe_cd": GapSafeCD,
}

METHOD_TYPES = {
    "lars": "lasso",
    "coordinate_descent": "lasso",
    "admm": "lasso",
    "celer": "lasso",
    "fista": "lasso",
    "screening_cd": "lasso",
    "anderson_cd": "lasso",
    "elastic_net": "lasso",
    "gap_safe_cd": "lasso",
    "block_cd": "group_lasso",
    "group_lars": "group_lasso",
}

GROUP_METHODS = {"block_cd", "group_lars"}


def generate_lambdas_for_dataset(X, y, groups, n_lambdas, ratio, include_group_methods):
    """生成 λ 序列，兼容 Lasso 和 Group Lasso 混合场景。"""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_samples = X.shape[0]
    ratio = float(ratio)

    # 如果同时包含 Lasso 和 Group Lasso 方法，并且 groups 存在，
    # 则取两种 lambda_max 的较大者，确保所有方法都从全零解开始。
    if include_group_methods and groups is not None:
        # Lasso 的 lambda_max
        lasso_max = np.max(np.abs(X.T @ y)) / n_samples
        # Group Lasso 的 lambda_max
        group_norm_max = 0.0
        for g in groups:
            norm = np.linalg.norm(X[:, g].T @ y)
            if norm > group_norm_max:
                group_norm_max = norm
        group_max = group_norm_max / n_samples
        # 取较大者，并确保为标量
        lambda_max = max(float(lasso_max), float(group_max))
        if lambda_max <= 0:
            lambda_max = 1e-12
        lambda_min = lambda_max * ratio
        return np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), n_lambdas))
    else:
        # 仅 Lasso 或仅 Group Lasso，直接使用 utils 中的函数
        return create_lambda_sequence(X, y, n_lambdas=n_lambdas, ratio=ratio, groups=groups)


def run_experiment(method_name, X, y, lambdas, groups=None):
    """运行单个方法实验，返回包含完整路径和内存信息的结果字典。"""
    if method_name in GROUP_METHODS and groups is None:
        print(f"  Skipping {method_name} (requires groups, but groups is None)")
        return None

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # 统一中心化
    X_mean = np.mean(X, axis=0)
    y_mean = np.mean(y)
    X_centered = X - X_mean
    y_centered = y - y_mean

    method = METHODS[method_name]()
    tracemalloc.start()
    result = method.fit(X_centered, y_centered, lambdas, groups=groups)
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result["peak_memory"] = peak_memory  # bytes

    return result


def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    method_names = config["methods"]
    include_group = any(m in GROUP_METHODS for m in method_names)
    n_lambdas = config["lambda_settings"]["n_lambdas"]
    ratio = float(config["lambda_settings"].get("ratio", 1e-4))

    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    tables_dir = os.path.join(output_dir, "tables")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    results_list = []

    for dataset in config["datasets"]:
        name = dataset["name"]
        is_synthetic = dataset.get("synthetic", False)

        if is_synthetic:
            print(f"Generating synthetic dataset: {name}")
            params = dataset["params"]
            X, y, beta_true, groups = make_group_sparse_regression(**params)
        else:
            print(f"Loading real dataset: {name}")
            X, y, groups = load_real_dataset(name)
            beta_true = None

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_features = X.shape[1]

        # 根据方法列表生成 λ 序列
        lambdas = generate_lambdas_for_dataset(
            X, y, groups, n_lambdas, ratio, include_group
        )

        for method_name in method_names:
            print(f"  Running method: {method_name}")
            result = run_experiment(method_name, X, y, lambdas, groups)
            if result is None:
                continue

            # 保存完整路径到 .npz
            save_path = os.path.join(results_dir, f"{name}_{method_name}.npz")
            np.savez(
                save_path,
                coef_path=result["coef_path"],
                objective=result["objective"],
                sparsity=result["sparsity"],
                timing=result["timing"],
                lambdas=np.array(lambdas),
                peak_memory=result["peak_memory"],
            )

            # 汇总指标
            total_time = np.sum(result["timing"])
            avg_objective = np.mean(result["objective"])
            final_sparsity = result["sparsity"][-1]
            peak_memory_mb = result["peak_memory"] / 1024**2  # 转换为 MB

            coeff_err = np.nan
            if beta_true is not None:
                coeff_err = coefficient_error(result["coef_path"][:, -1], beta_true)

            screened_ratio = np.nan
            if "screened_ratios" in result:
                screened_ratio = np.mean(result["screened_ratios"])

            results_list.append({
                "dataset": name,
                "method": method_name,
                "method_type": METHOD_TYPES[method_name],
                "total_time": total_time,
                "avg_objective": avg_objective,
                "final_sparsity": final_sparsity,
                "coefficient_error": coeff_err,
                "screened_ratio": screened_ratio,
                "peak_memory_mb": peak_memory_mb,
                "n_features": n_features,
                "result_file": os.path.relpath(save_path, output_dir),
            })

    if results_list:
        df = pd.DataFrame(results_list)
        output_path = os.path.join(tables_dir, "results.csv")
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    else:
        print("No results were generated.")


if __name__ == "__main__":
    main()