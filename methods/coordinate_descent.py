import numpy as np
import time
from sklearn.linear_model import lasso_path
from ._base import BasePathMethod


class CoordinateDescent(BasePathMethod):
    def __init__(self):
        super().__init__("CoordinateDescent")

    def fit(self, X, y, lambdas, groups=None):
        n_samples, n_features = X.shape
        n_lambdas = len(lambdas)

        # 目标函数：0.5 * ||y - Xw||^2 + lambda * ||w||_1
        # sklearn lasso_path 使用 (1/(2*n_samples)) * ||y - Xw||^2 + alpha * ||w||_1
        # 因此 alpha = lambda / n_samples
        alpha_list = np.array(lambdas) / n_samples

        start = time.time()
        alphas, coefs, _ = lasso_path(
            X, y,
            alphas=alpha_list,
            max_iter=100000,
            tol=1e-6
        )
        total_time = time.time() - start

        coef_path = coefs  # shape (n_features, n_lambdas)

        objective = np.zeros(n_lambdas)
        sparsity = np.zeros(n_lambdas, dtype=int)
        for j, lam in enumerate(lambdas):
            beta = coef_path[:, j]
            residual = y - X @ beta
            objective[j] = 0.5 * np.sum(residual ** 2) + lam * np.sum(np.abs(beta))
            sparsity[j] = np.sum(beta != 0)

        timing = np.full(n_lambdas, total_time / n_lambdas)

        return {
            "coef_path": coef_path,
            "objective": objective,
            "sparsity": sparsity,
            "timing": timing,
            "lambdas": np.array(lambdas),
        }