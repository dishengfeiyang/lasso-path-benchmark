import numpy as np
import time
from sklearn.linear_model import enet_path
from ._base import BasePathMethod


class ElasticNet(BasePathMethod):
    """
    Elastic Net path method.
    """

    def __init__(self, l1_ratio=0.5, max_iter=100000, tol=1e-4):
        super().__init__("ElasticNet")
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y, lambdas, groups=None):
        n_samples, n_features = X.shape
        n_lambdas = len(lambdas)

        # sklearn enet_path minimizes:
        # (1/(2n)) * ||y - Xw||^2 + alpha * (l1_ratio * ||w||_1 + 0.5*(1-l1_ratio)*||w||^2)
        # Our objective:
        # 0.5 * ||y - Xw||^2 + lam * (l1_ratio * ||w||_1 + 0.5*(1-l1_ratio)*||w||^2)
        # Hence alpha = lam / n_samples
        alpha_list = np.array(lambdas) / n_samples

        start = time.time()
        alphas, coefs, _ = enet_path(
            X, y,
            l1_ratio=self.l1_ratio,
            alphas=alpha_list,
            max_iter=self.max_iter,
            tol=self.tol
        )
        total_time = time.time() - start

        coef_path = coefs

        objective = np.zeros(n_lambdas)
        sparsity = np.zeros(n_lambdas, dtype=int)
        for j, lam in enumerate(lambdas):
            beta = coef_path[:, j]
            residual = y - X @ beta
            objective[j] = 0.5 * np.sum(residual ** 2) + lam * (
                self.l1_ratio * np.sum(np.abs(beta)) +
                0.5 * (1 - self.l1_ratio) * np.sum(beta ** 2)
            )
            sparsity[j] = np.sum(beta != 0)

        timing = np.full(n_lambdas, total_time / n_lambdas)

        return {
            "coef_path": coef_path,
            "objective": objective,
            "sparsity": sparsity,
            "timing": timing,
            "lambdas": np.array(lambdas),
        }