import numpy as np
import time
from ._base import BasePathMethod


class BlockCoordinateDescent(BasePathMethod):
    def __init__(self, max_iter=200, tol=1e-4):
        super().__init__("BlockCD")
        self.max_iter = max_iter
        self.tol = tol

    def _group_soft_threshold(self, u, kappa):
        norm_u = np.linalg.norm(u)
        if norm_u <= kappa:
            return np.zeros_like(u)
        return (1 - kappa / norm_u) * u

    def fit(self, X, y, lambdas, groups=None):
        if groups is None:
            raise ValueError("BlockCD requires groups.")

        n_samples, n_features = X.shape
        n_lambdas = len(lambdas)

        coef_path = np.zeros((n_features, n_lambdas))
        objective = np.zeros(n_lambdas)
        sparsity = np.zeros(n_lambdas, dtype=int)
        timing = np.zeros(n_lambdas)

        beta = np.zeros(n_features)
        residual = y.copy()

        for j, lam in enumerate(lambdas):
            start = time.time()

            for _ in range(self.max_iter):
                beta_old = beta.copy()
                max_diff = 0.0

                for group in groups:
                    X_g = X[:, group]
                    # 从残差中移除旧贡献
                    residual += X_g @ beta[group]

                    beta_ls = np.linalg.lstsq(X_g, residual, rcond=None)[0]
                    beta_new = self._group_soft_threshold(beta_ls, lam)

                    residual -= X_g @ beta_new
                    beta[group] = beta_new

                    diff = np.linalg.norm(beta_new - beta_old[group])
                    if diff > max_diff:
                        max_diff = diff

                if max_diff < self.tol:
                    break

            end = time.time()

            coef_path[:, j] = beta
            obj_res = y - X @ beta
            objective[j] = 0.5 * np.sum(obj_res ** 2) + lam * sum(
                np.linalg.norm(beta[g]) for g in groups
            )
            sparsity[j] = np.sum(beta != 0)
            timing[j] = end - start

        return {
            "coef_path": coef_path,
            "objective": objective,
            "sparsity": sparsity,
            "timing": timing,
            "lambdas": np.array(lambdas),
        }