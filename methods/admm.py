import numpy as np
import time
from scipy.linalg import cho_factor, cho_solve
from ._base import BasePathMethod


class ADMM(BasePathMethod):
    def __init__(self, rho=1.0, max_iter=1000, tol=1e-5):
        super().__init__("ADMM")
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol

    def _soft_threshold(self, v, kappa):
        return np.sign(v) * np.maximum(np.abs(v) - kappa, 0)

    def fit(self, X, y, lambdas, groups=None):
        n_samples, n_features = X.shape
        n_lambdas = len(lambdas)

        coef_path = np.zeros((n_features, n_lambdas))
        objective = np.zeros(n_lambdas)
        sparsity = np.zeros(n_lambdas, dtype=int)
        timing = np.zeros(n_lambdas)

        XtX = X.T @ X
        Xty = X.T @ y

        if n_samples < n_features:
            XXt = X @ X.T
            L = cho_factor(XXt + self.rho * np.eye(n_samples))
            use_woodbury = True
        else:
            L = cho_factor(XtX + self.rho * np.eye(n_features))
            use_woodbury = False

        beta = np.zeros(n_features)
        z = np.zeros(n_features)
        w = np.zeros(n_features)

        for j, lam in enumerate(lambdas):
            start = time.time()
            for _ in range(self.max_iter):
                b = Xty + self.rho * (z - w)
                if use_woodbury:
                    tmp = X @ b
                    q = cho_solve(L, tmp)
                    beta = (1.0 / self.rho) * b - (1.0 / (self.rho ** 2)) * (X.T @ q)
                else:
                    beta = cho_solve(L, b)

                z_old = z.copy()
                v = beta + w
                z = self._soft_threshold(v, lam / self.rho)

                w += beta - z

                r_prim = np.linalg.norm(beta - z)
                r_dual = self.rho * np.linalg.norm(z - z_old)
                if r_prim < self.tol and r_dual < self.tol:
                    break
            end = time.time()

            # 后处理：将绝对值小于阈值的系数置零
            beta[np.abs(beta) < 1e-4] = 0.0

            coef_path[:, j] = beta
            residual = y - X @ beta
            objective[j] = 0.5 * np.sum(residual ** 2) + lam * np.sum(np.abs(beta))
            sparsity[j] = np.sum(beta != 0)
            timing[j] = end - start

        return {
            "coef_path": coef_path,
            "objective": objective,
            "sparsity": sparsity,
            "timing": timing,
            "lambdas": np.array(lambdas),
        }