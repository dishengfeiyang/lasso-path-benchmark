import numpy as np
import time
from sklearn.linear_model import lasso_path
from ._base import BasePathMethod


class ScreeningCoordinateDescent(BasePathMethod):
    """
    使用 Strong rules 的坐标下降路径求解器。
    对每个 λ，利用前一个 λ 的解估计候选活跃集，
    只在该子集上调用 lasso_path，从而大幅减少计算量。
    """

    def __init__(self, max_iter=200000, tol=1e-4):
        super().__init__("ScreeningCD")
        self.max_iter = max_iter
        self.tol = tol
        self.screened_ratios = []

    def fit(self, X, y, lambdas, groups=None):
        n_samples, n_features = X.shape
        n_lambdas = len(lambdas)

        coef_path = np.zeros((n_features, n_lambdas))
        objective = np.zeros(n_lambdas)
        sparsity = np.zeros(n_lambdas, dtype=int)
        timing = np.zeros(n_lambdas)
        screened_ratios = np.zeros(n_lambdas)

        beta = np.zeros(n_features)
        residual = y.copy()

        for j, lam in enumerate(lambdas):
            start = time.time()

            # ---------- Strong rule 筛选 ----------
            if j == 0:
                corr = X.T @ residual
                active = np.where(np.abs(corr) >= lam)[0]
            else:
                active_prev = np.where(beta != 0)[0]
                corr = X.T @ residual
                threshold = lam
                potentially_active = np.where(np.abs(corr) >= threshold)[0]
                active = np.union1d(active_prev, potentially_active)

            screened_ratio = 1.0 - len(active) / n_features
            screened_ratios[j] = screened_ratio
            self.screened_ratios.append(screened_ratio)

            if len(active) == 0:
                beta = np.zeros(n_features)
                residual = y.copy()
            else:
                X_active = X[:, active]
                alpha = np.array([lam]) / n_samples
                _, coefs_active, _ = lasso_path(
                    X_active, y,
                    alphas=alpha,
                    max_iter=self.max_iter,
                    tol=self.tol
                )
                beta_active = coefs_active[:, 0]
                beta = np.zeros(n_features)
                beta[active] = beta_active
                residual = y - X @ beta

            end = time.time()

            beta[np.abs(beta) < 1e-8] = 0.0
            coef_path[:, j] = beta
            obj_res = y - X @ beta
            objective[j] = 0.5 * np.sum(obj_res ** 2) + lam * np.sum(np.abs(beta))
            sparsity[j] = np.sum(beta != 0)
            timing[j] = end - start

        return {
            "coef_path": coef_path,
            "objective": objective,
            "sparsity": sparsity,
            "timing": timing,
            "lambdas": np.array(lambdas),
            "screened_ratios": screened_ratios,
        }