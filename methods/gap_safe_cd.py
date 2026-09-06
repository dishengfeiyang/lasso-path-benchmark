import numpy as np
import time
from sklearn.linear_model import lasso_path
from ._base import BasePathMethod


class GapSafeCD(BasePathMethod):
    """
    基于 gap safe 预筛 + sklearn 高效求解的 Lasso 路径方法。
    对每个 lambda，先通过上一步解计算 gap safe 筛选得到候选集，
    再在候选集上调用 lasso_path 求解，保证精度并得到筛选比例。
    """

    def __init__(self, max_iter=100000, tol=1e-4):
        super().__init__("GapSafeCD")
        self.max_iter = max_iter
        self.tol = tol
        self.screened_ratios = []

    def _soft_threshold(self, x, kappa):
        return np.sign(x) * np.maximum(np.abs(x) - kappa, 0)

    def _construct_dual_feasible(self, X, y, beta, lam):
        r = X @ beta - y
        Xtr = X.T @ r
        rho = max(1.0, np.max(np.abs(Xtr)) / lam)
        theta = -r / rho
        return theta

    def _dual_gap(self, X, y, beta, theta, lam):
        primal = 0.5 * np.sum((y - X @ beta) ** 2) + lam * np.sum(np.abs(beta))
        dual = 0.5 * np.sum(y ** 2) - 0.5 * np.sum(theta ** 2)
        return primal - dual

    def _gap_safe_screening(self, X, y, beta, lam):
        n_features = X.shape[1]
        theta = self._construct_dual_feasible(X, y, beta, lam)
        gap = self._dual_gap(X, y, beta, theta, lam)

        if gap <= 0:
            return np.arange(n_features), 0.0

        Xt_theta = X.T @ theta
        X_norms = np.sqrt(np.sum(X ** 2, axis=0))
        # 注意：theta 满足 |X^T theta| <= lam，所以阈值用 lam，半径不除以 lam
        upper_bounds = np.abs(Xt_theta) + X_norms * np.sqrt(2 * gap)
        active = np.where(upper_bounds >= lam)[0]

        # 确保当前 beta 中的非零特征保留
        non_zero = np.where(np.abs(beta) > 1e-12)[0]
        active = np.union1d(active, non_zero)

        screened_ratio = 1.0 - len(active) / n_features
        return active, screened_ratio

    def fit(self, X, y, lambdas, groups=None):
        n_samples, n_features = X.shape
        n_lambdas = len(lambdas)

        coef_path = np.zeros((n_features, n_lambdas))
        objective = np.zeros(n_lambdas)
        sparsity = np.zeros(n_lambdas, dtype=int)
        timing = np.zeros(n_lambdas)
        screened_ratios = np.zeros(n_lambdas)

        beta = np.zeros(n_features)

        for j, lam in enumerate(lambdas):
            start = time.time()

            if j == 0:
                # 第一个 lambda 不筛选，直接在全特征上求解
                active = np.arange(n_features)
                screened_ratios[j] = 0.0
            else:
                active, screened_ratios[j] = self._gap_safe_screening(X, y, beta, lam)
                # 确保非零特征包含在内
                non_zero = np.where(np.abs(beta) > 1e-12)[0]
                active = np.union1d(active, non_zero)

            if len(active) == 0:
                beta = np.zeros(n_features)
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

            end = time.time()

            beta[np.abs(beta) < 1e-8] = 0.0
            coef_path[:, j] = beta
            residual = y - X @ beta
            objective[j] = 0.5 * np.sum(residual ** 2) + lam * np.sum(np.abs(beta))
            sparsity[j] = np.sum(beta != 0)
            timing[j] = end - start

        self.screened_ratios = screened_ratios
        return {
            "coef_path": coef_path,
            "objective": objective,
            "sparsity": sparsity,
            "timing": timing,
            "lambdas": np.array(lambdas),
            "screened_ratios": screened_ratios,
        }