import numpy as np
import time
from ._base import BasePathMethod


class GroupLARS(BasePathMethod):
    def __init__(self, max_iter=200, tol=1e-4, inner_max_iter=20, inner_tol=1e-6):
        super().__init__("GroupLARS")
        self.max_iter = max_iter
        self.tol = tol
        self.inner_max_iter = inner_max_iter
        self.inner_tol = inner_tol

    def _group_correlation(self, X, residual, groups):
        corr = np.zeros(len(groups))
        for g_idx, group in enumerate(groups):
            X_g = X[:, group]
            corr[g_idx] = np.linalg.norm(X_g.T @ residual)
        return corr

    def _update_group(self, X_g, residual, lam, beta_g_init):
        """使用迭代岭回归求解 Group Lasso 的组更新子问题"""
        beta_g = beta_g_init.copy()
        n_features_g = X_g.shape[1]
        # 处理初始为零的情况
        if np.linalg.norm(beta_g) < 1e-10:
            beta_g = X_g.T @ residual
            if np.linalg.norm(beta_g) < 1e-12:
                return np.zeros_like(beta_g)

        for _ in range(self.inner_max_iter):
            norm_beta = np.linalg.norm(beta_g)
            if norm_beta < 1e-12:
                return np.zeros_like(beta_g)
            A = X_g.T @ X_g + (lam / norm_beta) * np.eye(n_features_g)
            b = X_g.T @ residual
            try:
                beta_g_new = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                beta_g_new = np.linalg.pinv(A) @ b
            if np.linalg.norm(beta_g_new - beta_g) < self.inner_tol:
                beta_g = beta_g_new
                break
            beta_g = beta_g_new
        return beta_g

    def fit(self, X, y, lambdas, groups=None):
        if groups is None:
            raise ValueError("GroupLARS requires groups.")

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
            # 计算组相关性
            corr = self._group_correlation(X, residual, groups)
            active = [idx for idx, c in enumerate(corr) if c > lam]
            if not active:
                active = [int(np.argmax(corr))]

            for _ in range(self.max_iter):
                beta_old = beta.copy()
                max_diff = 0.0

                for g_idx in active:
                    group = groups[g_idx]
                    X_g = X[:, group]
                    # 从残差中移除当前组的贡献
                    residual += X_g @ beta[group]

                    # 更新组系数
                    beta_new = self._update_group(X_g, residual, lam, beta[group])

                    # 更新残差和系数
                    residual -= X_g @ beta_new
                    beta[group] = beta_new

                    diff = np.linalg.norm(beta_new - beta_old[group])
                    if diff > max_diff:
                        max_diff = diff

                if max_diff < self.tol:
                    break

            end = time.time()

            # 后处理
            beta[np.abs(beta) < 1e-8] = 0.0

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