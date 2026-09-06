import numpy as np
import time
from ._base import BasePathMethod


class FISTA(BasePathMethod):
    def __init__(self, max_iter=20000, tol=1e-6):
        super().__init__("FISTA")
        self.max_iter = max_iter
        self.tol = tol

    def _soft_threshold(self, x, kappa):
        return np.sign(x) * np.maximum(np.abs(x) - kappa, 0)

    def _estimate_lipschitz(self, X):
        """估计 Lipschitz 常数 L = ||X||_2^2"""
        n_features = X.shape[1]
        if n_features <= 1000:
            # 低维时直接精确计算谱范数平方，避免幂法误差
            return np.linalg.norm(X, 2) ** 2
        else:
            # 高维时使用多次随机初始化幂法
            return self._power_iteration_lipschitz(X, n_init=5)

    def _power_iteration_lipschitz(self, X, n_init=5, max_iter=100):
        """多次随机初始化幂法估计最大特征值"""
        n = X.shape[1]
        max_eig = 0.0
        for _ in range(n_init):
            v = np.random.randn(n)
            v /= np.linalg.norm(v)
            eig = 0.0
            for _ in range(max_iter):
                Xv = X @ v
                XtXv = X.T @ Xv
                eig_new = np.linalg.norm(XtXv)
                if eig_new < 1e-12:
                    break
                v_new = XtXv / eig_new
                if np.linalg.norm(v_new - v) < 1e-6:
                    v = v_new
                    eig = eig_new
                    break
                v = v_new
                eig = eig_new
            if eig > max_eig:
                max_eig = eig
        return max_eig

    def fit(self, X, y, lambdas, groups=None):
        n_samples, n_features = X.shape
        n_lambdas = len(lambdas)

        coef_path = np.zeros((n_features, n_lambdas))
        objective = np.zeros(n_lambdas)
        sparsity = np.zeros(n_lambdas, dtype=int)
        timing = np.zeros(n_lambdas)

        # 估计 Lipschitz 常数
        L = self._estimate_lipschitz(X)
        if L < 1e-12:
            L = 1.0

        beta = np.zeros(n_features)  # 用于 warm start（上一个 lambda 的解）

        for j, lam in enumerate(lambdas):
            start = time.time()

            # 从上一个 lambda 的解 warm start
            beta_k = beta.copy()
            yk = beta_k.copy()
            t_k = 1.0
            prev_obj = np.inf

            for it in range(self.max_iter):
                # 梯度步
                grad = X.T @ (X @ yk - y)
                beta_new = self._soft_threshold(yk - grad / L, lam / L)

                # 梯度重启条件（动量方向与更新方向不一致时重启）
                if it > 0:
                    restart = np.dot(yk - beta_new, beta_new - beta_k) > 0
                else:
                    restart = False

                if restart:
                    t_k = 1.0
                    yk = beta_k.copy()
                    grad = X.T @ (X @ yk - y)
                    beta_new = self._soft_threshold(yk - grad / L, lam / L)

                # 计算目标值（用于收敛判断）
                obj = 0.5 * np.sum((y - X @ beta_new) ** 2) + lam * np.sum(np.abs(beta_new))

                # 检查收敛：相对目标值变化和系数变化
                if abs(prev_obj - obj) / max(1.0, abs(prev_obj)) < self.tol or \
                   np.linalg.norm(beta_new - beta_k) < self.tol * (1 + np.linalg.norm(beta_k)):
                    beta_k = beta_new
                    break

                prev_obj = obj

                # 更新动量
                t_new = (1 + np.sqrt(1 + 4 * t_k * t_k)) / 2
                yk = beta_new + ((t_k - 1) / t_new) * (beta_new - beta_k)

                beta_k = beta_new
                t_k = t_new

            # 后处理：截断接近零的系数
            beta_k[np.abs(beta_k) < 1e-8] = 0.0

            end = time.time()

            coef_path[:, j] = beta_k
            residual = y - X @ beta_k
            objective[j] = 0.5 * np.sum(residual ** 2) + lam * np.sum(np.abs(beta_k))
            sparsity[j] = np.sum(beta_k != 0)
            timing[j] = end - start

            # 更新 warm start
            beta = beta_k.copy()

        return {
            "coef_path": coef_path,
            "objective": objective,
            "sparsity": sparsity,
            "timing": timing,
            "lambdas": np.array(lambdas),
        }