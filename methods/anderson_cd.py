import numpy as np
import time
from ._base import BasePathMethod


class AndersonCD(BasePathMethod):
    """
    带重启和阻尼的 Anderson 加速坐标下降。
    修复了之前数值发散的问题。
    """

    def __init__(self, max_iter=500, tol=1e-6, M=5, K=5, reg=1e-8, damping=0.5):
        super().__init__("AndersonCD")
        self.max_iter = max_iter
        self.tol = tol
        self.M = M
        self.K = K
        self.reg = reg
        self.damping = damping

    def _soft_threshold(self, x, kappa):
        return np.sign(x) * np.maximum(np.abs(x) - kappa, 0)

    def _objective(self, X, y, beta, lam):
        residual = y - X @ beta
        return 0.5 * np.sum(residual ** 2) + lam * np.sum(np.abs(beta))

    def _coordinate_descent_step(self, X, y, beta, lam, lipschitz):
        beta = beta.copy()
        residual = y - X @ beta
        n_features = X.shape[1]

        for j in range(n_features):
            X_j = X[:, j]
            old = beta[j]
            rho = X_j @ residual + lipschitz[j] * old
            beta_j = self._soft_threshold(rho / lipschitz[j], lam / lipschitz[j])
            residual += X_j * (old - beta_j)
            beta[j] = beta_j

        return beta

    def _anderson(self, X_hist, R_hist):
        K = X_hist.shape[1]
        if K < 2:
            return X_hist[:, -1] + R_hist[:, -1]

        dX = np.diff(X_hist, axis=1)
        dR = np.diff(R_hist, axis=1)
        A = dR.T @ dR + self.reg * np.eye(K - 1)
        b = -dR.T @ R_hist[:, -1]
        try:
            gamma = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            gamma = np.zeros(K - 1)

        x_new = X_hist[:, -1] + R_hist[:, -1] - (dX + dR) @ gamma
        return x_new

    def fit(self, X, y, lambdas, groups=None):
        n_samples, n_features = X.shape
        n_lambdas = len(lambdas)

        coef_path = np.zeros((n_features, n_lambdas))
        objective = np.zeros(n_lambdas)
        sparsity = np.zeros(n_lambdas, dtype=int)
        timing = np.zeros(n_lambdas)

        lipschitz = np.sum(X ** 2, axis=0)
        lipschitz[lipschitz < 1e-12] = 1.0

        beta = np.zeros(n_features)

        for j, lam in enumerate(lambdas):
            start = time.time()
            X_hist = np.zeros((n_features, 0))
            R_hist = np.zeros((n_features, 0))
            obj_best = np.inf
            beta_best = beta.copy()

            for it in range(self.max_iter):
                beta_new = self._coordinate_descent_step(X, y, beta, lam, lipschitz)
                r = beta_new - beta

                if X_hist.shape[1] >= self.K:
                    X_hist = np.delete(X_hist, 0, axis=1)
                    R_hist = np.delete(R_hist, 0, axis=1)
                X_hist = np.column_stack((X_hist, beta_new))
                R_hist = np.column_stack((R_hist, r))

                obj_new = self._objective(X, y, beta_new, lam)
                if obj_new < obj_best:
                    obj_best = obj_new
                    beta_best = beta_new.copy()

                if np.linalg.norm(r) < self.tol:
                    beta = beta_best
                    break

                if (it + 1) % self.M == 0 and X_hist.shape[1] >= 2:
                    x_acc = self._anderson(X_hist, R_hist)
                    x_damped = beta_new + self.damping * (x_acc - beta_new)
                    if np.all(np.isfinite(x_damped)) and np.linalg.norm(x_damped) < 1e6:
                        obj_damped = self._objective(X, y, x_damped, lam)
                        if obj_damped < obj_new:
                            beta = x_damped
                            X_hist = np.zeros((n_features, 0))
                            R_hist = np.zeros((n_features, 0))
                            continue
                beta = beta_new

            end = time.time()

            beta = beta_best
            beta[np.abs(beta) < 1e-8] = 0.0

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