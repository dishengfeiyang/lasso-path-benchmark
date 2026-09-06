import numpy as np
import time
from celer import Lasso as CelerLasso
from ._base import BasePathMethod


class Celer(BasePathMethod):
    def __init__(self):
        super().__init__("Celer")

    def fit(self, X, y, lambdas, groups=None):
        n_samples, n_features = X.shape
        n_lambdas = len(lambdas)

        coef_path = np.zeros((n_features, n_lambdas))
        objective = np.zeros(n_lambdas)
        sparsity = np.zeros(n_lambdas, dtype=int)
        timing = np.zeros(n_lambdas)

        for j, lam in enumerate(lambdas):
            alpha = lam / n_samples

            start = time.time()
            model = CelerLasso(alpha=alpha, fit_intercept=False,
                               max_iter=50000, tol=1e-4)
            model.fit(X, y)
            end = time.time()

            beta = model.coef_
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