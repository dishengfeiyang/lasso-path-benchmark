import numpy as np
import pytest
from sklearn.datasets import make_regression

from methods.lars import LarsPath
from methods.coordinate_descent import CoordinateDescent
from methods.admm import ADMM
from methods.celer import Celer
from methods.fista import FISTA
from methods.screening_cd import ScreeningCoordinateDescent
from methods.anderson_cd import AndersonCD
from methods.elastic_net import ElasticNet
from methods.gap_safe_cd import GapSafeCD
from methods.block_cd import BlockCoordinateDescent
from methods.group_lars import GroupLARS
from benchmarks.utils import create_lambda_sequence


# 所有 Lasso 方法（不需要 groups）
LASSO_METHODS = [
    LarsPath(),
    CoordinateDescent(),
    ADMM(),
    Celer(),
    FISTA(),
    ScreeningCoordinateDescent(),
    AndersonCD(),
    ElasticNet(),
    GapSafeCD(),
]

# 所有 Group Lasso 方法（需要 groups）
GROUP_METHODS = [
    BlockCoordinateDescent(),
    GroupLARS(),
]


@pytest.fixture
def small_data():
    X, y, beta_true, groups = make_group_sparse_regression(
        n_samples=50, n_features=20, n_informative=5, n_groups=5,
        rho=0.1, noise=0.1, random_state=0
    )
    X_c = X - X.mean(axis=0)
    y_c = y - y.mean()
    lambdas = create_lambda_sequence(X_c, y_c, n_lambdas=10, ratio=0.1, groups=None)
    return X_c, y_c, lambdas, groups


@pytest.fixture
def group_data():
    X, y, beta_true, groups = make_group_sparse_regression(
        n_samples=50, n_features=20, n_informative=5, n_groups=5,
        rho=0.1, noise=0.1, random_state=0
    )
    X_c = X - X.mean(axis=0)
    y_c = y - y.mean()
    lambdas = create_lambda_sequence(X_c, y_c, n_lambdas=10, ratio=0.1, groups=groups)
    return X_c, y_c, lambdas, groups


def test_lasso_methods_return_required_keys(small_data):
    X, y, lambdas, _ = small_data
    for method in LASSO_METHODS:
        result = method.fit(X, y, lambdas)
        for key in ["coef_path", "objective", "sparsity", "timing", "lambdas"]:
            assert key in result, f"{method.name} missing key {key}"


def test_group_methods_return_required_keys(group_data):
    X, y, lambdas, groups = group_data
    for method in GROUP_METHODS:
        result = method.fit(X, y, lambdas, groups=groups)
        for key in ["coef_path", "objective", "sparsity", "timing", "lambdas"]:
            assert key in result, f"{method.name} missing key {key}"


def test_output_shapes(small_data, group_data):
    X, y, lambdas, _ = small_data
    n_features = X.shape[1]
    n_lambdas = len(lambdas)
    for method in LASSO_METHODS:
        result = method.fit(X, y, lambdas)
        assert result["coef_path"].shape == (n_features, n_lambdas)
        assert len(result["objective"]) == n_lambdas
        assert len(result["sparsity"]) == n_lambdas
        assert len(result["timing"]) == n_lambdas

    Xg, yg, lambdas_g, groups = group_data
    n_features_g = Xg.shape[1]
    n_lambdas_g = len(lambdas_g)
    for method in GROUP_METHODS:
        result = method.fit(Xg, yg, lambdas_g, groups=groups)
        assert result["coef_path"].shape == (n_features_g, n_lambdas_g)
        assert len(result["objective"]) == n_lambdas_g


def test_lasso_solution_accuracy(small_data):
    X, y, lambdas, _ = small_data
    # 使用坐标下降作为参考
    ref = CoordinateDescent().fit(X, y, lambdas)
    for method in [LarsPath(), ADMM(), FISTA(), ScreeningCoordinateDescent(), AndersonCD()]:
        result = method.fit(X, y, lambdas)
        # 目标值平均相对误差应小于 5%
        avg_rel_diff = np.mean(np.abs(result["objective"] - ref["objective"]) /
                               (np.abs(ref["objective"]) + 1e-12))
        assert avg_rel_diff < 0.05, f"{method.name} objective deviates too much"