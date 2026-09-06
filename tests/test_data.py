import numpy as np
import pytest
from data.synthetic import make_group_sparse_regression
from data.real_datasets import load_real_dataset


def test_synthetic_generator_shape():
    X, y, beta_true, groups = make_group_sparse_regression(
        n_samples=30, n_features=20, n_informative=5, n_groups=4, random_state=0
    )
    assert X.shape == (30, 20)
    assert y.shape == (30,)
    assert beta_true.shape == (20,)
    assert len(groups) == 4
    # 每个组大小相同
    group_sizes = [len(g) for g in groups]
    assert len(set(group_sizes)) == 1


def test_real_datasets_return_tuple():
    for name in ["california_housing", "diabetes", "newsgroups", "olivetti_faces"]:
        X, y, groups = load_real_dataset(name)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        if groups is not None:
            # groups 是列表的列表，每个组内索引在有效范围内
            for g in groups:
                assert isinstance(g, list)
                assert max(g) < X.shape[1]