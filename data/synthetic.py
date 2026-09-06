"""
合成数据生成器
可根据参数控制样本量、维度、稀疏度、组结构、特征相关性等
"""
import numpy as np
from sklearn.utils import check_random_state


def make_group_sparse_regression(
    n_samples=200,
    n_features=100,
    n_informative=20,
    n_groups=10,
    group_size=None,
    rho=0.5,       # 特征间相关系数
    noise=0.1,
    random_state=None,
):
    """
    生成带有分组稀疏结构的回归数据

    Parameters
    ----------
    n_samples : int
        样本数量
    n_features : int
        特征总数量
    n_informative : int
        真正起作用的特征个数（稀疏度控制）
    n_groups : int
        分组个数，如果 group_size 未指定，则每组大小均分
    group_size : int or None
        每组大小，如果指定，则 n_features = n_groups * group_size
    rho : float
        特征间的相关系数（0 表示独立，1 表示完全相关）
    noise : float
        噪声标准差
    random_state : int or None
        随机种子

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    beta_true : np.ndarray of shape (n_features,)
    groups : list of lists
        每个子列表包含属于该组的特征索引
    """
    rng = check_random_state(random_state)

    # 确定分组结构
    if group_size is not None:
        n_groups = n_features // group_size
    else:
        group_size = n_features // n_groups
    n_actual_features = n_groups * group_size
    groups = [list(range(i * group_size, (i + 1) * group_size))
              for i in range(n_groups)]

    # 生成真实系数 beta_true：先全部为零，再在部分组的全部特征上设非零值
    beta_true = np.zeros(n_actual_features)
    # 从所有组中随机选出若干组作为 informative 的组
    n_informative_groups = max(1, int(np.ceil(n_informative / group_size)))
    informative_group_idx = rng.choice(n_groups, size=n_informative_groups, replace=False)

    for g_idx in informative_group_idx:
        for f_idx in groups[g_idx]:
            beta_true[f_idx] = rng.normal(0, 1)

    # 生成具有相关性的特征矩阵 X
    # 首先为每个特征生成独立的标准正态向量
    X_indep = rng.randn(n_samples, n_actual_features)
    if rho > 0:
        # 为了简化，让每个特征与第一个特征有相关系数 rho
        common = rng.randn(n_samples, 1)
        X = np.sqrt(rho) * common + np.sqrt(1 - rho) * X_indep
    else:
        X = X_indep

    # 生成响应变量 y = X * beta + noise
    y = X @ beta_true + noise * rng.randn(n_samples)

    return X, y, beta_true, groups