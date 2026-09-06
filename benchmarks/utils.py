import numpy as np


def create_lambda_sequence(X, y, n_lambdas=50, ratio=1e-4, groups=None):
    """根据数据自适应生成对数均匀的 lambda 序列（从大到小）。

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        中心化后的特征矩阵。
    y : np.ndarray, shape (n_samples,)
        中心化后的目标向量。
    n_lambdas : int
        生成的 lambda 个数。
    ratio : float
        最小 lambda 与最大 lambda 的比值（默认 1e-4）。
    groups : list of lists or None
        分组信息；如果提供，则按 Group Lasso 的方式计算 lambda_max。

    Returns
    -------
    lambdas : np.ndarray
        对数均匀的 lambda 序列，从大到小。
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_samples = X.shape[0]
    ratio = float(ratio)

    if groups is None:
        # 标准 Lasso: lambda_max = max(|X^T y|) / n
        lambda_max = np.max(np.abs(X.T @ y)) / n_samples
    else:
        # Group Lasso: lambda_max = max_g ||X_g^T y||_2 / n
        max_norm = 0.0
        for g in groups:
            X_g = X[:, g]
            norm = np.linalg.norm(X_g.T @ y)
            if norm > max_norm:
                max_norm = norm
        lambda_max = max_norm / n_samples

    # 将 lambda_max 安全转换为 Python float 标量
    if hasattr(lambda_max, 'item'):
        lambda_max = lambda_max.item()
    else:
        lambda_max = float(lambda_max)

    # 防止 lambda_max 为 0 或负数导致 log 出错
    if lambda_max <= 0:
        lambda_max = 1e-12

    lambda_min = lambda_max * ratio
    return np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), n_lambdas))