"""
统一路径方法接口。所有方法都继承 BasePathMethod，并实现 fit 方法。
fit 方法必须返回一个字典，包含：
- coef_path: 系数矩阵，形状 (n_features, n_lambdas)
- objective: 每个 lambda 对应的目标函数值
- sparsity: 每个 lambda 下非零系数的个数
- timing: 每个 lambda 的计算时间
- lambdas: 实际使用的正则化参数序列
"""

from abc import ABC, abstractmethod

class BasePathMethod(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def fit(self, X, y, lambdas, groups=None):
        """
        生成正则化路径

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        lambdas : array-like, 正则化参数列表（从大到小）
        groups : list of lists or None, 分组信息（仅 Group Lasso 方法使用）

        Returns
        -------
        dict
        """
        pass