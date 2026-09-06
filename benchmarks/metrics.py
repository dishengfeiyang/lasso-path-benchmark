import numpy as np


def coefficient_error(beta_est, beta_true):
    """相对系数误差"""
    return np.linalg.norm(beta_est - beta_true) / (np.linalg.norm(beta_true) + 1e-12)


def prediction_error(X, y, beta):
    """预测均方误差"""
    y_pred = X @ beta
    return np.mean((y - y_pred) ** 2)


def path_difference(coef_path1, coef_path2):
    """两条路径之间的平均绝对差"""
    return np.mean(np.abs(coef_path1 - coef_path2))