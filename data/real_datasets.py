"""
统一真实数据集加载器
每个数据集返回: X (np.ndarray), y (np.ndarray), groups (list of lists or None)
优先从 data/groups/<dataset_name>_groups.json 读取分组；
若不存在，则使用内置生成函数生成，并自动保存为 JSON。
"""

import os
import json
import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    load_diabetes,
    fetch_20newsgroups,
    fetch_olivetti_faces,
)
from sklearn.feature_extraction.text import TfidfVectorizer


def _ensure_groups_dir():
    """确保 data/groups/ 目录存在"""
    groups_dir = os.path.join(os.path.dirname(__file__), "groups")
    os.makedirs(groups_dir, exist_ok=True)
    return groups_dir


def _load_groups_from_json(dataset_name):
    """从 data/groups/<dataset_name>_groups.json 读取分组

    返回 list of lists 或 None（文件不存在时）
    """
    group_path = os.path.join(_ensure_groups_dir(), f"{dataset_name}_groups.json")
    if not os.path.exists(group_path):
        return None
    with open(group_path, "r") as f:
        groups_dict = json.load(f)
    # 确保返回的是列表的列表
    return list(groups_dict.values())


def _save_groups_to_json(dataset_name, groups):
    """将分组保存为 data/groups/<dataset_name>_groups.json"""
    group_path = os.path.join(_ensure_groups_dir(), f"{dataset_name}_groups.json")
    groups_dict = {f"group_{i+1}": g for i, g in enumerate(groups)}
    with open(group_path, "w") as f:
        json.dump(groups_dict, f, indent=2)


def _get_groups(dataset_name, generator_fn, **kwargs):
    """获取分组：优先读 JSON，否则动态生成并保存

    Parameters
    ----------
    dataset_name : str
    generator_fn : callable
        生成分组的函数，返回 list of lists
    **kwargs
        传递给 generator_fn 的参数

    Returns
    -------
    groups : list of lists
    """
    groups = _load_groups_from_json(dataset_name)
    if groups is None:
        groups = generator_fn(**kwargs)
        _save_groups_to_json(dataset_name, groups)
    return groups


def _load_california_housing():
    data = fetch_california_housing()
    return data.data, data.target, None


def _load_diabetes():
    data = load_diabetes()
    return data.data, data.target, None


def _generate_newsgroups_groups(n_features, n_groups=20):
    """生成 Newsgroups 的均分分组"""
    group_size = n_features // n_groups
    groups = [list(range(i * group_size, (i + 1) * group_size))
              for i in range(n_groups)]
    remainder = n_features % n_groups
    if remainder:
        for i in range(remainder):
            groups[i].append(n_groups * group_size + i)
    return groups


def _generate_olivetti_groups(img_size=64, block_size=8):
    """生成 Olivetti Faces 的空间块分组"""
    groups = []
    for i in range(0, img_size, block_size):
        for j in range(0, img_size, block_size):
            indices = []
            for r in range(i, i + block_size):
                for c in range(j, j + block_size):
                    indices.append(r * img_size + c)
            groups.append(indices)
    return groups


def _load_newsgroups():
    newsgroups = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        data_home=os.path.join(os.path.dirname(__file__), "..", "data_cache"),
    )
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(newsgroups.data).toarray()
    y = newsgroups.target

    n_features = X.shape[1]
    groups = _get_groups(
        "newsgroups",
        _generate_newsgroups_groups,
        n_features=n_features,
        n_groups=20,
    )
    return X, y, groups


def _load_olivetti_faces():
    data = fetch_olivetti_faces()
    X = data.data
    y = data.target

    groups = _get_groups("olivetti_faces", _generate_olivetti_groups)
    return X, y, groups


LOADERS = {
    "california_housing": _load_california_housing,
    "diabetes": _load_diabetes,
    "newsgroups": _load_newsgroups,
    "olivetti_faces": _load_olivetti_faces,
}


def load_real_dataset(name: str):
    """
    根据数据集名称加载真实数据集

    Returns
    -------
    X, y, groups
    """
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(LOADERS.keys())}")
    return LOADERS[name]()