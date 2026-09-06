import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(BASE_DIR, "outputs", "tables", "results.csv")
SCALING_CSV = os.path.join(BASE_DIR, "outputs", "tables", "scaling_results.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def save_fig(fig, filename):
    """保存图片到 figures 目录"""
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_time_comparison(df):
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=df, x="dataset", y="total_time", hue="method", ax=ax)
    ax.set_yscale("log")
    ax.set_title("Total time by dataset and method (log scale)")
    ax.set_ylabel("Time (seconds, log scale)")
    ax.set_xlabel("Dataset")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_fig(fig, "time_comparison_log.png")


def plot_objective_lasso(df):
    df_lasso = df[df["method_type"] == "lasso"].copy()
    if df_lasso.empty:
        return
    # 使用分面，每个数据集一个子图，y轴对数刻度
    g = sns.catplot(
        data=df_lasso, x="method", y="avg_objective",
        col="dataset", kind="bar", col_wrap=2, sharey=False,
        height=4, aspect=1.2, palette="viridis"
    )
    g.set(yscale="log")
    g.set_axis_labels("Method", "Average objective (log scale)")
    g.set_titles("{col_name}")
    g.fig.suptitle("Average objective for Lasso methods (log scale)", y=1.02)
    g.fig.tight_layout()
    save_fig(g.fig, "objective_lasso_facet.png")


def plot_objective_group(df):
    df_group = df[df["method_type"] == "group_lasso"].copy()
    if df_group.empty:
        return
    g = sns.catplot(
        data=df_group, x="method", y="avg_objective",
        col="dataset", kind="bar", col_wrap=2, sharey=False,
        height=4, aspect=1.2, palette="viridis"
    )
    g.set(yscale="log")
    g.set_axis_labels("Method", "Average objective (log scale)")
    g.set_titles("{col_name}")
    g.fig.suptitle("Average objective for Group Lasso methods (log scale)", y=1.02)
    g.fig.tight_layout()
    save_fig(g.fig, "objective_group_lasso_facet.png")


def plot_objective_elastic(df):
    df_en = df[df["method"] == "elastic_net"].copy()
    if df_en.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_en, x="dataset", y="avg_objective", hue="method", ax=ax)
    ax.set_yscale("log")
    ax.set_title("Average objective for Elastic Net (log scale)")
    ax.set_ylabel("Average objective (log scale)")
    ax.set_xlabel("Dataset")
    ax.tick_params(axis="x", rotation=45)
    save_fig(fig, "objective_elastic_net_log.png")


def plot_memory(df):
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=df, x="dataset", y="peak_memory_mb", hue="method", ax=ax)
    ax.set_yscale("log")
    ax.set_title("Peak memory by dataset and method (log scale)")
    ax.set_ylabel("Memory (MB, log scale)")
    ax.set_xlabel("Dataset")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_fig(fig, "memory_comparison_log.png")


def plot_sparsity(df):
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=df, x="dataset", y="final_sparsity", hue="method", ax=ax)
    ax.set_title("Final sparsity by dataset and method")
    ax.set_ylabel("Number of non-zero coefficients")
    ax.set_xlabel("Dataset")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_fig(fig, "sparsity_comparison.png")


def plot_screened_ratio(df):
    # 只展示有筛选比例的方法
    df_screen = df.dropna(subset=["screened_ratio"]).copy()
    if df_screen.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_screen, x="dataset", y="screened_ratio", hue="method", ax=ax)
    ax.set_title("Average screened ratio (only screening methods)")
    ax.set_ylabel("Screened ratio")
    ax.set_xlabel("Dataset")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_fig(fig, "screened_ratio_comparison.png")


def compute_path_difference_from_npz(df):
    """从 .npz 文件计算每个方法相对于 LARS 的路径平均绝对差"""
    if not os.path.isdir(RESULTS_DIR):
        print("Results directory not found, skipping path difference plot.")
        return None

    datasets = df["dataset"].unique()
    rows = []
    for dataset in datasets:
        # 寻找该数据集的 LARS 结果文件
        lars_file = os.path.join(RESULTS_DIR, f"{dataset}_lars.npz")
        if not os.path.exists(lars_file):
            # 如果没有 LARS，尝试用 coordinate_descent 作为参考
            lars_file = os.path.join(RESULTS_DIR, f"{dataset}_coordinate_descent.npz")
            if not os.path.exists(lars_file):
                continue
        ref_data = np.load(lars_file)
        ref_path = ref_data["coef_path"]

        for method in df[df["dataset"] == dataset]["method"].unique():
            method_file = os.path.join(RESULTS_DIR, f"{dataset}_{method}.npz")
            if not os.path.exists(method_file):
                continue
            try:
                m_data = np.load(method_file)
                m_path = m_data["coef_path"]
                # 确保形状一致
                if ref_path.shape != m_path.shape:
                    continue
                # 平均绝对差
                diff = np.mean(np.abs(m_path - ref_path))
                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "path_diff": diff
                })
            except Exception as e:
                print(f"Error loading {method_file}: {e}")

    if not rows:
        return None
    return pd.DataFrame(rows)


def plot_path_difference(df):
    path_df = compute_path_difference_from_npz(df)
    if path_df is None or path_df.empty:
        print("No path difference data available.")
        return
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=path_df, x="dataset", y="path_diff", hue="method", ax=ax)
    ax.set_yscale("log")
    ax.set_title("Path difference relative to reference method (log scale)")
    ax.set_ylabel("Mean absolute coefficient difference (log scale)")
    ax.set_xlabel("Dataset")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_fig(fig, "path_difference_log.png")


def plot_scaling_time(scaling_df):
    # 分别绘制三个实验的时间变化
    experiments = ["vary_n", "vary_p", "vary_n_lambdas"]
    x_vars = {"vary_n": "n_samples", "vary_p": "n_features", "vary_n_lambdas": "n_lambdas"}
    for exp in experiments:
        sub = scaling_df[scaling_df["experiment"] == exp]
        if sub.empty:
            continue
        x = x_vars[exp]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.lineplot(data=sub, x=x, y="time", hue="method", marker="o", ax=ax)
        ax.set_yscale("log")
        ax.set_title(f"Time scaling: {exp}")
        ax.set_ylabel("Time (seconds, log scale)")
        ax.set_xlabel(x)
        ax.legend(title="Method")
        save_fig(fig, f"scaling_time_{exp}.png")


def plot_scaling_memory(scaling_df):
    experiments = ["vary_n", "vary_p", "vary_n_lambdas"]
    x_vars = {"vary_n": "n_samples", "vary_p": "n_features", "vary_n_lambdas": "n_lambdas"}
    for exp in experiments:
        sub = scaling_df[scaling_df["experiment"] == exp]
        if sub.empty:
            continue
        x = x_vars[exp]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.lineplot(data=sub, x=x, y="memory_mb", hue="method", marker="o", ax=ax)
        ax.set_yscale("log")
        ax.set_title(f"Memory scaling: {exp}")
        ax.set_ylabel("Memory (MB, log scale)")
        ax.set_xlabel(x)
        ax.legend(title="Method")
        save_fig(fig, f"scaling_memory_{exp}.png")


def main():
    if not os.path.exists(RESULTS_CSV):
        print(f"Results file not found: {RESULTS_CSV}")
        return
    df = pd.read_csv(RESULTS_CSV)

    # 主基准测试图
    plot_time_comparison(df)
    plot_objective_lasso(df)
    plot_objective_group(df)
    plot_objective_elastic(df)
    plot_memory(df)
    plot_sparsity(df)
    plot_screened_ratio(df)
    plot_path_difference(df)

    # 扩展性实验图
    if os.path.exists(SCALING_CSV):
        scaling_df = pd.read_csv(SCALING_CSV)
        plot_scaling_time(scaling_df)
        plot_scaling_memory(scaling_df)
    else:
        print(f"Scaling results file not found: {SCALING_CSV}")

    print("All figures generated.")


if __name__ == "__main__":
    main()