import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    if not os.path.isdir(RESULTS_DIR):
        print("Results directory not found, skipping path difference plot.")
        return None

    datasets = df["dataset"].unique()
    rows = []
    for dataset in datasets:
        lars_file = os.path.join(RESULTS_DIR, f"{dataset}_lars.npz")
        if not os.path.exists(lars_file):
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
                if ref_path.shape != m_path.shape:
                    continue
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


def plot_accuracy_time_memory(df):
    """按数据集分面的散点图：accuracy vs time，点大小表示内存，颜色表示方法。"""
    df_lasso = df[(df["method_type"] == "lasso") & (df["method"] != "elastic_net")].copy()
    if df_lasso.empty:
        print("No Lasso methods found for accuracy plot.")
        return

    best_obj_per_dataset = df_lasso.groupby("dataset")["avg_objective"].min()
    df_lasso["best_obj"] = df_lasso["dataset"].map(best_obj_per_dataset)
    eps = 1e-12
    df_lasso["rel_err"] = (df_lasso["avg_objective"] - df_lasso["best_obj"]) / (df_lasso["best_obj"] + eps)
    df_lasso["accuracy"] = 1.0 / (1.0 + df_lasso["rel_err"])

    n_datasets = df_lasso["dataset"].nunique()
    col_wrap = min(3, n_datasets)

    g = sns.relplot(
        data=df_lasso,
        x="accuracy",
        y="total_time",
        size="peak_memory_mb",
        sizes=(40, 400),
        hue="method",
        col="dataset",
        col_wrap=col_wrap,
        kind="scatter",
        alpha=0.8,
        height=4,
        aspect=1.2,
        facet_kws={"sharex": False, "sharey": False},
    )
    g.set(yscale="log")
    g.set_axis_labels("Accuracy (1 / (1 + relative objective error))", "Total time (log scale)")
    g.set_titles("{col_name}")

    # 对线性 x 轴使用固定小数位显示，避免科学计数法
    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))

    # 调整图例位置，避免与子图重叠
    g.fig.suptitle("Accuracy vs. Solution Time (dot size ∝ memory)", y=1.02)
    g.fig.subplots_adjust(right=0.85)
    if g.legend is not None:
        g.legend.set_bbox_to_anchor((1.02, 0.5))
        g.legend.set_loc('center left')

    g.fig.tight_layout()
    save_fig(g.fig, "accuracy_time_memory.png")


def plot_pathdiff_time_memory(df):
    """按数据集分面的散点图：path difference vs time，点大小表示内存，颜色表示方法。"""
    path_df = compute_path_difference_from_npz(df)
    if path_df is None or path_df.empty:
        print("No path difference data available for pathdiff-time-memory plot.")
        return

    merged = pd.merge(
        path_df,
        df[["dataset", "method", "total_time", "peak_memory_mb"]],
        on=["dataset", "method"],
        how="inner"
    )
    if merged.empty:
        print("No merged data for pathdiff-time-memory plot.")
        return

    merged = merged[merged["method"] != "elastic_net"]

    n_datasets = merged["dataset"].nunique()
    col_wrap = min(3, n_datasets)

    g = sns.relplot(
        data=merged,
        x="path_diff",
        y="total_time",
        size="peak_memory_mb",
        sizes=(40, 400),
        hue="method",
        col="dataset",
        col_wrap=col_wrap,
        kind="scatter",
        alpha=0.8,
        height=4,
        aspect=1.2,
        facet_kws={"sharex": False, "sharey": False},
    )
    g.set(xscale="log", yscale="log")
    g.set_axis_labels("Path difference (log scale)", "Total time (log scale)")
    g.set_titles("{col_name}")

    # 对数 x 轴不需要 ticklabel_format，保留默认对数格式即可
    # 调整图例位置
    g.fig.suptitle("Path Difference vs. Solution Time (dot size ∝ memory)", y=1.02)
    g.fig.subplots_adjust(right=0.85)
    if g.legend is not None:
        g.legend.set_bbox_to_anchor((1.02, 0.5))
        g.legend.set_loc('center left')

    g.fig.tight_layout()
    save_fig(g.fig, "pathdiff_time_memory.png")


def plot_scaling_time(scaling_df):
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
    plot_accuracy_time_memory(df)
    plot_pathdiff_time_memory(df)

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