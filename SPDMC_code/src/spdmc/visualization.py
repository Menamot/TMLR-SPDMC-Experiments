import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

DEFAULT_MODEL_ORDER = [
    "DBC-CMeans",
    "DBC-CMeans-ClassWise",
    "DBC-DiagGMM",
    "DBC-DiagGMM-ClassWise",
    "DBC-Tree-DiagGMM",
    "DBC-Tree-FullGMM",
    "DMC-KMeans",
    "DMC-KMeans-ClassWise",
    "DMC-CMeans",
    "DMC-CMeans-ClassWise",
    "DMC-DiagGMM",
    "DMC-DiagGMM-ClassWise",
    "DMC-Tree-DiagGMM",
    "DMC-Tree-FullGMM",
    "Weighted Logistic Regression",
    "Weighted Decision Tree",
]


def _get_model_order(df, model_order=None):
    present = set(df["model_name"].unique())
    if model_order is not None:
        return [m for m in model_order if m in present]
    return [m for m in DEFAULT_MODEL_ORDER if m in present]


def _get_colors(n):
    cmap = plt.colormaps.get_cmap("tab20")
    return [cmap(i % cmap.N) for i in range(n)]


def _style_boxplot(bp, colors, face_alpha=0.6, edge_color="black", line_width=1.2):
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(face_alpha)
        patch.set_edgecolor(edge_color)
        patch.set_linewidth(line_width)

    for median in bp.get("medians", []):
        median.set_color(edge_color)
        median.set_linewidth(line_width)

    for mean in bp.get("means", []):
        mean.set_color(edge_color)
        mean.set_linewidth(line_width)

    for whisker in bp.get("whiskers", []):
        whisker.set_color(edge_color)
        whisker.set_linewidth(line_width)

    for cap in bp.get("caps", []):
        cap.set_color(edge_color)
        cap.set_linewidth(line_width)


def _aggregate_test(df_test, agg):
    if agg == "prior_mean":
        metric_cols = [
            "global_risk",
            "max_class_risk",
            "class_risk_range",
            "class_risk_variance",
        ]
        group_cols = ["model_name", "repeat_idx", "fold_idx"]
        present_metrics = [c for c in metric_cols if c in df_test.columns]
        return df_test.groupby(group_cols, as_index=False)[present_metrics].mean()
    return df_test


def _collect_metric_values(df, model_name, metric, data_split, test_agg):
    sub = df[(df["model_name"] == model_name) & (df["data_split"] == data_split)]
    if data_split == "train" or test_agg == "all_priors":
        return sub[metric].dropna().to_numpy(dtype=float)
    if test_agg == "prior_mean":
        grouped = sub.groupby(["repeat_idx", "fold_idx"], as_index=False)[metric].mean()
        return grouped[metric].dropna().to_numpy(dtype=float)
    raise ValueError("Invalid test_agg. Use 'all_priors' or 'prior_mean'.")


def _compute_metric_ylim(df, models, metric, test_agg):
    values = []
    for split_key in ["train", "test"]:
        for model_name in models:
            metric_values = _collect_metric_values(
                df, model_name, metric, split_key, test_agg
            )
            if len(metric_values) > 0:
                values.append(metric_values)

    if not values:
        return None

    all_vals = np.concatenate(values)
    ymin = float(np.min(all_vals))
    ymax = float(np.max(all_vals))
    margin = max((ymax - ymin) * 0.05, 0.01)
    return ymin - margin, ymax + margin


def plot_metric_boxplot(
    df,
    metrics=None,
    metric_labels=None,
    model_order=None,
    test_agg="prior_mean",
    figsize=(24, 10),
    title=None,
    save_path=None,
    short_labels=None,
    legend_ncol=4,
    fontsize=None,
    **boxplot_kwargs,
):
    if metrics is None:
        metrics = [
            "global_risk",
            "max_class_risk",
            "class_risk_range",
            # "class_risk_variance",
        ]
    if metric_labels is None:
        metric_labels = [
            "Global Risk",
            "Max Class Risk",
            "Class Risk Range",
            # "Class Risk Variance",
        ]
    if len(metric_labels) != len(metrics):
        raise ValueError("metric_labels must have the same length as metrics")

    models = _get_model_order(df, model_order)
    if not models:
        raise ValueError("No models found to plot")

    n_models = len(models)
    if short_labels is None:
        short_labels = [f"M{i+1}" for i in range(n_models)]
    if len(short_labels) != n_models:
        raise ValueError("short_labels must have the same length as plotted models")

    fs = fontsize or 12
    fs_title = fs + 2
    fs_suptitle = fs + 4
    fs_legend = fs - 2
    fs_tick = fs - 2

    colors = _get_colors(len(models))
    metric_ylims = {
        metric: _compute_metric_ylim(df, models, metric, test_agg) for metric in metrics
    }
    fig, axes = plt.subplots(
        2,
        len(metrics),
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey="col",
    )

    row_info = [("train", "Train"), ("test", "Test")]
    for row_idx, (split_key, split_label) in enumerate(row_info):
        for col_idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row_idx, col_idx]
            data = [
                _collect_metric_values(df, m, metric, split_key, test_agg)
                for m in models
            ]
            bp_kwargs = dict(patch_artist=True, showmeans=True, meanline=False)
            bp_kwargs.update(boxplot_kwargs)
            bp = ax.boxplot(data, **bp_kwargs)
            _style_boxplot(bp, colors)

            ylim = metric_ylims.get(metric)
            if ylim is not None:
                ax.set_ylim(ylim)

            ax.set_xticks(range(1, n_models + 1))
            ax.set_xticklabels(short_labels, rotation=0, fontsize=fs_tick)
            ax.set_title(f"{split_label} | {metric_label}", fontsize=fs_title)
            ax.set_ylabel(metric_label, fontsize=fs)
            ax.tick_params(axis="y", labelsize=fs_tick)
            ax.grid(axis="y", alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=fs_suptitle)

    legend_elements = [
        Patch(facecolor=colors[i], alpha=0.6, label=f"{short_labels[i]}: {models[i]}")
        for i in range(n_models)
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=min(n_models, legend_ncol),
        frameon=True,
        fontsize=fs_legend,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92 if title else 0.9])
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_class_risk(
    df,
    df_class_risks,
    model_order=None,
    test_agg="prior_mean",
    ylim=(0, 1),
    figsize=None,
    save_path=None,
    fontsize=None,
    **boxplot_kwargs,
):
    fs = fontsize or 10
    fs_title = fs
    fs_tick = fs - 4

    models = _get_model_order(df, model_order)
    n_models = len(models)
    short_labels = [f"M{i+1}" for i in range(n_models)]
    class_labels = sorted(df_class_risks["class_label"].unique())
    n_classes = len(class_labels)
    if figsize is None:
        figsize = (3 * n_models, 7)
    class_colors = _get_colors(n_classes)

    fig, axes = plt.subplots(2, n_models, figsize=figsize, squeeze=False)
    for row_idx, split in enumerate(["train", "test"]):
        df_cr_split = df_class_risks[df_class_risks["data_split"] == split]
        if split == "test" and test_agg == "prior_mean":
            df_cr_split = df_cr_split.groupby(
                ["model_name", "repeat_idx", "fold_idx", "class_label"],
                as_index=False,
            )["class_risk"].mean()
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            df_model = df_cr_split[df_cr_split["model_name"] == model]
            data = [
                df_model[df_model["class_label"] == c]["class_risk"].dropna().values
                for c in class_labels
            ]
            bp_kwargs = dict(patch_artist=True, showmeans=True, meanline=False)
            bp_kwargs.update(boxplot_kwargs)
            bp = ax.boxplot(data, **bp_kwargs)
            _style_boxplot(bp, class_colors)
            ax.set_xticks(range(1, n_classes + 1))
            ax.set_xticklabels([str(c) for c in class_labels], fontsize=fs_tick)
            ax.tick_params(axis="y", labelsize=fs_tick)
            ax.set_ylim(ylim)
            # ax.set_title(f"{split.title()} — {model}", fontsize=fs_title)
            ax.set_title(
                f"{split.title()} — {short_labels[col_idx]}", fontsize=fs_title
            )
            ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")


def plot_noise_sweep(
    df,
    metric="max_class_risk",
    model_order=None,
    figsize=(8, 5),
    capsize=4,
    save_path=None,
    fontsize=None,
    **errorbar_kwargs,
):
    fs = fontsize or 12
    fs_title = fs + 2
    fs_legend = fs - 2
    fs_tick = fs - 2

    df_test = df[df["data_split"] == "test"]
    models = _get_model_order(df_test, model_order)
    colors = _get_colors(len(models))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]

    fig, ax = plt.subplots(figsize=figsize)
    for i, model in enumerate(models):
        df_m = df_test[df_test["model_name"] == model]
        grouped = df_m.groupby("feature_noise_std")[metric]
        means = grouped.mean()
        stds = grouped.std()
        eb_kwargs = dict(
            marker=markers[i % len(markers)],
            capsize=capsize,
            label=model,
            color=colors[i],
        )
        eb_kwargs.update(errorbar_kwargs)
        ax.errorbar(means.index, means.values, yerr=stds.values, **eb_kwargs)
    ax.set_xlabel("Feature Noise Std", fontsize=fs)
    ax.set_ylabel(metric, fontsize=fs)
    ax.set_title(f"{metric} vs Feature Noise", fontsize=fs_title)
    ax.tick_params(labelsize=fs_tick)
    ax.legend(fontsize=fs_legend, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")


def plot_noise_class_risks(
    noise_data,
    model_order=None,
    test_agg="prior_mean",
    data_split="test",
    ylim=(0, 1),
    figsize=None,
    save_path=None,
    fontsize=None,
    **boxplot_kwargs,
):
    """Plot per-class risk boxplots for each noise level.

    Args:
        noise_data: list of (noise_std, df_results, df_class_risks) tuples.
        model_order: Optional list of model names to include/order.
        test_agg: How to aggregate test priors ('prior_mean' or 'all_priors').
        data_split: 'train', 'test', or 'both'.
        ylim: Y-axis limits for all subplots.
        figsize: Optional figure size.
        save_path: Optional path to save the figure.
        fontsize: Base font size.
    """
    fs = fontsize or 10
    fs_title = fs
    fs_tick = fs - 2

    noise_data = sorted(noise_data, key=lambda x: x[0])

    # Determine models from the first entry
    all_df = pd.concat([d[1] for d in noise_data], ignore_index=True)
    models = _get_model_order(all_df, model_order)
    n_models = len(models)

    # Determine class labels from the first class risk df
    all_cr = pd.concat([d[2] for d in noise_data], ignore_index=True)
    class_labels = sorted(all_cr["class_label"].unique())
    n_classes = len(class_labels)
    class_colors = _get_colors(n_classes)

    splits = ["train", "test"] if data_split == "both" else [data_split]
    n_noise = len(noise_data)
    n_rows = len(splits) * n_noise

    if figsize is None:
        figsize = (3 * n_models, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_models, figsize=figsize, squeeze=False)

    for noise_idx, (noise_std, df_results, df_cr) in enumerate(noise_data):
        for split_idx, split in enumerate(splits):
            row = noise_idx * len(splits) + split_idx
            df_cr_split = df_cr[df_cr["data_split"] == split]
            if split == "test" and test_agg == "prior_mean":
                df_cr_split = df_cr_split.groupby(
                    ["model_name", "repeat_idx", "fold_idx", "class_label"],
                    as_index=False,
                )["class_risk"].mean()

            for col, model in enumerate(models):
                ax = axes[row, col]
                df_model = df_cr_split[df_cr_split["model_name"] == model]
                data = [
                    df_model[df_model["class_label"] == c]["class_risk"].dropna().values
                    for c in class_labels
                ]
                bp_kwargs = dict(patch_artist=True, showmeans=True, meanline=False)
                bp_kwargs.update(boxplot_kwargs)
                bp = ax.boxplot(data, **bp_kwargs)
                _style_boxplot(bp, class_colors)
                ax.set_xticks(range(1, n_classes + 1))
                ax.set_xticklabels([str(c) for c in class_labels], fontsize=fs_tick)
                ax.tick_params(axis="y", labelsize=fs_tick)
                ax.set_ylim(ylim)
                ax.set_title(f"σ={noise_std:.2f} {split} — {model}", fontsize=fs_title)
                ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
