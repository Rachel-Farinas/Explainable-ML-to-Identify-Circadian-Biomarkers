# plotting.py
# All visualisations: correlation heatmaps, network graphs, spider plots,
# feature-importance bars, and confusion-matrix grids.

import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from .config import PLOTS_DIR


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(fname: str) -> None:
    """Save current figure to PLOTS_DIR/<fname>."""
    path = os.path.join(PLOTS_DIR, fname)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[plot] Saved → {path}")


# ── 1. Per-group correlation heatmaps ────────────────────────────────────────

def plot_correlation_heatmaps(
    metadata: pd.DataFrame,
    features: list[str],
    save_fname: str = "vertical_monochrome_signature_grid.png",
) -> None:
    """
    Stacked vertical grid of per-group correlation heatmaps (monochrome).
    """
    groups = metadata["group"].unique()
    fig, axes = plt.subplots(len(groups), 1, figsize=(8, 7 * len(groups)))

    if len(groups) == 1:
        axes = [axes]

    for i, group in enumerate(groups):
        group_corr = (
            metadata[metadata["group"] == group][features].corr()
        )
        sns.heatmap(
            group_corr,
            annot=True,
            cmap="Greys",
            vmin=-1,
            vmax=1,
            ax=axes[i],
            cbar=True,
            fmt=".2f",
            square=True,
            annot_kws={"size": 11},
            linecolor="white",
            linewidths=1.5,
        )
        axes[i].set_title(
            f"Behavioral Signature: {group.upper()}",
            fontsize=16,
            weight="bold",
            pad=25,
        )

    plt.tight_layout(pad=4.0)
    _save(save_fname)
    plt.show()


# ── 2. Feature-redundancy check heatmap ──────────────────────────────────────

def plot_redundancy_heatmap(
    metadata: pd.DataFrame,
    features: list[str],
    save_fname: str = "feature_redundancy_heatmap.png",
) -> None:
    """
    Single correlation heatmap across all `features` (used to check whether
    new circadian features are redundant with existing ones).
    """
    corr_matrix = metadata[features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="Greys",
        fmt=".2f",
        square=True,
        linewidths=0.5,
    )
    plt.title("Feature redundancy check: Global vs Circadian Dynamics", fontsize=14, pad=20)
    _save(save_fname)
    plt.show()


# ── 3. Behavioral network graph ───────────────────────────────────────────────

def plot_behavioral_network_vertical_dynamic(
    df: pd.DataFrame,
    features: list[str],
    threshold: float = 0.6,
    save_fname: str = "dynamic_behavioral_networks.png",
) -> None:
    """
    Stacked vertical grid of correlation-network graphs; edge opacity scales
    with correlation strength.
    """
    groups = df["group"].unique()
    fig, axes = plt.subplots(len(groups), 1, figsize=(8, 8 * len(groups)))

    if len(groups) == 1:
        axes = [axes]

    for i, group in enumerate(groups):
        ax         = axes[i]
        group_data = df[df["group"] == group][features]
        corr       = group_data.corr()

        G = nx.Graph()
        G.add_nodes_from(features)

        for row in range(len(corr.columns)):
            for col in range(row):
                val = corr.iloc[row, col]
                if abs(val) > threshold:
                    G.add_edge(
                        corr.columns[row], corr.columns[col], weight=abs(val)
                    )

        pos = nx.circular_layout(G)

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color="#f0f0f0",
            node_size=3000,
            edgecolors="black",
            linewidths=1.5,
        )
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_size=11,
            font_family="sans-serif",
            font_weight="bold",
        )

        if G.edges():
            weights = [G[u][v]["weight"] * 7 for u, v in G.edges()]
            edge_colors = []
            for u, v in G.edges():
                w     = G[u][v]["weight"]
                alpha = 0.2 + (w - threshold) * (0.8 / (1.0 - threshold))
                edge_colors.append((0, 0, 0, alpha))

            nx.draw_networkx_edges(
                G, pos, ax=ax,
                width=weights,
                edge_color=edge_colors,
            )

        ax.margins(0.2)
        ax.set_aspect("equal")
        ax.set_title(
            f"Network Topology: {group.upper()}",
            fontsize=18,
            pad=45,
            weight="bold",
        )
        ax.axis("off")

    plt.tight_layout(pad=7.0)
    _save(save_fname)
    plt.show()


# ── 4. Spider (radar) plot ────────────────────────────────────────────────────

def plot_spider(
    metadata: pd.DataFrame,
    metric_cols: list[str],
    labels: list[str],
    save_fname: str = "spider_behavioral_groups.png",
) -> None:
    """
    Radar/spider chart comparing normalised group-mean metrics.
    """
    num_vars = len(labels)

    df_norm = metadata.groupby("group")[metric_cols].mean()
    df_norm = (df_norm - df_norm.min()) / (df_norm.max() - df_norm.min())

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors      = ["#0072B2", "#D55E00", "#009E73", "#F0E442"]
    line_styles = ["-", "--", ":", "-."]
    markers     = ["o", "s", "D", "^"]

    for i, group in enumerate(df_norm.index):
        values  = df_norm.loc[group].tolist()
        values += values[:1]

        ax.plot(
            angles, values,
            color=colors[i % len(colors)],
            linestyle=line_styles[i % len(line_styles)],
            marker=markers[i % len(markers)],
            linewidth=3,
            label=group,
            markersize=8,
        )
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1.1)

    plt.title("Behavioral group metrics comparison", y=1.1, fontsize=15, fontweight="bold")
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True)

    _save(save_fname)
    plt.show()


# ── 5. Confusion-matrix grid ──────────────────────────────────────────────────

def plot_confusion_matrix_grid(
    y_true,
    y_preds: dict,          # {feature_set_name: y_pred array}
    class_labels,
    accuracies: dict,       # {feature_set_name: accuracy float}
    save_fname: str = "confusion_matrices.png",
) -> None:
    """
    2×2 grid of confusion matrices, one per feature set.
    """
    n       = len(y_preds)
    ncols   = 2
    nrows   = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 14))
    axes    = axes.flatten()

    for i, (name, y_pred) in enumerate(y_preds.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            display_labels=class_labels,
            cmap="Blues",
            xticks_rotation=45,
            ax=axes[i],
            colorbar=False,
        )
        acc = accuracies.get(name, 0.0)
        axes[i].set_title(f"{name}\nAcc: {acc:.2%}", fontsize=12, fontweight="bold")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _save(save_fname)
    plt.show()


# ── 6. Feature-importance bar chart ──────────────────────────────────────────

def plot_feature_importance(
    feature_names,
    importances,
    save_fname: str = "feature_importance.png",
) -> None:
    """
    Horizontal bar chart of XGBoost feature importances sorted descending.
    """
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="magma")
    plt.title("Final Feature Importance", fontsize=14)
    plt.xlabel("Relative Importance (Gain)")
    _save(save_fname)
    plt.show()
