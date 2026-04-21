
# embedding_analysis.py
# Interprets PAT embeddings via three complementary approaches:
#   1. Dimensionality reduction + visualization (t-SNE and UMAP)
#   2. Classification (logistic regression, SVM, XGBoost) vs. handcrafted features
#   3. Cosine similarity analysis (group cohesion + cross-diagnostic overlap)

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.svm import SVC
import xgboost as xgb

from .config import EMBEDDINGS_DIR, PLOTS_DIR, RANDOM_STATE


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(fname: str) -> None:
    path = os.path.join(PLOTS_DIR, fname)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[plot] Saved → {path}")


def load_embeddings(embeddings_dir: str = EMBEDDINGS_DIR) -> tuple:
    """
    Loads the four .npy files saved by save_embeddings().

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    """
    X_train = np.load(os.path.join(embeddings_dir, "X_train_embeddings.npy"))
    X_test  = np.load(os.path.join(embeddings_dir, "X_test_embeddings.npy"))
    y_train = np.load(os.path.join(embeddings_dir, "y_train.npy"))
    y_test  = np.load(os.path.join(embeddings_dir, "y_test.npy"))
    return X_train, X_test, y_train, y_test


# ── 1. Dimensionality reduction + visualization ───────────────────────────────

def plot_tsne(
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    save_fname: str = "embeddings_tsne.png",
    perplexity: int = 30,
) -> None:
    """
    Projects embeddings to 2D with t-SNE and plots a scatter coloured by
    diagnostic group. Tight clusters indicate the transformer learned
    group-separating features.
    """
    print("[embedding_analysis] Running t-SNE (this may take a moment)...")
    tsne    = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_STATE)
    X_2d    = tsne.fit_transform(X)

    df_plot = pd.DataFrame({
        "x":     X_2d[:, 0],
        "y":     X_2d[:, 1],
        "group": label_encoder.inverse_transform(y),
    })

    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    palette = {g: c for g, c in zip(sorted(df_plot["group"].unique()), colors)}

    plt.figure(figsize=(9, 7))
    for group, grp_df in df_plot.groupby("group"):
        plt.scatter(
            grp_df["x"], grp_df["y"],
            label=group,
            color=palette[group],
            alpha=0.75,
            s=55,
            edgecolors="white",
            linewidths=0.4,
        )

    plt.title("PAT Embeddings — t-SNE Projection", fontsize=15, fontweight="bold")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Group", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    _save(save_fname)
    plt.show()


def plot_umap(
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    save_fname: str = "embeddings_umap.png",
) -> None:
    """
    Projects embeddings to 2D with UMAP and plots a scatter coloured by
    diagnostic group. UMAP tends to preserve global structure better than
    t-SNE, making it useful for spotting cross-diagnostic overlap.
    """
    try:
        import umap
    except ImportError:
        print("[embedding_analysis] UMAP not installed — skipping. "
              "Run: pip install umap-learn")
        return

    print("[embedding_analysis] Running UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
    X_2d    = reducer.fit_transform(X)

    df_plot = pd.DataFrame({
        "x":     X_2d[:, 0],
        "y":     X_2d[:, 1],
        "group": label_encoder.inverse_transform(y),
    })

    colors  = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    palette = {g: c for g, c in zip(sorted(df_plot["group"].unique()), colors)}

    plt.figure(figsize=(9, 7))
    for group, grp_df in df_plot.groupby("group"):
        plt.scatter(
            grp_df["x"], grp_df["y"],
            label=group,
            color=palette[group],
            alpha=0.75,
            s=55,
            edgecolors="white",
            linewidths=0.4,
        )

    plt.title("PAT Embeddings — UMAP Projection", fontsize=15, fontweight="bold")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title="Group", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    _save(save_fname)
    plt.show()


# ── 2. Classification on embeddings ──────────────────────────────────────────

def run_embedding_classification(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    label_encoder: LabelEncoder,
) -> pd.DataFrame:
    """
    Trains three classifiers on PAT embeddings and evaluates on the held-out
    test set. Results are printed and returned as a summary DataFrame.

    Classifiers
    -----------
    - Logistic Regression (linear baseline)
    - SVM with RBF kernel   (non-linear, good for dense embeddings)
    - XGBoost               (same family as the handcrafted-feature model)
    """
    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "XGBoost": xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            tree_method="hist",
            device="cpu",
            eval_metric="mlogloss",
        ),
    }

    results = []
    print("\n--- PAT Embedding Classification Results ---")

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)

        results.append({"Classifier": name, "Test Accuracy": acc})
        print(f"\n{name}  |  Accuracy: {acc:.2%}")
        print(classification_report(
            y_test, y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        ))

    summary = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)
    print("\n--- Summary ---")
    print(summary.round(3).to_string(index=False))
    return summary


def plot_embedding_vs_handcrafted(
    embedding_accuracies: dict,
    handcrafted_accuracy: float,
    save_fname: str = "embedding_vs_handcrafted.png",
) -> None:
    """
    Bar chart comparing PAT embedding classifier accuracies against the best
    handcrafted-feature XGBoost accuracy from the main pipeline.

    Parameters
    ----------
    embedding_accuracies : {classifier_name: accuracy}
    handcrafted_accuracy : best overall accuracy from run_feature_set_comparison
    """
    all_results = dict(embedding_accuracies)
    all_results["XGBoost\n(Handcrafted)"] = handcrafted_accuracy

    df = pd.DataFrame(
        list(all_results.items()), columns=["Model", "Accuracy"]
    ).sort_values("Accuracy", ascending=False)

    colors = ["#2196F3" if "Handcrafted" not in m else "#FF9800"
              for m in df["Model"]]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(df["Model"], df["Accuracy"], color=colors, edgecolor="white", width=0.5)

    for bar, acc in zip(bars, df["Accuracy"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{acc:.1%}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    plt.ylim(0, min(1.0, df["Accuracy"].max() + 0.12))
    plt.ylabel("Test Accuracy")
    plt.title("PAT Embeddings vs. Handcrafted Features", fontsize=14, fontweight="bold")
    plt.xticks(fontsize=10)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="PAT Embedding classifiers"),
        Patch(facecolor="#FF9800", label="Handcrafted features (XGBoost)"),
    ]
    plt.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    _save(save_fname)
    plt.show()


# ── 3. Cosine similarity analysis ────────────────────────────────────────────

def run_similarity_analysis(
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    top_n_atypical: int = 5,
) -> pd.DataFrame:
    """
    Computes cosine similarity between all participant embeddings and reports:
      - Per-group mean intra-group similarity (cohesion)
      - Cross-group mean similarity (overlap)
      - The most atypical participants per group (lowest similarity to group peers)

    Returns a DataFrame of per-participant mean intra-group similarity.
    """
    # Normalise so dot product == cosine similarity
    X_norm   = normalize(X, norm="l2")
    sim_matrix = X_norm @ X_norm.T          # shape (N, N)
    groups   = label_encoder.inverse_transform(y)
    unique_groups = sorted(set(groups))
    n        = len(y)

    # ── intra- and inter-group mean similarity table ──────────────────────────
    print("\n--- Cosine Similarity: Group Cohesion & Overlap ---")
    sim_table = pd.DataFrame(index=unique_groups, columns=unique_groups, dtype=float)

    for g1 in unique_groups:
        idx1 = np.where(groups == g1)[0]
        for g2 in unique_groups:
            idx2 = np.where(groups == g2)[0]
            block = sim_matrix[np.ix_(idx1, idx2)]
            if g1 == g2:
                # exclude self-similarity (diagonal)
                mask = ~np.eye(len(idx1), dtype=bool)
                sim_table.loc[g1, g2] = block[mask].mean()
            else:
                sim_table.loc[g1, g2] = block.mean()

    print(sim_table.round(3).to_string())

    # ── heatmap ───────────────────────────────────────────────────────────────
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        sim_table.astype(float),
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0, vmax=1,
        linewidths=0.5,
        linecolor="white",
    )
    plt.title("Mean Cosine Similarity Between Groups", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("embedding_similarity_heatmap.png")
    plt.show()

    # ── per-participant intra-group similarity (find atypical participants) ───
    participant_sim = []
    for i in range(n):
        group_i  = groups[i]
        peers    = np.where(groups == group_i)[0]
        peers    = peers[peers != i]           # exclude self
        mean_sim = sim_matrix[i, peers].mean() if len(peers) > 0 else np.nan
        participant_sim.append({
            "participant_index": i,
            "group":             group_i,
            "mean_intra_sim":    mean_sim,
        })

    sim_df = pd.DataFrame(participant_sim)

    print(f"\n--- Most Atypical Participants (lowest intra-group similarity, "
          f"top {top_n_atypical} per group) ---")
    for group in unique_groups:
        atypical = (
            sim_df[sim_df["group"] == group]
            .nsmallest(top_n_atypical, "mean_intra_sim")
        )
        print(f"\n{group.upper()}")
        print(atypical[["participant_index", "mean_intra_sim"]].round(3).to_string(index=False))

    return sim_df


def plot_similarity_distributions(
    sim_df: pd.DataFrame,
    label_encoder: LabelEncoder,
    save_fname: str = "embedding_similarity_distributions.png",
) -> None:
    """
    Violin plot of intra-group similarity distributions per diagnostic group.
    Wide, high-similarity distributions indicate tight, cohesive groups.
    Flat or low distributions suggest heterogeneity within that group.
    """
    unique_groups = sorted(sim_df["group"].unique())
    colors        = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    palette       = {g: c for g, c in zip(unique_groups, colors)}

    plt.figure(figsize=(9, 5))
    sns.violinplot(
        data=sim_df,
        x="group",
        y="mean_intra_sim",
        palette=palette,
        inner="box",
        order=unique_groups,
    )
    plt.title("Intra-Group Similarity Distributions", fontsize=14, fontweight="bold")
    plt.xlabel("Diagnostic Group")
    plt.ylabel("Mean Cosine Similarity to Group Peers")
    plt.tight_layout()
    _save(save_fname)
    plt.show()