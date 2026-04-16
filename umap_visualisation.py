# Pranav Minasandra
# 16 Apr 2026
# pminasandra.github.io

"""
Helper functions to make some repetitive plots
"""

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config

palette=config.DISTINCT_COLORS if config.DISTINCT_COLORS else "Set2"
def plot_umap_embedding(embedding,
                        labels=None,
                        palette=palette,
                        ax=None,
                        s=20,
                        alpha=0.8):
    """
    Plot a 2D or 3D embedding using seaborn/matplotlib.

    Args:
        embedding (numpy.ndarray of shape (n_samples, n_dims)):
            Embedding coordinates. Must have n_dims = 2 or 3.
        labels (array-like of shape (n_samples,), optional):
            Labels for coloring points. If None, points are plotted
            without grouping.
        palette (str or list, optional):
            Seaborn color palette name or list of colors.
        ax (matplotlib.axes.Axes, optional):
            Existing axes to plot on. If None, a new figure and axes
            are created.
        s (float, optional):
            Marker size.
        alpha (float, optional):
            Marker transparency.
    Returns:
        tuple:
            (fig, ax) matplotlib figure and axes objects.
    """

    embedding = np.asarray(embedding)
    n_dims = embedding.shape[1]

    if n_dims not in (2, 3):
        raise ValueError("embedding must have 2 or 3 dimensions")

    if ax is None:
        if n_dims == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if labels is None:
        colors = None
    else:
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        pal = sns.color_palette(palette, n_colors=len(unique_labels))
        color_dict = dict(zip(unique_labels, pal))
        colors = [color_dict[l] for l in labels]

    if n_dims == 2:
        ax.scatter(embedding[:, 0], embedding[:, 1],
                   c=colors, s=s, alpha=alpha)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")

    else:  # 3D
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                   c=colors, s=s, alpha=alpha)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_zlabel("UMAP3")

    if labels is not None:
        handles = [
            plt.Line2D([0], [0], marker='o', linestyle='',
                       color=color_dict[l], label=str(l))
            for l in unique_labels
        ]
        ax.legend(handles=handles)

    return fig, ax

def rectangular_confusion_matrix(y_true,
                                 y_pred,
                                 ax=None,
                                 normalize=None,
                                 cmap="viridis",
                                 annot=True):
    """
    Compute and plot a rectangular confusion matrix with separate label sets.

    Args:
        y_true (array-like of shape (n_samples,)):
            Ground truth labels (rows).
        y_pred (array-like of shape (n_samples,)):
            Predicted/cluster labels (columns).
        ax (matplotlib.axes.Axes, optional):
            Axes to plot on. If None, a new figure and axes are created.
        normalize (str or None, optional):
            If 'true', normalize rows; if 'pred', normalize columns;
            if 'all', normalize entire matrix; if None, no normalization.
        cmap (str, optional):
            Colormap for heatmap.
        annot (bool, optional):
            If True, annotate cells with values.
    Returns:
        tuple:
            (cm, true_labels, pred_labels, fig, ax)
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    all_labels = np.unique(np.concatenate([true_labels, pred_labels]))
    cm_full = confusion_matrix(y_true, y_pred, labels=all_labels)

    idx_true = [np.where(all_labels == l)[0][0] for l in true_labels]
    idx_pred = [np.where(all_labels == l)[0][0] for l in pred_labels]

    cm = cm_full[np.ix_(idx_true, idx_pred)]

    # normalization
    if normalize == "true":
        cm = cm / cm.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        cm = cm / cm.sum(axis=0, keepdims=True)
    elif normalize == "all":
        cm = cm / cm.sum()

    # plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    sns.heatmap(cm,
                ax=ax,
                cmap=cmap,
                annot=annot,
                fmt=".2f" if normalize else "d",
                xticklabels=pred_labels,
                yticklabels=true_labels)

    ax.set_xlabel("Predicted / Cluster labels")
    ax.set_ylabel("True labels")

    return cm, true_labels, pred_labels, fig, ax
