#!/usr/bin/env python
# coding: utf-8

# Pranav Minasandra
# 16 Apr 2026
# pminasandra.github.io

"""
Stats and plots for evaluating labelling of sound elements/calls.
"""

import numpy as np
import seaborn as sns

import config
import umap_clustering
from functions.evaluation_functions import plot_within_without
from functions.evaluation_functions import nn, sil


def get_nn_stats(df, nn_k=config.NUM_NEAREST_NEIGHBOURS, dropna=True):
    """
    Compute nearest-neighbour label statistics in embedding space.

    Args:
        df (pandas.DataFrame):
            Input dataset containing features and labels.
        nn_k (int, optional):
            Number of nearest neighbours to consider.
        dropna (bool, optional):
            If True, exclude rows where LABEL_COL equals NA_INDICATOR.
    Returns:
        (<evaluation_functions.nn object>,
            float: S score,
            float: Snorm score)

    Notes:
        See https://doi.org/10.1111/1365-2656.13754 for metric details
    """
    df = df.copy()
    if dropna:
        df = df.loc[df[config.LABEL_COL] != config.NEW_NA_INDICATOR,:]

    labels, embeddings = umap_clustering.labels_and_umap(df)

    nn_stats = nn(embeddings, np.asarray(labels), k=nn_k)

    return nn_stats, {"s": nn_stats.get_S(), "s_norm": nn_stats.get_Snorm()}


def make_nn_stat_visualisations(nn_stats, fname_base="nn"):
    """
    Reproduce plots for nearest-neighbour overlap among 'true' classes
    """
    nn_stats.plot_heat_S(vmin=0,       # lower end (for color scheme)
                         vmax=100,     # upper end (for color scheme)
                         center=50,    # center(for color scheme)
                         cmap=sns.color_palette("Greens", as_cmap=True),# color scheme
                         cbar=None,    # show colorbar if True else don't
                         outname=f"{fname_base}_s_score.pdf")

    nn_stats.plot_heat_fold(center=1,    # center(for color scheme)
                            cmap=sns.diverging_palette(20, 145, as_cmap=True),
                            cbar=None,    # show colorbar if True else don't
                            outname=f"{fname_base}_s_normalised.pdf")


    nn_stats.plot_heat_Snorm(vmin=-13,     # lower end (for color scheme)
                             vmax=13,      # upper end (for color scheme)
                             center=1,     # center(for color scheme)
                             cmap=sns.diverging_palette(240, 10, as_cmap=True),
                             cbar=None,    # show colorbar if True else don't
                             outname=f"{fname_base}_s_normalised_logtrans.pdf")


def pairwise_analyses(labels, embeddings, fname_base="pw"):
    """
    Run pairwise distance and silhouette analyses on an embedding.

    Args:
        labels (array-like of shape (n_samples,)):
            Labels associated with each sample.
        embeddings (array-like of shape (n_samples, n_features)):
            Embedding coordinates.
        fname_base (str, optional):
            Base filename used for saving output plots.

    Returns:
        float:
            Average silhouette score.
    """
    plot_within_without(embedding=embeddings,
                        labels=labels,
                        distance_metric=config.DISTANCE_METRIC,
                        xmin=0,xmax=12,
                        ymax=0.5,
                        nbins=150,
                        nrows=7,
                        ncols=2,
                        outname=f"{fname_base}_within_without_disthists.pdf",
                        density=True)

    sil_stats = sil(embeddings, labels)
    sil_stats.plot_sil(outname=f"{fname_base}_silhouette_plot.pdf")

    return sil_stats.get_avrg_score()


def make_umap_evaluation_plots(df):
    """
    Plotting wrapper, makes all relevant plots
    """

    nn_stats, _ = get_nn_stats(df)
    make_nn_stat_visualisations(nn_stats)
    labels, embeddings = umap_clustering.labels_and_umap(df)
    pairwise_analyses(labels, embeddings)
