#!/usr/bin/env python
# coding: utf-8

# Pranav Minasandra
# 16 Apr 2026
# pminasandra.github.io

"""
Evaluates clustering within UMAP projections
"""

from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
import hdbscan
import numpy as np
import scipy

import config


def rand_index_score(clusters, classes):
    """
    Compute the Rand Index between two clusterings.

    The Rand Index measures the similarity between two data partitions by
    considering all pairs of samples and counting pairs that are assigned
    consistently (either in the same cluster in both labelings or in different
    clusters in both labelings).

    Args:
        clusters (array-like of shape (n_samples,)):
            Predicted cluster labels for each sample. Labels need not be
            consecutive or start from zero.
        classes (array-like of shape (n_samples,)):
            Ground truth class labels for each sample. Labels need not be
            consecutive or start from zero.

    Returns:
        float:
            Rand Index score in the range [0, 1], where 1 indicates perfect
            agreement between the two labelings and 0 indicates no agreement.

    Notes:
        - This implementation computes pair counts explicitly using combinatorics.
        - It does not correct for chance (i.e., this is not the Adjusted Rand Index).
        - Both inputs must have the same length.
    """
    tp_plus_fp = scipy.special.comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = scipy.special.comb(np.bincount(classes), 2).sum()
    a = np.c_[(clusters, classes)]
    tp = sum(scipy.special.comb(np.bincount(a[a[:, 0] == i, 1]), 2).sum() for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = scipy.special.comb(len(a), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


def calc_rand(pred, true):
    """
    Compute the Rand Index between predicted and true labels, ensuring labels
    are reindexed to consecutive integers.

    This function normalizes arbitrary label values (e.g., strings or non-
    consecutive integers) into a compact integer representation before
    computing the Rand Index.

    Args:
        pred (array-like of shape (n_samples,)):
            Predicted cluster labels for each sample.
        true (array-like of shape (n_samples,)):
            Ground truth class labels for each sample.

    Returns:
        float:
            Rand Index score in the range [0, 1].

    Notes:
        - Uses `numpy.unique(..., return_inverse=True)` to remap labels to
          consecutive integers.
    """
    _, pred = np.unique(pred, return_inverse=True)
    _, true = np.unique(true, return_inverse=True)
    return rand_index_score(pred, true)


def labels_and_umap(df):
    """
    Helper, extracts label- and umap- columns.
    """
    labels = df[config.LABEL_COL]
    umap_cols = [col for col in df.columns if 'UMAP' in col]
    embedding = np.asarray(df[umap_cols])

    return labels, embedding


def assign_cluster_labels(embedding,
                            min_cluster_size=None,
                            cluster_selection_method='leaf'):
    """
    Cluster an embedding using HDBSCAN and return cluster labels.

    Args:
        embedding (array-like of shape (n_samples, n_features)):
            Low-dimensional representation of the data (e.g., UMAP output)
            on which clustering will be performed.
        min_cluster_size (int, optional):
            The minimum size of clusters. Smaller values allow detection of
            smaller clusters but may increase noise. If None, it is set to
            1.5% of the number of samples.
        cluster_selection_method (str, optional):
            Method used to select clusters from the condensed tree. Common
            options are 'leaf' (more fine-grained clusters) and 'eom'
            (excess of mass; more conservative clustering). Default is 'leaf'.

    Returns:
        numpy.ndarray of shape (n_samples,):
            Cluster labels assigned by HDBSCAN. Noise points are labeled as -1.
    """
    if min_cluster_size is None:
        min_cluster_size=int(0.015*embedding.shape[0])

    scan = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        cluster_selection_method=cluster_selection_method
                        ).fit(embedding)
    hdb_labels = scan.labels_ # these are the predicted clusters labels
    return hdb_labels


def hdb_noise_mask(hdb_labels):
    """
    True for the datapoints labelled as noise (-1) by HDBSCAN
    """

    return hdb_labels == -1


def compare_hdb_to_real(hdb_labels, real_labels, embedding):
    """
    Compare clustering labels against ground truth using standard metrics.

    Args:
        hdb_labels (array-like of shape (n_samples,)):
            Cluster labels predicted by HDBSCAN.
        real_labels (array-like of shape (n_samples,)):
            Ground truth class labels.
        embedding (np.ndarray):
            List of UMAP values

    Returns:
        dict:
            Dictionary containing:
            - 'rand_index' (float): Rand Index score.
            - 'adjusted_rand_index' (float): Adjusted Rand Index score.
            - 'silhouette_score' (float): Silhouette score of the clustering.
            - 'n_clust' (int): Number of unique clusters.
    """
    results = {}
    results['rand_index'] = calc_rand(hdb_labels, real_labels)
    results['adjusted_rand_index'] = adjusted_rand_score(hdb_labels, real_labels)
    results['silhouette_score'] = silhouette_score(embedding, hdb_labels)
    results['n_clust'] = len(list(set(hdb_labels)))

# someone wants to add mutual information here? that one makes a lot more sense to me?
    return results


def overall_cluster_comparison_analyses(df):
    """
    Run clustering and evaluation pipeline with and without noise points.

    Args:
        df (pandas.DataFrame):
            Input dataset from which labels and embeddings are derived.

    Returns:
        tuple:
            (evals_noise, evals_no_noise, noise_fraction), where:
            - evals (dict): Evaluation metrics on all points, with and without noise.
    """
    labels, embeddings = labels_and_umap(df)
    clustering = assign_cluster_labels(embeddings)
    evals_noise = compare_hdb_to_real(clustering, labels, embeddings)

    unclustered = hdb_noise_mask(clustering) # a mask
    evals_no_noise = compare_hdb_to_real(clustering[~unclustered],
                                            labels[~unclustered],
                                            embeddings[~unclustered]
                                            )
    evals_no_noise = {f"{x}_no_noise": y for (x, y) in evals_no_noise.items()}

    return {**evals_noise, **evals_no_noise,
                "frac_noise": sum(unclustered)/len(unclustered)}
