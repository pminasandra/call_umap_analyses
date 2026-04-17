#!/usr/bin/env python
# coding: utf-8

# Pranav Minasandra
# 16 Apr 2026
# pminasandra.github.io

"""
Generates a de-dimensionalisation of spectrogram inputs using UMAP.
"""

import time

import numpy as np
import umap

import config
from functions.preprocessing_functions import calc_zscore, pad_spectro


def preprocess_spectrograms(df, preprocess=True):
    """
    z-transforms spectrogram values, does zero-padding for max length.
    """
    specs = df[config.SPEC_COL]

    data = specs
    if preprocess:
        specs = [calc_zscore(s) for s in specs]
        maxlen= np.max([spec.shape[1] for spec in specs])
        flattened_specs = [pad_spectro(spec, maxlen).flatten() for spec in specs]
        data = flattened_specs
    data = np.asarray(data)

    return data


def project_to_umap_space(data,
                            n_components=config.N_DIM,
                            metric=config.DISTANCE_METRIC,
                            min_dist=0
                         ):
    """
    Fits a reducer, and projects all spectrograms to a lower dimensional space.
    Args:
        data (array-like or pandas.DataFrame):
            Input data to be projected into UMAP space. Shape should be
            (n_samples, n_features).
        n_components (int):
            The dimensionality of the embedding (e.g., 2 for 2D, 3 for 3D).
        metric (str or callable):
            The distance metric to use for computing distances in the input space.
            Common options include 'euclidean', 'manhattan', 'cosine', etc.
        min_dist (float, optional):
            The effective minimum distance between embedded points. Smaller values
            result in tighter clustering, while larger values preserve more global
            structure. Default is 0.
        random_state (int, optional):
            Seed for the random number generator to ensure reproducibility.
            Defaults to the current system time.
    Returns:
        embedding (np.ndarray)
        reducer (umap.UMAP)
    """

    reducer = umap.UMAP(n_components=n_components,
                        metric=metric,
                        min_dist=min_dist)

    embedding = reducer.fit_transform(data)

    return embedding, reducer


def add_umap_data_to_df(df, embedding, inplace=False):
    """
    Helper, adds umap projections to main df
    """
    if not inplace:
        df = df.copy()

    n_dims = embedding.shape[1]
    umap_labels = [f"UMAP{i+1}" for i in range(n_dims)]
    df.loc[:, umap_labels] = embedding

    return df

def perform_umap_reduction_and_store(df, preprocess=True):
    """
    Wrapper, performs a UMAP reduction and fits to all spectrograms, adds UMAP columns
    to your df.
    """
    data = preprocess_spectrograms(df, preprocess=preprocess)
    embeddings, _ = project_to_umap_space(data)
    df = add_umap_data_to_df(df, embeddings)

    return df
