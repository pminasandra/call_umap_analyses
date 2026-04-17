# Pranav Minasandra
# 17 Apr 2026
# pminasandra.github.io

"""
Carries out simplified version of analyses described in 
Thomas, M., Jensen, F. H., Averly, B., Demartsev, V., Manser, M. B., Sainburg, T., Roch,
M. A., & Strandburg-Peshkin, A. (2022). A practical guide for generating unsupervised,
spectrogram-based latent space representations of animal vocalizations. Journal of
Animal Ecology, 91, 1567–1581. https://doi.org/10.1111/1365-2656.13754
"""

import os.path

import matplotlib.pyplot as plt
import pandas as pd

import config
import generate_spectrograms
import generate_umaps
import umap_clustering
import umap_eval
import umap_visualisation

if __name__ == "__main__":

# First, load in audio files and create appropriate spectrograms
    df = generate_spectrograms.load_audio_data_and_features()


# With a set of replicates, evaluate the values of standard metrics
    metrics = []
    FIRST = True

    for i in range(config.N_REPEATS):
        print("Estimating UMAP metrics calculation for" +
                    f" repeat {i+1} of {config.N_REPEATS}.")
        df = generate_umaps.perform_umap_reduction_and_store(df, preprocess=True)
        estimates = umap_clustering.overall_cluster_comparison_analyses(df)
        _, s_ests = umap_eval.get_nn_stats(df)
        metrics.append({**estimates, **s_ests})
        if FIRST:
            FIRST = False

        metrics = pd.DataFrame(metrics)
        metrics = metrics.describe().loc[['mean', 'std']]
        print(metrics)
        metrics.to_csv(os.path.join(config.OUTPUT, "metrics.csv"))

# Make a {2 or 3}D plot of UMAP embeddings, coloured by labels
    labels, embeddings = umap_clustering.labels_and_umap(df)
    fig, ax = umap_visualisation.plot_umap_embedding(embeddings, labels=labels, s=1)
    plt.savefig(os.path.join(config.OUTPUT, "umap_plot"))

# Extract HDBSCAN labels, save them, plot how they map onto real data
    df['hdbscan'] = umap_clustering.assign_cluster_labels(embeddings)
    df_s = df[[config.LABEL_COL, 'hdbscan'] +
                    [f"UMAP{i+1}" for i in range(config.N_DIM)]]
    df_s.to_csv(os.path.join(config.OUTPUT, "hdbscan labels"), index=False)
    umap_visualisation.rectangular_confusion_matrix(
                                                    df[config.LABEL_COL],
                                                    df['hdbscan'],
                                                    normalize='pred',
                                                    annot=True
                                                )

# Make plots evaluating our labelling
    umap_eval.make_umap_evaluation_plots(df)
