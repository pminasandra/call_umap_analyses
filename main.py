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

import config
import generate_spectrograms
import generate_umaps


if __name__ == "__main__":
    df = generate_spectrograms.load_audio_data_and_features()
