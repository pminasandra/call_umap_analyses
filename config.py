# Pranav Minasandra
# 15 Apr 2026
# pminasandra.github.io

"""
Flow and parameter configuration
"""

import os
import os.path

import numpy as np

# DIRECTORIES
P_DIR = os.getcwd() # project directory, by default set to current directory

AUDIO_IN = os.path.join(P_DIR, 'audio') # --> wav files here
DATA = os.path.join(P_DIR, 'data')      # temporary storage of data
INFO_FILE = os.path.join(DATA, 'info_file.csv')

if not os.path.isdir(DATA):
    os.mkdir(DATA)


# GENERAL INFORMATION
LABEL_COL = "label"
SPEC_COL = "spectrogram"
NA_DESCRIPTORS = [0, np.nan, "NA", "na",
                  "not available", "None",
                  "Unknown", "unknown", None, ""]                                                             
NEW_NA_INDICATOR = "unknown"


# BIOLOGICAL SPECIFICS
KEEP_CALLS = ['psherp', 'twerp', 'whistle',
                'tweep', 'c.squak', 'c.squeek', 'c.a.squeek'] #Set to None to keep all


# AUDIO ANALYSES
MIN_DUR = 0  # min duration of audio inputs (seconds)
MAX_DUR = 3  # max duration of audio inputs (seconds)


# SPECTROGRAM GENERATION
BANDPASS_FILTER = True  # bandpass-filtered spectrograms?
MEDIAN_SUB = False  # median-subtracted spectrograms?
STRETCH = False    # time-stretched spectrograms?
N_MELS = 40 # number of mel bins (usually 20-40)
            # The frequency bins are transformed to this
            # number of logarithmically spaced mel bins.
FFT_WIN = 0.010 # length of audio chunk when applying STFT in seconds
                # FFT_WIN * samplerate = number of audio datapoints that go in one fft (=n_fft)
WINDOW = 'hann' # name of window function
                # each frame of audio is windowed by a window function.
                # https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows
F_MIN = 50 #lower bound for frequency (in Hz) when generating Mel filterbank
SPECTROGRAM_PARAMS_FILE = os.path.join(DATA, "spectrogram_parameters.json")


# UMAP METRICS
N_DIM = 2 # 3 for 3D, 2 for 2D, 4 if you are an extraterrestrial capable of
          # higher-dimensional visualisation. (5 is right out)
DISTANCE_METRIC = 'euclidean' # https://umap-learn.readthedocs.io/en/latest/parameters.html#metric


# PLOTTING
DISTINCT_COLORS = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                      '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
                      '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', 
                      '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', 
                      '#ffffff', '#000000']


# ANALYSES METRICS

## NEAREST NEIGHBOR
NUM_NEAREST_NEIGHBOURS = 40
