#!/usr/bin/env python
# coding: utf-8

# Pranav Minasandra
# (building on ipynb from Mara Thomas)
# 15 Apr 2026

"""
Initial pass, generation of spectrograms, sanity checks.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

import config
from functions.audio_functions import generate_mel_spectrogram, read_wavfile
from functions.audio_functions import butter_bandpass_filter
from functions.audio_functions import generate_stretched_mel_spectrogram


def load_info_file(info_file=config.INFO_FILE):
    """
    Loads/generates data from info file.
    """
    if os.path.isfile(info_file):
        df = pd.read_csv(info_file, sep=",")
    else:
        warnings.warn(f"load_info_file: {info_file} invalid, generating default info file.")
        audiofiles = os.listdir(config.AUDIO_IN)
        if len(audiofiles)>0:
            df = pd.DataFrame({'filename': [os.path.basename(x) for x in audiofiles],
                                    'label': ["unknown"] * len(audiofiles)})
    if config.KEEP_CALLS is not None:
        df = df.loc[df[config.LABEL_COL].isin(config.KEEP_CALLS)]


    audiofiles = df['filename'].values
    files_in_audio_directory = os.listdir(config.AUDIO_IN)

    # Are there any files that are in the info_file.csv, but not in AUDIO_IN?
    missing_files = list(set(audiofiles) - set(files_in_audio_directory))
    if len(missing_files)>0:
        warnings.warn(f"{len(missing_files)} files with no matching audio in audio folder")

    return df


def load_audio_data(df, audio_in, inplace=False):
    """
    Loads audio data pointed to by info_file
    """
    if not inplace:
        df = df.copy()
    audiofiles = df['filename'].values
    audio_filepaths = [os.path.join(audio_in ,x) for x in audiofiles]
    raw_audio,samplerate_hz = map(list,zip(*[read_wavfile(x) for x in audio_filepaths]))

    df['raw_audio'] = raw_audio
    df['samplerate_hz'] = samplerate_hz

    nrows = df.shape[0]
    df.dropna(subset=['raw_audio'], inplace=True)

    if nrows - df.shape[0] > 0:
        warnings.warn(f"load_audio_data: dropped {nrows-df.shape[0]}\
rows due to missing/failed audio")

    df['original_label'] = df[config.LABEL_COL]
    df[config.LABEL_COL] = df[config.LABEL_COL].mask(df[config.LABEL_COL]\
                                .isin(config.NA_DESCRIPTORS),
                            config.NEW_NA_INDICATOR)
    df[config.LABEL_COL] = df[config.LABEL_COL].astype(str)

    return df


def filter_inputs_by_duration(df,
                    min_dur=config.MIN_DUR,
                    max_dur=config.MAX_DUR,
                    inplace=False
                    ):
    """
    Drop calls that are too long.
    """
    if not inplace:
        df = df.copy()
    df['duration_s'] = [x.shape[0] for x in df['raw_audio']]/df['samplerate_hz']

    df = df.loc[df['duration_s'] >= min_dur, :]
    df = df.loc[df['duration_s'] <= max_dur, :]

    return df


def save_spectrogram_info(n_mels, fft_win, fft_hop, window, f_min, f_max,
                            fname=config.SPECTROGRAM_PARAMS_FILE, **kwargs):
    """
    Saves spectrogram-related parameters to a json file.
    """
    specs = {
                "n_mels":   n_mels,
                "fft_win":  fft_win,
                "fft_hop":  fft_hop,
                "window":   window,
                "f_min":    f_min,
                "f_max":    f_max,
                **kwargs
            }
    with open(fname, "w") as specfilejson:
        json.dump(specs, specfilejson)


def add_mel_spectrograms(df, n_mels=config.N_MELS,
                                fft_win=config.FFT_WIN,
                                fft_hop = None,
                                window=config.WINDOW,
                                f_min=config.F_MIN,
                                f_max=None,
                                inplace=False):
    """
    In this step, spectrograms are generated from audio files via short-time fourier
    transformation. Spectrograms capture the frequency components of a signal over time.
    A spectrogram is a 2D matrix, where each value represents the signal intensity in a
    specific time (columns) and frequency bin (row). In this case, the frequency axis of
    the spectrograms are also Mel-transformed (a logarithmic scale) and signal intensity
    is expressed on a Decibel scale.
    """
    if not inplace:
        df = df.copy()

    if fft_hop is None:
        fft_hop = fft_win/8 # hop_length in seconds
                            # FFT_HOP * samplerate = n of audio datapoints between
                            # successive ffts (=hop_length)

    if f_max is None:
        f_max = int(np.min(df['samplerate_hz'])/2)
        # upper bound for frequency (in Hz) when generating Mel filterbank
        # this is set to 0.5 times the samplerate (-> Nyquist rule)
        # If input files have different samplerates, the lowest samplerate is used
        # to ensure all spectrograms have the same frequency resolution.

    save_spectrogram_info(n_mels, fft_win, fft_hop, window, f_min, f_max)

    spectrograms = df.apply(lambda row: generate_mel_spectrogram(
                                data = row['raw_audio'],
                                 rate = row['samplerate_hz'],
                                 n_mels = n_mels,
                                 window = window,
                                 fft_win = fft_win,
                                 fft_hop = fft_hop,
                                 f_max = f_max,
                                 f_min = f_min),
                            axis=1)

    df[config.SPEC_COL] = spectrograms

    nrows = df.shape[0]
    df.dropna(subset=[config.SPEC_COL], inplace=True)
    if nrows - df.shape[0] > 0:
        warnings.warn(
            f"add_mel_spectrograms: {nrows-df.shape[0]} rows dropped: failed" +
                " spectrogram generation")

    return df


def apply_median_subtraction(df, inplace=False):
    """
    Creates denoised spectrograms from median subtraction
    """

    if not inplace:
        df = df.copy()
    df[config.SPEC_COL] =\
        [(spectrogram - np.median(spectrogram, axis=0)) for spectrogram in df[config.SPEC_COL]]
    return df


def apply_bandpass_filter(df, lowcut, highcut, n_mels_filtered=None, inplace=False):
    """
    Applies a bandpass filter to retain specific info only
    Args:
        df (pd.DataFrame)
        lowcut (float): frequency (Hz), only freqs above this are retained
        highcut (float): frequency (Hz), only freqs below this are retained
        n_mels_filtered (int): new n_mels value
        inplace (bool, default False): edit df inplace?
    """
    if not inplace:
        df = df.copy()

    # First, update spec_params
    with open(config.SPECTROGRAM_PARAMS_FILE) as specs:
        existing_specs = json.load(specs)

    if n_mels_filtered is None:
        n_mels_filtered = existing_specs['n_mels']

    existing_specs = {**existing_specs,
                            "lowcut": lowcut,
                            "highcut": highcut,
                            "n_mels_filtered": n_mels_filtered}

    with open(config.SPECTROGRAM_PARAMS_FILE, "w") as specs:
        json.dump(existing_specs, specs)

    df['filtered_audio'] = df.apply(lambda row: butter_bandpass_filter(
                                                    data = row['raw_audio'],
                                                    lowcut = lowcut,
                                                    highcut = highcut,
                                                    sr = row['samplerate_hz'],
                                                    order = 6),
                                            axis=1)

    # create spectrograms from filtered audio
    es = existing_specs
    df[config.SPEC_COL] = df.apply(lambda row: generate_mel_spectrogram(
                                                    data = row['filtered_audio'],
                                                    rate = row['samplerate_hz'],
                                                    n_mels = es['n_mels_filtered'],
                                                    window = es['window'],
                                                    fft_win = es['fft_win'],
                                                    fft_hop = es['fft_hop'],
                                                    f_max = highcut,
                                                    f_min = lowcut
                                                ),
                                            axis=1)

    return df


def apply_time_stretch(df, inplace=False):
    """
    Stretches all calls to be of the same length
    """
    if not inplace:
        df = df.copy()

    with open(config.SPECTROGRAM_PARAMS_FILE) as specs:
        es = json.load(specs)


    max_duration = np.max(df['duration_s'])

    gsms = generate_stretched_mel_spectrogram
    df[config.SPEC_COL] = df.apply(lambda row: gsms(
                                               row['raw_audio'],
                                               row['samplerate_hz'],
                                               row['duration_s'],
                                               es['n_mels'],
                                               es['window'],
                                               es['fft_win'],
                                               es['fft_hop'],
                                               max_duration
                                               ),
                                           axis=1)
    return df


def load_audio_data_and_features(audio_in=config.AUDIO_IN, info_file=config.INFO_FILE):
    """
    Wrapper, loads and generates appropriate spectrogram info
    """

    df = load_info_file(info_file)
    df = load_audio_data(df, audio_in)
    df = filter_inputs_by_duration(df)
    df = add_mel_spectrograms(df)

    if config.BANDPASS_FILTER:
        df = apply_bandpass_filter(df, lowcut=config.LOWCUT, highcut=config.HIGHCUT)
    if config.STRETCH:
        df = apply_time_stretch(df)
    if config.MEDIAN_SUB:
        df = apply_median_subtraction(df)

    return df
