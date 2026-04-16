# Pranav Minasandra
# 15 Apr 2026
# pminasandra.github.io


# DIRECTORIES
P_DIR = os.getcwd() # project directory, by default set to current directory
    
AUDIO_IN = os.path.join(P_DIR, 'audio') # --> wav files here
DATA = os.path.join(P_DIR, 'data')      # temporary storage of data
INFO_FILE = os.path.join(DATA, 'info_file.csv')

if not os.path.isdir(DATA):
    os.mkdir(DATA)

# INFORMATION ABOUT info_file.csv
LABEL_COL = "label"

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

FMIN = 50 #lower bound for frequency (in Hz) when generating Mel filterbank


SPECTROGRAM_PARAMS_FILE = os.path.join(config.DATA, "spectrogram_parameters.json")
