# -----------------------------
# Project paths
# -----------------------------

DATA_DIR = "data/sleep-cassette"
RESULTS_DIR = "results"
PSG_PATH = "data/sleep-cassette"

# -----------------------------
# Feature names
# -----------------------------

# These must match the columns created in feature_extraction.py.
FEATURE_NAMES = ["delta_rel", "theta_rel", "alpha_rel", "beta_rel"]

# Number of features per epoch.
F = len(FEATURE_NAMES)
D = F

N_PARTICIPANTS = 2

CHANNEL = "EEG Fpz-Cz"

EPOCH_SECONDS = 30

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (13, 30),
}

WELCH_SEGMENT_LENGTH = 4

# -----------------------------
# HMM state setup
# -----------------------------

# Current model version:
# K = 3 means the HMM learns 3 hidden states.
# During validation, we compare these to Wake, NREM, and REM.
#
# Later, if we use K = 5, validation will compare against:
# Wake, N1, N2, N3, REM.
K = 3

# These are the raw unsupervised HMM state names.
# They do not automatically mean Wake, NREM, or REM.
STATE_NAMES = [f"State {i}" for i in range(K)]


# -----------------------------
# Validation labels
# -----------------------------

# Expert labels we compare against for each possible K.
VALIDATION_CLASS_NAMES_BY_K = {
    3: ["Wake", "NREM", "REM"],
    5: ["Wake", "N1", "N2", "N3", "REM"],
}

# Sleep-EDF hypnogram labels mapped into our validation classes.
# Stages 3 and 4 are combined because modern scoring combines them as N3.
SLEEP_EDF_STAGE_MAP_BY_K = {
    3: {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 1,
        "Sleep stage 3": 1,
        "Sleep stage 4": 1,
        "Sleep stage R": 2,
    },
    5: {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,
        "Sleep stage R": 4,
    },
}

# These are selected based on the current K.
VALIDATION_CLASS_NAMES = VALIDATION_CLASS_NAMES_BY_K[K]
SLEEP_EDF_STAGE_MAP = SLEEP_EDF_STAGE_MAP_BY_K[K]

# Labels like "Movement time" and "Sleep stage ?" should not be scored.
INVALID_LABEL = -1


# -----------------------------
# Model output paths
# -----------------------------

MODEL_INITIAL_PROB_PATH = "results/initial_prob.npy"
MODEL_TRANSITION_PATH = "results/transition.npy"
MODEL_MEANS_PATH = "results/means.npy"
MODEL_VARIANCES_PATH = "results/variances.npy"

# -----------------------------
# Manual state labeling
# -----------------------------

# After training, inspect the learned Gaussian means and transition matrix.
# Then manually decide which real sleep label each HMM state represents.
#
# For K = 3, valid labels are:
# "Wake", "NREM", "REM"
#
# CHANGE AFTER INSPECTING TRAINED STATES
HMM_STATE_TO_VALIDATION_LABEL = {
    0: "Wake",
    1: "NREM",
    2: "REM",
}
# Options:
# "posterior"
# "viterbi"
DECODING_METHOD = "viterbi"

# Options for feature extraction
# "relative" - use relative power
# "log relative" - use log of power
FEATURE_EXTRACTION_METHOD = "relative"

# -----------------------------
# Train/test split
# -----------------------------

TRAIN_FRACTION = 0.7
SPLIT_RANDOM_SEED = 42


# -----------------------------
# Numerical stability constants
# -----------------------------

EPSILON = 1e-300
MIN_VARIANCE = 1e-6


# -----------------------------
# Training constants
# -----------------------------

THRESHOLD = 1e-3
ITERATIONS = 20
