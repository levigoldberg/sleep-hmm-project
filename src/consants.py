import numpy as np

# EEG channel to extract.
# Sleep-EDF contains two EEG channels: Fpz-Cz and Pz-Oz.
# We start with Fpz-Cz because it is one of the main EEG channels provided
# in the dataset. Later, we can repeat this for Pz-Oz or combine both.
CHANNEL = "EEG Fpz-Cz"


# Sleep stages in Sleep-EDF are scored in 30-second windows.
# We use the same 30-second window size so that each feature vector will
# correspond to one clinically annotated sleep-stage epoch.
EPOCH_SECONDS = 30


# Standard EEG frequency bands.
# Each band is defined by a lower and upper frequency bound in Hz.
# These bands summarize different types of brain rhythms that are relevant
# for sleep staging.
BANDS = {
    "delta": (0.5, 4),  # Slow-wave activity, especially important for deep sleep
    "theta": (4, 8),  # Common in lighter sleep stages
    "alpha": (8, 12),  # Often associated with relaxed wakefulness
    "beta": (13, 30),  # Faster activity, often more wake-like
}

<<<<<<< HEAD
# Number of hidden states in the HMM.
# Based on modern sleep staging conventions (AASM), sleep is divided into
# 5 stages. We use 5 hidden states to match this standard. Later, we could
# experiment with 4 states (merging N1 and N2) or 6 states to see if it
# improves accuracy.
K = 5

# Number of features per epoch.
# Each epoch is represented as a vector of 4 relative power values,
# one per standard EEG frequency band: delta, theta, alpha, beta.
# This matches the output of Levi's feature extraction module.
F = 4

# Names of the hidden states, in order.
# These correspond to the 5 standard sleep stages. The model learns
# these in an unsupervised way — the names are just for interpretation.
STATE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']



# Variance for each state and feature, used for the Gaussian emission model.
# Currently all states share the same variance (0.02, std ~0.14), which gives
# realistic scatter without too much noise. Later, we could use different
# variances per state — N3 is likely tighter (always high delta) while
# Wake may be more variable.
TRUE_VARS = np.full((K, F), 0.02)



# Transition matrix initialization.
# Currently uniform — every transition equally likely. This is used as
# a starting point before Baum-Welch learns the real transitions from data.
# Alternatives: random (np.random.dirichlet) or informed (based on known
# sleep architecture Wake→N1→N2→N3→REM, from AASM transition research).
A_init = np.full((K, K), 1/K)

=======

#training constants
THRESHOLD = 1e-4
ITERATIONS = 20
D = len(BANDS)
K = 5
WELCH_SEGMENT_LENGTH = 4
>>>>>>> 2ee9e5d8c062f5e407df8e5a0a559cb7c68828e9
