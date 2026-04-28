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

# Mean feature vector for each state, used to generate fake data and
# initialize the emission model. Each row is [delta, theta, alpha, beta].
# Values are based on known EEG patterns for each sleep stage.
# Later, Baum-Welch will update these from real data.
TRUE_MEANS = np.array([
    [0.10, 0.15, 0.40, 0.35],   # Wake: high alpha and beta, low delta
    [0.20, 0.45, 0.20, 0.15],   # N1: theta dominant
    [0.35, 0.35, 0.15, 0.15],   # N2: theta and some delta
    [0.65, 0.20, 0.10, 0.05],   # N3: delta dominant, deep sleep
    [0.20, 0.40, 0.15, 0.25],   # REM: theta and beta, active dreaming
])

# Variance for each state and feature, used for the Gaussian emission model.
# Currently all states share the same variance (0.02, std ~0.14), which gives
# realistic scatter without too much noise. Later, we could use different
# variances per state — N3 is likely tighter (always high delta) while
# Wake may be more variable.
TRUE_VARS = np.full((K, F), 0.02)

# Initial state distribution.
# We start certain in Wake because real sleep recordings begin before
# the person falls asleep. Later, we could try a uniform distribution
# to see how sensitive the model is to this assumption.
pi = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

# Transition matrix initialization.
# Currently uniform — every transition equally likely. This is used as
# a starting point before Baum-Welch learns the real transitions from data.
# Alternatives: random (np.random.dirichlet) or informed (based on known
# sleep architecture Wake→N1→N2→N3→REM, from AASM transition research).
A_init = np.full((K, K), 1/K)

