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

WELCH_SEGMENT_LENGTH = 4
