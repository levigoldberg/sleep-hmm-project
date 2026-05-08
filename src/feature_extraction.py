import os
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from constants import PSG_PATH, CHANNEL, EPOCH_SECONDS, BANDS, WELCH_SEGMENT_LENGTH, FEATURE_EXTRACTION_METHOD


def compute_bandpower(epoch_signal, sfreq, feature_method=None):
    """
    Compute relative EEG band power for one 30-second epoch.

    Parameters:
        epoch_signal:
            A one-dimensional array containing the EEG voltage values
            for one 30-second epoch.

        sfreq:
            Sampling frequency of the EEG signal, in Hz.
            For Sleep-EDF EEG, this is 100 Hz.

    Returns:
        features:
            A dictionary containing the computed features for the epoch.
            These values become the observation vector for one HMM time step.
    """

    # Estimate the power spectral density (PSD) of the epoch.
    #
    # Welch's method is a practical FFT-based method. Instead of applying one
    # Fourier Transform to the entire 30-second epoch, it breaks the epoch into
    # smaller segments, computes spectra, and averages them. This gives a more
    # stable estimate of power across frequencies.
    #
    # nperseg=sfreq * 4 means each Welch segment is 4 seconds long.
    # If sfreq = 100 Hz, then nperseg = 400 samples
    # A 4-second segment gives frequency resolution of about 0.25 Hz.
    # We can treat WELCH_SEGMENT_LENGTH as a tunable parameter and compare
    # different window lengths based on downstream HMM performance

    freqs, psd = welch(epoch_signal, fs=sfreq, nperseg=sfreq * WELCH_SEGMENT_LENGTH)

    # This dictionary will store the absolute power in each band first.
    band_powers = {}

    # Loop through each EEG frequency band.
    for band_name, (low, high) in BANDS.items():

        # Select only the frequency values that fall within this band.
        # Ex: for delta, keep frequencies from 0.5 Hz up to but not
        # including 4 Hz.
        band_mask = (freqs >= low) & (freqs < high)

        # Compute the area under the PSD curve within this frequency band.
        # This area represents the total power in that band.
        #
        # np.trapezoid performs numerical integration using the trapezoid rule.
        band_power = np.trapezoid(psd[band_mask], freqs[band_mask])

        # Store the absolute band power.
        band_powers[band_name] = band_power

    # Compute the total power across the selected bands.
    # This is used to convert absolute power into relative power.
    total_power = sum(band_powers.values())

    # Convert absolute band power into relative band power.
    #
    # Example:
    # delta_rel = delta_power / total_power
    #
    # This makes each feature a proportion rather than a raw power value.
    # Relative power is useful because it reduces the effect of overall signal
    # amplitude differences between recordings, subjects, or channels.
    # Use the default from constants.py unless an experiment passes a method in.
    if feature_method is None:
        feature_method = FEATURE_EXTRACTION_METHOD

    if FEATURE_EXTRACTION_METHOD == "log relative":
        features = {}

        for band, power in band_powers.items():
            relative_power = power / (total_power + 1e-12)
            features[f"{band}_log_rel"] = np.log10(relative_power + 1e-12)
    elif FEATURE_EXTRACTION_METHOD == 'relative':
        features = {
            f"{band}_rel": power / total_power
            for band, power in band_powers.items()
        }
    else:
        raise ValueError("feature_method must be 'relative' or 'log relative'.")

    # Return one feature vector for this epoch.
    # Example:
    # {
    #   "delta_rel": 0.65,
    #   "theta_rel": 0.20,
    #   "alpha_rel": 0.10,
    #   "beta_rel": 0.05
    # }
    return features


def discover_psg_files(data_dir="data/sleep-cassette"):
    """Discover PSG EDF files under a directory excluding EDFs (annotated files)."""
    psg_paths = []
    for root, _, files in os.walk(data_dir):
        for filename in files:
            lower = filename.lower()
            if not lower.endswith(".edf"):
                continue
            if "hypnogram" in lower:
                continue
            if "psg" not in lower:
                continue
            psg_paths.append(os.path.join(root, filename))

    psg_paths.sort()
    return psg_paths


def extract_single_participant_features(psg_path, feature_method = None):
    """Extract per-epoch features for a single participant PSG file."""
    raw = mne.io.read_raw_edf(psg_path, preload=True)
    raw.pick([CHANNEL])
    sfreq = int(raw.info["sfreq"])
    data = raw.get_data()[0]
    samples_per_epoch = EPOCH_SECONDS * sfreq
    num_epochs = len(data) // samples_per_epoch

    rows = []
    for i in range(num_epochs):
        start = i * samples_per_epoch
        end = start + samples_per_epoch
        epoch_signal = data[start:end]
        rows.append(compute_bandpower(epoch_signal, sfreq, feature_method))

    df = pd.DataFrame(rows)
    # if FEATURE_EXTRACTION_METHOD == "log":
    #     df = zscore_features(df)
    return df, data, sfreq


def extract_features_by_participant(psg_paths=None, feature_method=None):
    """
    Extract features for multiple participants without putting the time axes.

    Args:
        psg_paths:
            Optional dict[str, str] mapping participant_id -> PSG path,
            or list[str] of PSG paths (participant ids inferred from filename stem).

    Returns:
        participant_features:
            dict[participant_id, pd.DataFrame] with one row per epoch in-order.
        participant_ids:
            list[str] aligned with sequence_list.
        sequence_list:
            list[np.ndarray] where each element is (T_i, D) for one participant.
    """
    if isinstance(psg_paths, dict):
        items = [(str(k), str(v)) for k, v in psg_paths.items()]
    else:
        items = []
        for p in psg_paths:
            path = str(p)
            participant_id = path.split("/")[-1].split(".")[0]
            items.append((participant_id, path))

    participant_features = {}
    participant_ids = []
    sequence_list = []

    for participant_id, psg_path in items:
        df, _, _ = extract_single_participant_features(psg_path, feature_method) #can ignore data, and sfreq
        participant_features[participant_id] = df
        participant_ids.append(participant_id)
        sequence_list.append(df.to_numpy())

    return participant_features, participant_ids, sequence_list
