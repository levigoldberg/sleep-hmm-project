# EEG Epoch Frequency Band Visualization


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import mne

# ============================================
# YOUR EXISTING PROJECT CONSTANTS
# ============================================

CHANNEL = "EEG Fpz-Cz"
EPOCH_SECONDS = 30

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (13, 30),
}

# ============================================
# BANDPASS FILTER
# ============================================


def bandpass_filter(signal, sfreq, lowcut, highcut, order=4):
    """
    Apply a Butterworth bandpass filter.
    """

    nyquist = 0.5 * sfreq

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="band")

    filtered = filtfilt(b, a, signal)

    return filtered


# ============================================
# LOAD ONE EEG EPOCH
# ============================================


def load_single_epoch(psg_path, epoch_index=0):
    """
    Load one 30-second EEG epoch from a PSG EDF file.

    Parameters
    ----------
    psg_path : str
        Path to PSG EDF file.

    epoch_index : int
        Which epoch to extract.

    Returns
    -------
    epoch_signal : np.ndarray
        Raw EEG signal for one epoch.

    sfreq : int
        Sampling frequency.
    """

    raw = mne.io.read_raw_edf(psg_path, preload=True)

    raw.pick([CHANNEL])

    sfreq = int(raw.info["sfreq"])

    data = raw.get_data()[0]

    samples_per_epoch = sfreq * EPOCH_SECONDS

    start = epoch_index * samples_per_epoch
    end = start + samples_per_epoch

    epoch_signal = data[start:end]

    return epoch_signal, sfreq


# ============================================
# VISUALIZATION
# ============================================


def visualize_epoch_bands(epoch_signal, sfreq):
    """
    Create a visualization similar to your screenshot.

    Top panel:
        Individual EEG frequency bands.

    Bottom panel:
        Original raw EEG epoch.
    """

    # Time axis
    time = np.arange(len(epoch_signal)) / sfreq

    # Extract each EEG band
    delta_signal = bandpass_filter(
        epoch_signal,
        sfreq,
        BANDS["delta"][0],
        BANDS["delta"][1],
    )

    theta_signal = bandpass_filter(
        epoch_signal,
        sfreq,
        BANDS["theta"][0],
        BANDS["theta"][1],
    )

    alpha_signal = bandpass_filter(
        epoch_signal,
        sfreq,
        BANDS["alpha"][0],
        BANDS["alpha"][1],
    )

    beta_signal = bandpass_filter(
        epoch_signal,
        sfreq,
        BANDS["beta"][0],
        BANDS["beta"][1],
    )

    # ========================================
    # PLOT
    # ========================================

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(14, 10),
        sharex=True,
    )

    # ----------------------------------------
    # Delta
    # ----------------------------------------

    axes[0].plot(time, delta_signal)
    axes[0].set_title("Delta Band (0.5 - 4 Hz)")
    axes[0].set_ylabel("Amplitude")

    # ----------------------------------------
    # Theta
    # ----------------------------------------

    axes[1].plot(time, theta_signal)
    axes[1].set_title("Theta Band (4 - 8 Hz)")
    axes[1].set_ylabel("Amplitude")

    # ----------------------------------------
    # Alpha
    # ----------------------------------------

    axes[2].plot(time, alpha_signal)
    axes[2].set_title("Alpha Band (8 - 12 Hz)")
    axes[2].set_ylabel("Amplitude")

    # ----------------------------------------
    # Beta
    # ----------------------------------------

    axes[3].plot(time, beta_signal)
    axes[3].set_title("Beta Band (13 - 30 Hz)")
    axes[3].set_ylabel("Amplitude")

    # ----------------------------------------
    # Raw EEG
    # ----------------------------------------

    axes[4].plot(time, epoch_signal)
    axes[4].set_title("Raw EEG Epoch")
    axes[4].set_ylabel("Amplitude")
    axes[4].set_xlabel("Time (seconds)")

    plt.tight_layout()
    plt.show()


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":

    # Replace with your real PSG EDF path
    psg_path = "data/sleep-cassette/SC4001E0-PSG.edf"

    # Which epoch to visualize
    epoch_index = 100

    # Load one epoch
    epoch_signal, sfreq = load_single_epoch(
        psg_path,
        epoch_index,
    )

    # Visualize
    visualize_epoch_bands(epoch_signal, sfreq)
