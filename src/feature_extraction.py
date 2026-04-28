import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
import plotly.graph_objects as go
from visualizations import (
    plot_feature_heatmap,
    plot_smoothed_bandpower,
    plot_raw_epoch_window,
)
from consants import PSG_PATH, CHANNEL, EPOCH_SECONDS, BANDS, WELCH_SEGMENT_LENGTH







def compute_bandpower(epoch_signal, sfreq):
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
        relative_powers:
            A dictionary containing the relative power in each EEG band.
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
        # Example: for delta, keep frequencies from 0.5 Hz up to but not
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
    # TODO:
    relative_powers = {
        f"{band}_rel": power / total_power
        for band, power in band_powers.items()
    }

    # Return one feature vector for this epoch.
    # Example:
    # {
    #   "delta_rel": 0.65,
    #   "theta_rel": 0.20,
    #   "alpha_rel": 0.10,
    #   "beta_rel": 0.05
    # }
    return relative_powers


def extract_features():
    """
    Main feature extraction pipeline.

    This function:
    1. Loads one Sleep-EDF PSG file.
    2. Selects one EEG channel.
    3. Splits the signal into 30-second epochs.
    4. Computes relative band-power features for each epoch.
    5. Saves the resulting feature table as a CSV file.
    """

    # Load the raw EDF file.
    #
    # preload=True loads the signal data into memory immediately.
    # This is fine for one file and makes later operations faster/easier.
    raw = mne.io.read_raw_edf(PSG_PATH, preload=True)

    # Print all channel names in the file.
    # This is useful for checking the exact channel names available in the EDF.
    # The channel name must match exactly when selecting it.
    print(raw.ch_names)

    # Keep only the EEG channel we want to analyze.
    # This simplifies the data so we are working with one signal.
    raw.pick([CHANNEL])

    # Get the sampling frequency.
    # If sfreq = 100, this means the EEG has 100 samples per second.
    sfreq = int(raw.info["sfreq"])

    # Extract the actual EEG signal values.
    #
    # raw.get_data() returns a 2D array: channels x samples.
    # Since we selected only one channel, [0] gives the one EEG time series.
    data = raw.get_data()[0]

    # Convert epoch length from seconds into number of samples.
    #
    # Example:
    # 30 seconds * 100 samples/second = 3000 samples per epoch.
    samples_per_epoch = EPOCH_SECONDS * sfreq

    # Calculate the number of complete 30-second epochs in the recording.
    #
    # Integer division // ignores any leftover partial epoch at the end.
    # This is appropriate because the hypnogram labels correspond to complete
    # 30-second epochs.
    num_epochs = len(data) // samples_per_epoch

    # Each item in rows will become one row in the final feature table.
    rows = []

    # Loop through each complete 30-second epoch.
    for i in range(num_epochs):

        # Compute the start and end sample indices for this epoch.
        start = i * samples_per_epoch
        end = start + samples_per_epoch

        # Extract the EEG signal for this epoch only.
        epoch_signal = data[start:end]

        # Compute relative band-power features for this epoch.
        features = compute_bandpower(epoch_signal, sfreq)

        # Add this epoch's features to the list of rows.
        rows.append(features)

    # Convert the list of dictionaries into a pandas DataFrame.
    # Each row is one 30-second epoch.
    # Each column is one feature.
    df = pd.DataFrame(rows)
    plot_feature_heatmap(df, epoch_seconds=EPOCH_SECONDS, smooth_window=5)

    plot_smoothed_bandpower(df, epoch_seconds=EPOCH_SECONDS, smooth_window=10)

    plot_raw_epoch_window(
        data=data,
        sfreq=sfreq,
        epoch_seconds=EPOCH_SECONDS,
        start_epoch=0,
        num_epochs_to_show=10,
    )
    # Print the first few rows to confirm that feature extraction worked.
    print(df.head())

    # Print the shape of the table.
    # This tells us: number of epochs x number of columns.
    print(df.shape)

    # Save the feature table as a CSV file.
    #
    # This CSV can later be loaded by the HMM training script.
    # Saving features separately means we do not need to reload and process
    # the raw EDF file every time we train the model.
    df.to_csv("results/features_one_subject.csv", index=False)
    print("Sampling frequency:", sfreq, "Hz")
    return df, data, sfreq

def plot_raw_epochs(data, sfreq, samples_per_epoch, num_epochs, max_epochs=20):
    """
    Plot the raw EEG signal over time with vertical lines showing epoch boundaries.

    max_epochs controls how many 30-second epochs are shown at once.
    Showing the full night can be visually overwhelming, so we start with a chunk.
    """

    # Only plot the first max_epochs so the graph is readable
    samples_to_plot = min(max_epochs * samples_per_epoch, len(data))

    signal_chunk = data[:samples_to_plot]
    time_seconds = np.arange(samples_to_plot) / sfreq

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time_seconds,
            y=signal_chunk,
            mode="lines",
            name="Raw EEG"
        )
    )

    # Add vertical lines every 30 seconds
    for epoch_idx in range(min(max_epochs, num_epochs) + 1):
        x = epoch_idx * EPOCH_SECONDS
        fig.add_vline(
            x=x,
            line_width=1,
            line_dash="dash"
        )

    fig.update_layout(
        title=f"Raw EEG Signal Split into {EPOCH_SECONDS}-Second Epochs",
        xaxis_title="Time (seconds)",
        yaxis_title="EEG voltage",
        hovermode="x unified",
        width=1100,
        height=500
    )

    fig.write_html("results/raw_epochs_interactive.html")
    fig.show()


def plot_bandpower_over_epochs(df):
    """
    Plot relative EEG band powers over time.

    Each row in df is one 30-second epoch.
    The x-axis is epoch number, which represents time moving forward.
    """

    df = df.copy()
    df["epoch"] = np.arange(len(df))
    df["time_minutes"] = df["epoch"] * EPOCH_SECONDS / 60

    fig = go.Figure()

    feature_columns = ["delta_rel", "theta_rel", "alpha_rel", "beta_rel"]

    for col in feature_columns:
        fig.add_trace(
            go.Scatter(
                x=df["time_minutes"],
                y=df[col],
                mode="lines",
                name=col
            )
        )

    fig.update_layout(
        title="Relative EEG Band Power Across Sleep Epochs",
        xaxis_title="Time (minutes)",
        yaxis_title="Relative power",
        hovermode="x unified",
        width=1100,
        height=600
    )

    fig.write_html("results/bandpower_epochs_interactive.html")
    fig.show()
