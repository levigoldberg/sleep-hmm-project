import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


FEATURE_COLUMNS = ["delta_rel", "theta_rel", "alpha_rel", "beta_rel"]


def add_time_columns(df, epoch_seconds):
    """
    Add epoch number and time in minutes to the dataframe.

    Each row in the dataframe represents one epoch.
    For your project, each epoch is 30 seconds.
    """

    df = df.copy()

    df["epoch"] = np.arange(len(df))
    df["time_minutes"] = df["epoch"] * epoch_seconds / 60

    return df


def zscore_columns(df, columns):
    """
    Convert each feature column into a z-score.

    This makes the bands easier to compare visually.

    Example:
    If delta power is usually much larger than beta power,
    z-scoring prevents delta from visually dominating the whole chart.
    """

    df = df.copy()

    for col in columns:
        mean = df[col].mean()
        std = df[col].std()

        if std == 0:
            df[col + "_z"] = 0
        else:
            df[col + "_z"] = (df[col] - mean) / std

    return df


def save_plot(fig, html_path, png_path=None):
    """
    Save a Plotly figure as an interactive HTML file.

    If png_path is provided, this also tries to save a PNG image.
    PNG export requires kaleido:
        pip install kaleido

    The HTML file is the easiest one to share because it stays interactive.
    """

    os.makedirs(os.path.dirname(html_path), exist_ok=True)

    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Saved interactive plot to: {html_path}")

    if png_path is not None:
        try:
            fig.write_image(png_path)
            print(f"Saved image to: {png_path}")
        except Exception:
            print("Could not save PNG.")
            print("To enable PNG saving, run:")
            print("pip install kaleido")


def plot_feature_heatmap(df, epoch_seconds=30, smooth_window=5):
    """
    Create an interactive heatmap of EEG features across time.

    This is probably the most useful visualization before training the HMM.

    Rows:
        EEG frequency bands

    Columns:
        Time / epochs

    Colors:
        Whether a feature is high or low relative to its usual value

    Why this helps:
        The HMM is trying to find repeated patterns in the features over time.
        A heatmap makes blocks of similar epochs easier to see.
    """

    df = add_time_columns(df, epoch_seconds)
    plot_df = df.copy()

    if smooth_window > 1:
        for col in FEATURE_COLUMNS:
            plot_df[col] = (
                plot_df[col]
                .rolling(window=smooth_window, center=True, min_periods=1)
                .mean()
            )

    plot_df = zscore_columns(plot_df, FEATURE_COLUMNS)

    z_matrix = np.array(
        [
            plot_df["delta_rel_z"].values,
            plot_df["theta_rel_z"].values,
            plot_df["alpha_rel_z"].values,
            plot_df["beta_rel_z"].values,
        ]
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=plot_df["time_minutes"],
            y=["delta", "theta", "alpha", "beta"],
            colorbar=dict(title="z-score"),
        )
    )

    fig.update_layout(
        title="EEG Feature Heatmap Across Sleep Epochs",
        xaxis_title="Time (minutes)",
        yaxis_title="EEG feature",
        width=1200,
        height=500,
    )

    save_plot(
        fig,
        html_path="results/visualizations/feature_heatmap.html",
        png_path="results/visualizations/feature_heatmap.png",
    )

    fig.show()


def plot_smoothed_bandpower(df, epoch_seconds=30, smooth_window=10):
    """
    Plot each EEG band in its own subplot.

    This is better than putting all bands on one graph because delta power
    often dominates the scale and makes the smaller bands hard to see.
    """

    df = add_time_columns(df, epoch_seconds)
    plot_df = df.copy()

    for col in FEATURE_COLUMNS:
        plot_df[col] = (
            plot_df[col]
            .rolling(window=smooth_window, center=True, min_periods=1)
            .mean()
        )

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Delta", "Theta", "Alpha", "Beta"],
    )

    for row_number, col in enumerate(FEATURE_COLUMNS, start=1):
        fig.add_trace(
            go.Scatter(
                x=plot_df["time_minutes"],
                y=plot_df[col],
                mode="lines",
                name=col,
            ),
            row=row_number,
            col=1,
        )

    fig.update_layout(
        title=f"Smoothed Relative EEG Band Power Across Epochs",
        width=1200,
        height=900,
        showlegend=False,
    )

    fig.update_xaxes(title_text="Time (minutes)", row=4, col=1)
    fig.update_yaxes(title_text="Relative power")

    save_plot(
        fig,
        html_path="results/visualizations/smoothed_bandpower.html",
        png_path="results/visualizations/smoothed_bandpower.png",
    )

    fig.show()


def plot_raw_epoch_window(
    data,
    sfreq,
    epoch_seconds=30,
    start_epoch=0,
    num_epochs_to_show=10,
):
    """
    Plot a small window of the raw EEG signal with epoch boundaries.

    This helps you see what the raw signal looks like before it becomes
    band-power features.

    start_epoch:
        Which epoch to start plotting from.

    num_epochs_to_show:
        How many epochs to display.
    """

    samples_per_epoch = int(epoch_seconds * sfreq)

    start_sample = start_epoch * samples_per_epoch
    end_sample = start_sample + num_epochs_to_show * samples_per_epoch

    signal_chunk = data[start_sample:end_sample]

    time_seconds = np.arange(len(signal_chunk)) / sfreq
    time_minutes = time_seconds / 60
    time_minutes = time_minutes + (start_epoch * epoch_seconds / 60)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time_minutes,
            y=signal_chunk,
            mode="lines",
            name="Raw EEG",
        )
    )

    for i in range(num_epochs_to_show + 1):
        boundary_minute = ((start_epoch + i) * epoch_seconds) / 60
        fig.add_vline(
            x=boundary_minute,
            line_width=1,
            line_dash="dash",
        )

    fig.update_layout(
        title=f"Raw EEG Signal From Epoch {start_epoch} to Epoch {start_epoch + num_epochs_to_show}",
        xaxis_title="Time (minutes)",
        yaxis_title="EEG voltage",
        width=1200,
        height=500,
        hovermode="x unified",
    )

    save_plot(
        fig,
        html_path="results/visualizations/raw_epoch_window.html",
        png_path="results/visualizations/raw_epoch_window.png",
    )

    fig.show()
