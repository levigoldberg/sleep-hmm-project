from feature_extraction import extract_features
from visualizations import (
    plot_feature_heatmap,
    plot_smoothed_bandpower,
    plot_raw_epoch_window,
)


def main():
    df, data, sfreq = extract_features()

    plot_feature_heatmap(df, epoch_seconds=30, smooth_window=5)

    plot_smoothed_bandpower(df, epoch_seconds=30, smooth_window=10)

    plot_raw_epoch_window(
        data=data,
        sfreq=sfreq,
        epoch_seconds=30,
        start_epoch=0,
        num_epochs_to_show=10,
    )


if __name__ == "__main__":
    main()
