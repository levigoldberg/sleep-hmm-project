from feature_extraction import extract_features
from visualizations import (
    plot_feature_heatmap,
    plot_smoothed_bandpower,
    plot_raw_epoch_window,
)


def main():
    
    df, data, sfreq = extract_features()


if __name__ == "__main__":
    main()
