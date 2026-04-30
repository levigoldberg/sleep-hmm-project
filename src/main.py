import pandas as pd
from feature_extraction import extract_features_by_participant
from training import baum_welch_training_shell


def main():
    participant_features, participant_ids, sequence_list = extract_features_by_participant(
        manifest_csv="data/participant_manifest.csv"
    )

    feature_arrays = [df.to_numpy() for df in participant_features.values()]
    training_sequences = feature_arrays if feature_arrays else sequence_list
    baum_welch_training_shell(training_sequences)

    output_rows = []
    for participant_id in participant_ids:
        df = participant_features[participant_id].copy()
        df["participant_id"] = participant_id
        df["epoch_idx"] = range(len(df))
        output_rows.append(df)
        df.to_csv(f"results/features_{participant_id}.csv", index=False)

    if output_rows:
        combined = pd.concat(output_rows, ignore_index=True)
        ordered_cols = ["participant_id", "epoch_idx"] + [
            c for c in combined.columns if c not in {"participant_id", "epoch_idx"}
        ]
        combined = combined[ordered_cols]
        combined.to_csv("results/features_all_participants.csv", index=False)


if __name__ == "__main__":
    main()
