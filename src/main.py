import pandas as pd
from feature_extraction import discover_psg_files, extract_features_by_participant
from training import baum_welch_training_shell
from constants import N_PARTICIPANTS, STATE_NAMES


def main():
    psg_paths = discover_psg_files("data/sleep-cassette")
    psg_paths = psg_paths[:N_PARTICIPANTS]

    participant_features, participant_ids, sequence_list = (
        extract_features_by_participant(psg_paths=psg_paths)
    )
    feature_arrays = [df.to_numpy() for df in participant_features.values()]
    training_sequences = feature_arrays if feature_arrays else sequence_list
    initial_prob, Transition, means, variances, log_likelihoods = (
        baum_welch_training_shell(training_sequences)
    )
    print("\nFinal Transition Matrix:")
    print("Rows = from state, Columns = to state\n")

    for i, row in enumerate(Transition):
        row_str = "  ".join(f"{p:.3f}" for p in row)
        print(f"{STATE_NAMES[i]:>5} -> {row_str}")
    print("\nFinal Gaussian Means:")
    print("Columns = delta_rel, theta_rel, alpha_rel, beta_rel\n")

    for i, row in enumerate(means):
        row_str = "  ".join(f"{x:.3f}" for x in row)
        print(f"{STATE_NAMES[i]:>5} -> {row_str}")  
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
