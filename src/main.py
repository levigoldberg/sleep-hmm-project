import pandas as pd
from feature_extraction import discover_psg_files, extract_features_by_participant
from training import baum_welch_training_shell
from constants import (
    STATE_NAMES,
    DATA_DIR,
    MODEL_INITIAL_PROB_PATH,
    MODEL_TRANSITION_PATH,
    MODEL_MEANS_PATH,
    MODEL_VARIANCES_PATH,
    RESULTS_DIR
)
import os
import numpy as np
from data_split import split_paths_by_participant

def main():
    all_psg_paths = discover_psg_files(DATA_DIR)
    # split the data into training and testing sets
    train_paths, validation_paths = split_paths_by_participant(all_psg_paths)

    print("\nTraining participants:")
    for path in train_paths:
        print(path)

    print("\nTesting participants:")
    for path in validation_paths:
        print(path)
        
    # Save the split so validation.py uses the same held-out participants.
    split_df = pd.DataFrame(
        [{"split": "train", "psg_path": path} for path in train_paths]
        + [{"split": "validation", "psg_path": path} for path in validation_paths]
    )

    split_df.to_csv(f"{RESULTS_DIR}/train_validation_split.csv", index=False)
    participant_features, participant_ids, training_sequences = (
        extract_features_by_participant(psg_paths=train_paths)
    )

    initial_prob, Transition, means, variances, log_likelihoods = (
        baum_welch_training_shell(training_sequences)
    )

    # Save the trained model
    os.makedirs("results", exist_ok=True)
    np.save(MODEL_INITIAL_PROB_PATH, initial_prob)
    np.save(MODEL_TRANSITION_PATH, Transition)
    np.save(MODEL_MEANS_PATH, means)
    np.save(MODEL_VARIANCES_PATH, variances)

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

    # Save extracted training features for inspection.
    output_rows = []

    for participant_id in participant_ids:
        df = participant_features[participant_id].copy()
        df["participant_id"] = participant_id
        df["epoch_idx"] = range(len(df))
        output_rows.append(df)
        df.to_csv(f"{RESULTS_DIR}/features_{participant_id}.csv", index=False)

    if output_rows:
        combined = pd.concat(output_rows, ignore_index=True)
        ordered_cols = ["participant_id", "epoch_idx"] + [
            c for c in combined.columns if c not in {"participant_id", "epoch_idx"}
        ]
        combined = combined[ordered_cols]
        combined.to_csv(f"{RESULTS_DIR}/features_training_participants.csv", index=False)


if __name__ == "__main__":
    main()
