import os

import mne
import numpy as np
import pandas as pd

from constants import (
    DATA_DIR,
    N_PARTICIPANTS,
    EPOCH_SECONDS,
    VALIDATION_CLASS_NAMES,
    SLEEP_EDF_STAGE_MAP,
    HMM_STATE_TO_VALIDATION_LABEL,
    INVALID_LABEL,
    MODEL_INITIAL_PROB_PATH,
    MODEL_TRANSITION_PATH,
    MODEL_MEANS_PATH,
    MODEL_VARIANCES_PATH,
    DECODING_METHOD,
    RESULTS_DIR,
)
from feature_extraction import discover_psg_files, extract_features_by_participant
from hmm_inference import forward_backward, posterior_decode, viterbi_decode


def record_key(path):
    """
    Get the shared ID that matches a PSG file with its Hypnogram file.

    Example:
        SC4001E0-PSG.edf       -> SC4001E
        SC4001EC-Hypnogram.edf -> SC4001E
    """

    filename = os.path.basename(path)
    stem = filename.split("-")[0]
    return stem[:-1]


def find_hypnograms():
    """
    Find all hypnogram files and store them by record key.
    """

    hypnograms = {}

    for root, _, files in os.walk(DATA_DIR):
        for filename in files:
            lower = filename.lower()

            if lower.endswith(".edf") and "hypnogram" in lower:
                path = os.path.join(root, filename)
                hypnograms[record_key(path)] = path
    return hypnograms


def load_model():
    """
    Load the trained HMM parameters saved by main.py.
    """

    return (
        np.load(MODEL_INITIAL_PROB_PATH),
        np.load(MODEL_TRANSITION_PATH),
        np.load(MODEL_MEANS_PATH),
        np.load(MODEL_VARIANCES_PATH),
    )


def load_true_labels(hypnogram_path, num_epochs):
    """
    Convert Sleep-EDF hypnogram annotations into one label per 30-second epoch.
    """

    annotations = mne.read_annotations(hypnogram_path)
    #ensures that list is the same size as the number of epochs before removing any invalid ones.
    y_true = np.full(num_epochs, INVALID_LABEL, dtype=int)

    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
    ):
        label = SLEEP_EDF_STAGE_MAP.get(description, INVALID_LABEL)

        # Skip movement time and unknown labels.
        if label == INVALID_LABEL:
            continue

        start_epoch = int(round(onset / EPOCH_SECONDS))
        num_labeled_epochs = int(round(duration / EPOCH_SECONDS))
        end_epoch = start_epoch + num_labeled_epochs

        # Prevent labels from going past the feature array.
        end_epoch = min(end_epoch, num_epochs)

        y_true[start_epoch:end_epoch] = label

    return y_true


def map_hmm_states(raw_states):
    """
    Convert raw HMM state numbers into manually assigned labels.
    """

    label_to_index = {
        label_name: i for i, label_name in enumerate(VALIDATION_CLASS_NAMES)
    }

    mapped_labels = []

    for state in raw_states:
        label_name = HMM_STATE_TO_VALIDATION_LABEL[int(state)]
        mapped_labels.append(label_to_index[label_name])

    return np.array(mapped_labels)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load the trained model that you already inspected manually.
    initial_prob, transition, means, variances = load_model()

    # Find PSG files and their matching expert hypnogram files.
    split_path = f"{RESULTS_DIR}/train_validation_split.csv"
    split_df = pd.read_csv(split_path)

    psg_paths = (
        split_df[split_df["split"] == "validation"]["psg_path"]
        .tolist()
)
    hypnograms = find_hypnograms()

    psg_paths_to_use = []
    hypnogram_paths_to_use = []

    for psg_path in psg_paths:
        key = record_key(psg_path)

        if key not in hypnograms:
            raise FileNotFoundError(f"No matching hypnogram found for {psg_path}")

        psg_paths_to_use.append(psg_path)
        hypnogram_paths_to_use.append(hypnograms[key])

    # Extract features in the same 30-second epoch format used for training.
    _, participant_ids, sequences = extract_features_by_participant(
        psg_paths=psg_paths_to_use
    )

    rows = []

    for participant_id, X, psg_path, hypnogram_path in zip(
        participant_ids,
        sequences,
        psg_paths_to_use,
        hypnogram_paths_to_use,
    ):
        # Get the most likely HMM state for each epoch.
        gamma, _, log_likelihood = forward_backward(
            X,
            transition,
            means,
            variances,
            initial_prob,
        )
        if DECODING_METHOD == "posterior":
            raw_states = posterior_decode(gamma)
        elif DECODING_METHOD == "viterbi":
            raw_states = viterbi_decode(X, transition, means, variances, initial_prob)
        else:
            raise ValueError("Invalid decoding method. Use 'posterior' or 'viterbi'.")

        # Convert raw HMM states into your manually chosen labels.
        y_pred = map_hmm_states(raw_states)

        # Load the expert labels from the paired hypnogram file.
        y_true = load_true_labels(hypnogram_path, num_epochs=len(X))

        # Only validate epochs with a usable expert label.
        valid_epochs = np.where(y_true != INVALID_LABEL)[0]

        for epoch_idx in valid_epochs:
            true_label = y_true[epoch_idx]
            pred_label = y_pred[epoch_idx]

            rows.append(
                {
                    "participant_id": participant_id,
                    "epoch_idx": int(epoch_idx),
                    "true_label": VALIDATION_CLASS_NAMES[true_label],
                    "pred_label": VALIDATION_CLASS_NAMES[pred_label],
                    "raw_hmm_state": int(raw_states[epoch_idx]),
                    "correct": true_label == pred_label,
                    "psg_path": psg_path,
                    "hypnogram_path": hypnogram_path,
                    "log_likelihood": log_likelihood,
                }
            )

    results = pd.DataFrame(rows)

    # Overall accuracy.
    accuracy = results["correct"].mean()

    # Accuracy per participant.
    participant_summary = (
        results.groupby("participant_id")["correct"]
        .agg(num_epochs="count", accuracy="mean")
        .reset_index()
    )

    # Confusion matrix.
    confusion = pd.crosstab(
        results["true_label"],
        results["pred_label"],
        rownames=["True"],
        colnames=["Predicted"],
        dropna=False,
    )

    # Save results.
    results.to_csv(f"{RESULTS_DIR}/validation_predictions.csv", index=False)
    participant_summary.to_csv(
        f"{RESULTS_DIR}/validation_summary_by_participant.csv",
        index=False,
    )
    confusion.to_csv(f"{RESULTS_DIR}/validation_confusion_matrix.csv")

    print("\nManual state mapping:")
    for state, label in HMM_STATE_TO_VALIDATION_LABEL.items():
        print(f"State {state} -> {label}")

    print("\nOverall accuracy:")
    print(round(accuracy, 3))

    print("\nAccuracy by participant:")
    print(participant_summary)

    print("\nConfusion matrix:")
    print(confusion)

    print("\nSaved validation results to results/")


if __name__ == "__main__":
    main()
