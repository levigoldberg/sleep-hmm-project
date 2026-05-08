import os

import mne
import numpy as np
import pandas as pd

from constants import (
    DATA_DIR,
    EPOCH_SECONDS,
    VALIDATION_CLASS_NAMES,
    SLEEP_EDF_STAGE_MAP,
    HMM_STATE_TO_VALIDATION_LABEL,
    INVALID_LABEL,
    DECODING_METHOD,
)

from feature_extraction import extract_features_by_participant
from hmm_inference import forward_backward, posterior_decode, viterbi_decode


def record_key(path):
    """
    Get the shared ID that matches a PSG file with its hypnogram file.

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


def load_true_labels(hypnogram_path, num_epochs):
    """
    Convert Sleep-EDF annotations into one label per 30-second epoch.
    """

    annotations = mne.read_annotations(hypnogram_path)

    # Start with all epochs marked invalid.
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

        # Avoid going past the feature sequence length.
        end_epoch = min(end_epoch, num_epochs)

        y_true[start_epoch:end_epoch] = label

    return y_true


def map_hmm_states(raw_states):
    """
    Convert raw HMM state numbers into validation label numbers.
    """

    label_to_index = {
        label_name: i for i, label_name in enumerate(VALIDATION_CLASS_NAMES)
    }

    mapped_labels = []

    for state in raw_states:
        label_name = HMM_STATE_TO_VALIDATION_LABEL[int(state)]
        mapped_labels.append(label_to_index[label_name])

    return np.array(mapped_labels)


def validate_model(
    validation_paths,
    initial_prob,
    transition,
    means,
    variances,
    feature_method=None,
    validation_sequences=None,
    validation_participant_ids=None,
):
    """
    Validate one trained model on the validation participants.

    Returns:
        overall accuracy
        balanced accuracy
        per-class accuracy
        confusion matrix
    """

    hypnograms = find_hypnograms()

    psg_paths_to_use = []
    hypnogram_paths_to_use = []

    # Match each PSG file to its hypnogram file.
    for psg_path in validation_paths:
        key = record_key(psg_path)

        if key not in hypnograms:
            raise FileNotFoundError(f"No matching hypnogram found for {psg_path}")

        psg_paths_to_use.append(psg_path)
        hypnogram_paths_to_use.append(hypnograms[key])

    # Use pre-extracted validation features if run_experiments.py passes them in.
    if validation_sequences is None or validation_participant_ids is None:
        _, validation_participant_ids, validation_sequences = (
            extract_features_by_participant(
                psg_paths=psg_paths_to_use,
                feature_method=feature_method,
            )
        )

    rows = []

    for participant_id, X, hypnogram_path in zip(
        validation_participant_ids,
        validation_sequences,
        hypnogram_paths_to_use,
    ):
        # Forward-backward is needed for posterior decoding.
        gamma, _, _ = forward_backward(
            X,
            transition,
            means,
            variances,
            initial_prob,
        )

        # Decode HMM states.
        if DECODING_METHOD == "posterior":
            raw_states = posterior_decode(gamma)
        elif DECODING_METHOD == "viterbi":
            raw_states = viterbi_decode(X, transition, means, variances, initial_prob)
        else:
            raise ValueError("DECODING_METHOD must be 'posterior' or 'viterbi'.")

        # Convert raw HMM states into Wake, NREM, REM.
        y_pred = map_hmm_states(raw_states)

        # Load expert labels.
        y_true = load_true_labels(hypnogram_path, num_epochs=len(X))

        # Only score epochs with valid expert labels.
        valid_epochs = np.where(y_true != INVALID_LABEL)[0]

        for epoch_idx in valid_epochs:
            true_label = y_true[epoch_idx]
            pred_label = y_pred[epoch_idx]

            rows.append(
                {
                    "participant_id": participant_id,
                    "true_label": VALIDATION_CLASS_NAMES[true_label],
                    "pred_label": VALIDATION_CLASS_NAMES[pred_label],
                    "correct": true_label == pred_label,
                }
            )

    results = pd.DataFrame(rows)

    if results.empty:
        raise ValueError("No valid validation epochs found.")

    # Overall accuracy.
    overall_accuracy = results["correct"].mean()

    # Per-class accuracy.
    per_class_accuracy = {}

    for label in VALIDATION_CLASS_NAMES:
        subset = results[results["true_label"] == label]

        if len(subset) == 0:
            per_class_accuracy[label] = np.nan
        else:
            per_class_accuracy[label] = subset["correct"].mean()

    # Balanced accuracy averages the class accuracies.
    balanced_accuracy = np.nanmean(list(per_class_accuracy.values()))

    # Confusion matrix.
    confusion_matrix = pd.crosstab(
        results["true_label"],
        results["pred_label"],
        rownames=["True"],
        colnames=["Predicted"],
        dropna=False,
    )

    # Force all labels to appear as rows and columns.
    confusion_matrix = confusion_matrix.reindex(
        index=VALIDATION_CLASS_NAMES,
        columns=VALIDATION_CLASS_NAMES,
        fill_value=0,
    )

    return {
        "overall_accuracy": overall_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion_matrix,
    }
