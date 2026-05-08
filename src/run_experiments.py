"""
run_experiments.py

Runs a small set of HMM experiments.

This file should only control experiment settings and save results.
Training logic stays in training.py.
Feature extraction logic stays in feature_extraction.py.
Validation logic stays in validation.py.
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from constants import DATA_DIR, RESULTS_DIR, VALIDATION_CLASS_NAMES
from data_split import split_paths_by_participant
from feature_extraction import discover_psg_files, extract_features_by_participant
from training import baum_welch_training_shell
from validation import validate_model
import validation

# One dictionary = one experiment.
EXPERIMENTS = [
    {
        "feature_method": "relative",
        "transition_init": "informed",
        "emission_init": "informed",
    },
    {
        "feature_method": "relative",
        "transition_init": "uniform",
        "emission_init": "informed",
    },
    {
        "feature_method": "relative",
        "transition_init": "random",
        "emission_init": "informed",
    },
    {
        "feature_method": "log relative",
        "transition_init": "informed",
        "emission_init": "informed",
    },
    {
        "feature_method": "log relative",
        "transition_init": "uniform",
        "emission_init": "informed",
    },
    {
        "feature_method": "log relative",
        "transition_init": "random",
        "emission_init": "informed",
    },
    {
        "feature_method": "relative",
        "transition_init": "informed",
        "emission_init": "random",
    },
    {
        "feature_method": "log relative",
        "transition_init": "informed",
        "emission_init": "random",
    },
]


def safe_name(text):
    """Make names safe for folders and files."""
    return text.replace(" ", "_")


def make_experiment_name(run_number, exp):
    """Create a readable name for the experiment folder."""
    feature = safe_name(exp["feature_method"])
    transition = safe_name(exp["transition_init"])
    emission = safe_name(exp["emission_init"])

    return f"run_{run_number:02d}_{feature}_{transition}_{emission}"


def save_matrix(path, matrix, row_labels=None, col_labels=None):
    """Save a numpy matrix as a CSV."""
    df = pd.DataFrame(matrix)

    if row_labels is not None:
        df.index = row_labels

    if col_labels is not None:
        df.columns = col_labels

    df.to_csv(path)


def save_participant_split(path, train_paths, validation_paths):
    """Save which participants were used for training and validation."""
    rows = []

    for psg_path in train_paths:
        rows.append({"split": "train", "psg_path": psg_path})

    for psg_path in validation_paths:
        rows.append({"split": "validation", "psg_path": psg_path})

    pd.DataFrame(rows).to_csv(path, index=False)


def infer_state_mapping_from_means(means):
    """
    Automatically map HMM states to Wake, NREM, and REM so we dont
    have to interrupt before every experiment to manually map and assign.
    
    - NREM: strongest delta activity
    - Wake: strongest alpha/beta activity among remaining states
    - REM: remaining low-delta mixed-frequency state

    Feature order:
        0 = delta
        1 = theta
        2 = alpha
        3 = beta
    """

    delta_col = 0
    alpha_col = 2
    beta_col = 3

    # NREM usually has the strongest slow-wave delta activity.
    nrem_state = int(np.argmax(means[:, delta_col]))

    remaining_states = [s for s in range(len(means)) if s != nrem_state]

    # Wake usually has more alpha/beta activity than REM or NREM.
    wake_state = max(
        remaining_states,
        key=lambda s: means[s, alpha_col] + means[s, beta_col],
    )

    # The remaining state is treated as REM.
    rem_state = [s for s in remaining_states if s != wake_state][0]

    return {
        wake_state: "Wake",
        nrem_state: "NREM",
        rem_state: "REM",
    }


def run_single_experiment(
    run_number, exp, train_paths, validation_paths, base_output_dir
):
    """Run one experiment and save its results."""

    experiment_name = make_experiment_name(run_number, exp)
    output_dir = os.path.join(base_output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    feature_method = exp["feature_method"]
    transition_init = exp["transition_init"]
    emission_init = exp["emission_init"]

    print("\n" + "=" * 70)
    print(f"Starting {experiment_name}")
    print("=" * 70)

    start_time = time.perf_counter()

    # Save the exact train and validation files used.
    save_participant_split(
        os.path.join(output_dir, "participant_split.csv"),
        train_paths,
        validation_paths,
    )

    # Extract features for training.
    _, train_participant_ids, training_sequences = extract_features_by_participant(
        psg_paths=train_paths,
        feature_method=feature_method,
    )

    # Extract features for validation.
    _, validation_participant_ids, validation_sequences = (
        extract_features_by_participant(
            psg_paths=validation_paths,
            feature_method=feature_method,
        )
    )

    # Train the model.
    model = baum_welch_training_shell(
        training_sequences,
        transition_init=transition_init,
        emission_init=emission_init,
        feature_method=feature_method,
        return_initial_params=True,
    )

    # Automatically map learned HMM states before validation.
    state_mapping = infer_state_mapping_from_means(model["means"])
    validation.HMM_STATE_TO_VALIDATION_LABEL = state_mapping

    # Validate the trained model.
    metrics = validate_model(
        validation_paths=validation_paths,
        initial_prob=model["initial_prob"],
        transition=model["transition"],
        means=model["means"],
        variances=model["variances"],
        feature_method=feature_method,
        validation_sequences=validation_sequences,
        validation_participant_ids=validation_participant_ids,
    )
    pd.DataFrame(
        [{"state": state, "label": label} for state, label in state_mapping.items()]
    ).to_csv(
        os.path.join(output_dir, "state_mapping.csv"),
        index=False,
    )

    runtime_seconds = time.perf_counter() - start_time

    # Save model inputs and outputs.
    feature_names = ["delta", "theta", "alpha", "beta"]
    state_names = [f"State {i}" for i in range(len(model["means"]))]

    save_matrix(
        os.path.join(output_dir, "input_transition_matrix.csv"),
        model["input_transition"],
        row_labels=state_names,
        col_labels=state_names,
    )

    save_matrix(
        os.path.join(output_dir, "output_transition_matrix.csv"),
        model["transition"],
        row_labels=state_names,
        col_labels=state_names,
    )

    save_matrix(
        os.path.join(output_dir, "input_gaussian_means.csv"),
        model["input_means"],
        row_labels=state_names,
        col_labels=state_names,
    )

    save_matrix(
        os.path.join(output_dir, "output_gaussian_means.csv"),
        model["means"],
        row_labels=state_names,
        col_labels=state_names,
    )

    save_matrix(
        os.path.join(output_dir, "input_gaussian_variances.csv"),
        model["input_variances"],
        row_labels=state_names,
        col_labels=state_names,
    )

    save_matrix(
        os.path.join(output_dir, "output_gaussian_variances.csv"),
        model["variances"],
        row_labels=state_names,
        col_labels=state_names,
    )

    # Save validation outputs.
    metrics["confusion_matrix"].to_csv(os.path.join(output_dir, "confusion_matrix.csv"))

    pd.DataFrame(
        {
            "iteration": range(1, len(model["log_likelihoods"]) + 1),
            "log_likelihood": model["log_likelihoods"],
        }
    ).to_csv(
        os.path.join(output_dir, "log_likelihoods.csv"),
        index=False,
    )

    # Create one row for the master summary CSV.
    summary_row = {
        "run": run_number,
        "experiment_name": experiment_name,
        "feature_method": feature_method,
        "transition_init": transition_init,
        "emission_init": emission_init,
        "num_train_participants": len(train_participant_ids),
        "num_validation_participants": len(validation_participant_ids),
        "num_train_epochs": sum(len(seq) for seq in training_sequences),
        "num_validation_epochs": sum(len(seq) for seq in validation_sequences),
        "overall_accuracy": metrics["overall_accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "num_iterations": len(model["log_likelihoods"]),
        "final_log_likelihood": model["log_likelihoods"][-1],
        "runtime_seconds": runtime_seconds,
        "output_dir": output_dir,
        "state_0_label": state_mapping[0],
        "state_1_label": state_mapping[1],
        "state_2_label": state_mapping[2],
    }

    # Add one accuracy column per class.
    for class_name, accuracy in metrics["per_class_accuracy"].items():
        summary_row[f"{class_name}_accuracy"] = accuracy

    print(f"Finished {experiment_name}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.3f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"Runtime: {runtime_seconds:.1f} seconds")

    return summary_row


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_output_dir = os.path.join(RESULTS_DIR, "experiments", timestamp)
    os.makedirs(base_output_dir, exist_ok=True)

    # Use the same split for every experiment.
    all_psg_paths = discover_psg_files(DATA_DIR)
    train_paths, validation_paths = split_paths_by_participant(all_psg_paths)

    summary_rows = []

    for run_number, exp in enumerate(EXPERIMENTS, start=1):
        row = run_single_experiment(
            run_number,
            exp,
            train_paths,
            validation_paths,
            base_output_dir,
        )
        summary_rows.append(row)

    summary_path = os.path.join(base_output_dir, "experiment_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("\nAll experiments complete.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
