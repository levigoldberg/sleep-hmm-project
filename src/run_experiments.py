"""
run_experiments.py

Runs 6 HMM training + validation experiments automatically.
Each run changes one or two parameters while keeping everything else fixed.

Fixed across all runs:
    K                = 3 (Wake, NREM, REM)
    pi               = [0.90, 0.05, 0.05]
    DECODING_METHOD  = viterbi
    CHANNEL          = EEG Fpz-Cz
    WELCH_SEGMENT    = 4
    TRAIN_FRACTION   = 0.7

Varied:
    FEATURE_EXTRACTION_METHOD : relative | log
    TRANSITION_INIT           : informed | uniform | random

Results saved to timestamped files so re-running never overwrites
previous results.
"""

import os
import time
import csv
from datetime import datetime

import numpy as np
import pandas as pd

#project imports
from feature_extraction import discover_psg_files, extract_features_by_participant
from hmm_inference import forward_backward, viterbi_decode
from data_split import split_paths_by_participant
from training import m_step_update_multiple_sequences
from validation import load_true_labels, map_hmm_states, find_hypnograms, record_key
from constants import (
    DATA_DIR,
    RESULTS_DIR,
    K,
    VALIDATION_CLASS_NAMES,
    INVALID_LABEL,
    THRESHOLD,
    ITERATIONS,
)

#output paths 
EXPERIMENTS_DIR = os.path.join(RESULTS_DIR, "experiments")

#Timestamp added to filename so each run of this script creates a new file instead of overwriting previous results (ex:experiment_results_20260507_1423.txt)
_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
RESULTS_TXT = os.path.join(EXPERIMENTS_DIR, f"experiment_results_{_TIMESTAMP}.txt")
RESULTS_CSV = os.path.join(EXPERIMENTS_DIR, f"experiment_summary_{_TIMESTAMP}.csv")

#fixed parameters
# Always start very likely in Wake because recordings begin before sleep.
# Small nonzero values for other states avoid impossible states at t=0.
PI = np.array([0.90, 0.05, 0.05])

# Fixed seed for random transition matrix so results are reproducible
RANDOM_SEED = 42

#6 experiment combinations
EXPERIMENTS = [
    {"run": 1, "feature_method": "relative", "transition_init": "informed"},
    {"run": 2, "feature_method": "relative", "transition_init": "uniform"},
    {"run": 3, "feature_method": "relative", "transition_init": "random"},
    {"run": 4, "feature_method": "log",      "transition_init": "informed"},
    {"run": 5, "feature_method": "log",      "transition_init": "uniform"},
    {"run": 6, "feature_method": "log",      "transition_init": "random"},
]


#transition matrix builders 

def build_informed_transition():
    """
    Biologically motivated transition matrix for K=3 (Wake, NREM, REM).

    Based on AASM sleep staging guidelines:
    - Wake is persistent at recording start, leads into NREM as sleep begins
    - NREM is the most persistent stage (~75% of total sleep time)
    - REM is persistent but can return to Wake or cycle back through NREM

    Source: Patel et al. StatPearls NBK526132, training.py initialization
    """
    return np.array([
        [0.92, 0.07, 0.01],   # Wake  → mostly stays Wake
        [0.03, 0.94, 0.03],   # NREM  → highly persistent
        [0.08, 0.12, 0.80],   # REM   → persistent, can return to Wake/NREM
    ])


def build_uniform_transition():
    """
    Uniform transition matrix — every transition equally likely.
    No biological assumption. Used as a naive baseline to compare
    against informed and random initializations.
    """
    return np.full((K, K), 1.0 / K)


def build_random_transition(seed=RANDOM_SEED):
    """
    Random transition matrix using Dirichlet distribution.
    Fixed seed ensures the same random matrix every run for reproducibility.
    """
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(K), size=K)


def get_transition_matrix(init_type):
    if init_type == "informed":
        return build_informed_transition()
    elif init_type == "uniform":
        return build_uniform_transition()
    elif init_type == "random":
        return build_random_transition()
    else:
        raise ValueError(f"Unknown transition init type: {init_type}")


#emission initializer 

def get_initial_emissions(feature_method):
    """
    Initial Gaussian means and variances for K=3 (Wake, NREM, REM).
    These are biologically motivated starting points — Baum-Welch
    will refine them during training.

    Sources:
        Attar (Neurosciences 2022) — EEG band behavior per sleep stage
        NIH NBK526132 — EEG characteristics per sleep stage
    """
    if feature_method == "relative":
        # columns: delta_rel, theta_rel, alpha_rel, beta_rel (sum to ~1)
        means = np.array([
            [0.12, 0.15, 0.38, 0.35],   # Wake:  high alpha/beta
            [0.55, 0.30, 0.10, 0.05],   # NREM:  high delta/theta
            [0.18, 0.42, 0.15, 0.25],   # REM:   theta dominant, some beta
        ])
        variances = np.full((K, 4), 0.03)

    elif feature_method == "log":
        # z-scored log power — values centered around 0
        # positive = above participant average, negative = below
        means = np.array([
            [-0.4, -0.3,  0.9,  0.8],   # Wake:  high alpha/beta
            [ 0.9,  0.4, -0.5, -0.6],   # NREM:  high delta/theta
            [-0.2,  0.8, -0.1,  0.4],   # REM:   theta/mixed
        ])
        variances = np.full((K, 4), 1.0)

    else:
        raise ValueError(f"Unknown feature method: {feature_method}")

    return means, variances


#baum-welch training 

def run_baum_welch(feature_sequences, transition_init, feature_method):
    """
    Run Baum-Welch EM training with the given initialization.
    Returns trained parameters, log likelihood history, and the
    initial transition matrix saved before training updates it.
    """
    A = get_transition_matrix(transition_init)
    means, variances = get_initial_emissions(feature_method)
    initial_prob = PI.copy()
    initial_A = A.copy()   # save before Baum-Welch overwrites it
    log_likelihoods = []

    for iteration in range(ITERATIONS):
        gammas, xis = [], []
        total_log_likelihood = 0.0

        for X in feature_sequences:
            gamma, xi, log_likelihood = forward_backward(
                X, A, means, variances, initial_prob
            )
            gammas.append(gamma)
            xis.append(xi)
            total_log_likelihood += log_likelihood

        initial_prob, A, means, variances = m_step_update_multiple_sequences(
            feature_sequences, gammas, xis
        )

        log_likelihoods.append(total_log_likelihood)
        print(f"  Iteration {iteration + 1}, log likelihood: {total_log_likelihood:.2f}")

        # stop early if improvement falls below convergence threshold
        if iteration > 0:
            if abs(log_likelihoods[-1] - log_likelihoods[-2]) < THRESHOLD:
                print(f"  Converged at iteration {iteration + 1}")
                break

    return initial_prob, A, means, variances, log_likelihoods, initial_A


#validation

def run_validation(validation_sequences, participant_ids, hypnogram_paths,
                   initial_prob, A, means, variances):
    """
    Run viterbi decoding on validation set and compare to expert labels.
    Returns overall accuracy, per-class accuracy, and confusion matrix.
    """
    rows = []

    for X, participant_id, hypnogram_path in zip(
        validation_sequences, participant_ids, hypnogram_paths
    ):
        raw_states = viterbi_decode(X, A, means, variances, initial_prob)
        y_pred = map_hmm_states(raw_states)
        y_true = load_true_labels(hypnogram_path, num_epochs=len(X))
        valid_epochs = np.where(y_true != INVALID_LABEL)[0]

        for epoch_idx in valid_epochs:
            rows.append({
                "participant_id": participant_id,
                "true_label": VALIDATION_CLASS_NAMES[y_true[epoch_idx]],
                "pred_label": VALIDATION_CLASS_NAMES[y_pred[epoch_idx]],
                "correct": y_true[epoch_idx] == y_pred[epoch_idx],
            })

    results = pd.DataFrame(rows)
    overall_accuracy = results["correct"].mean()

    per_class = {}
    for label in VALIDATION_CLASS_NAMES:
        subset = results[results["true_label"] == label]
        per_class[label] = subset["correct"].mean() if len(subset) > 0 else float("nan")

    confusion = pd.crosstab(
        results["true_label"],
        results["pred_label"],
        rownames=["True"],
        colnames=["Predicted"],
        dropna=False,
    )

    return overall_accuracy, per_class, confusion


#formatting helpers

def format_matrix(matrix, row_labels=None):
    """Format a numpy matrix as a readable aligned string."""
    lines = []
    for i, row in enumerate(matrix):
        row_str = "  ".join(f"{v:.3f}" for v in row)
        label = f"State {i}" if row_labels is None else row_labels[i]
        lines.append(f"  {label:>8} ->  {row_str}")
    return "\n".join(lines)


def format_confusion(confusion):
    """Format confusion matrix as a readable string."""
    return confusion.to_string()


#main

def main():
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    import constants

    print("Discovering PSG files...")
    all_psg_paths = discover_psg_files(DATA_DIR)
    train_paths, validation_paths = split_paths_by_participant(all_psg_paths)

    # save split so validation always uses the same held-out participants
    split_df = pd.DataFrame(
        [{"split": "train", "psg_path": p} for p in train_paths]
        + [{"split": "validation", "psg_path": p} for p in validation_paths]
    )
    split_df.to_csv(os.path.join(RESULTS_DIR, "train_validation_split.csv"), index=False)

    # find matching hypnogram for each validation participant
    hypnograms = find_hypnograms()
    validation_hypnogram_paths = []
    for p in validation_paths:
        key = record_key(p)
        if key not in hypnograms:
            raise FileNotFoundError(f"No hypnogram found for {p}")
        validation_hypnogram_paths.append(hypnograms[key])

    csv_rows = []

    with open(RESULTS_TXT, "w") as txt:

        txt.write("=" * 65 + "\n")
        txt.write("SLEEP HMM EXPERIMENT RESULTS\n")
        txt.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        txt.write("=" * 65 + "\n\n")

        txt.write("FIXED PARAMETERS\n")
        txt.write("-" * 40 + "\n")
        txt.write(f"  K                  : {K}\n")
        txt.write(f"  pi                 : {PI.tolist()}\n")
        txt.write(f"  DECODING_METHOD    : viterbi\n")
        txt.write(f"  CHANNEL            : EEG Fpz-Cz\n")
        txt.write(f"  WELCH_SEGMENT      : 4 seconds\n")
        txt.write(f"  TRAIN_FRACTION     : 0.7\n")
        txt.write(f"  ITERATIONS         : {ITERATIONS}\n")
        txt.write(f"  THRESHOLD          : {THRESHOLD}\n\n")

        for exp in EXPERIMENTS:
            run_num         = exp["run"]
            feature_method  = exp["feature_method"]
            transition_init = exp["transition_init"]

            label = f"RUN {run_num} — {feature_method} + {transition_init}"
            print(f"\n{'=' * 55}")
            print(f"Starting {label}")
            print("=" * 55)

            # patch constants so feature_extraction uses the right method
            # for this run without needing to edit constants.py manually
            constants.FEATURE_EXTRACTION_METHOD = feature_method

            print("  Extracting training features...")
            _, _, training_sequences = extract_features_by_participant(
                psg_paths=train_paths
            )

            print("  Extracting validation features...")
            _, val_participant_ids, validation_sequences = extract_features_by_participant(
                psg_paths=validation_paths
            )

            print("  Training...")
            t_start = time.time()

            initial_prob, A_final, means_final, variances_final, \
                log_likelihoods, A_initial = run_baum_welch(
                    training_sequences, transition_init, feature_method
                )

            t_end = time.time()
            runtime = t_end - t_start

            print("  Validating...")
            overall_acc, per_class_acc, confusion = run_validation(
                validation_sequences,
                val_participant_ids,
                validation_hypnogram_paths,
                initial_prob,
                A_final,
                means_final,
                variances_final,
            )

            #write to txt 
            txt.write("=" * 65 + "\n")
            txt.write(f"{label}\n")
            txt.write("=" * 65 + "\n\n")

            txt.write("Parameters:\n")
            txt.write(f"  FEATURE_METHOD     : {feature_method}\n")
            txt.write(f"  TRANSITION_INIT    : {transition_init}\n")
            txt.write(f"  Runtime            : {runtime:.1f} seconds\n\n")

            txt.write("Initial Transition Matrix:\n")
            txt.write(format_matrix(A_initial, row_labels=["Wake", "NREM", "REM"]) + "\n\n")

            txt.write("Final Learned Transition Matrix:\n")
            txt.write(format_matrix(A_final, row_labels=["Wake", "NREM", "REM"]) + "\n\n")

            txt.write("Final Gaussian Means (columns: delta, theta, alpha, beta):\n")
            txt.write(format_matrix(means_final, row_labels=["Wake", "NREM", "REM"]) + "\n\n")

            txt.write("Log Likelihood History:\n")
            for i, ll in enumerate(log_likelihoods):
                txt.write(f"  Iteration {i + 1:>2}: {ll:.2f}\n")
            txt.write("\n")

            txt.write("Confusion Matrix:\n")
            txt.write(format_confusion(confusion) + "\n\n")

            txt.write("Accuracy:\n")
            txt.write(f"  Overall            : {overall_acc:.3f}\n")
            for label_name, acc in per_class_acc.items():
                txt.write(f"  {label_name:<18} : {acc:.3f}\n")
            txt.write("\n")

            #collect csv row
            row = {
                "Run": run_num,
                "Feature_Method": feature_method,
                "Transition_Init": transition_init,
                "Runtime_seconds": round(runtime, 1),
                "Overall_Accuracy": round(overall_acc, 3),
            }
            for label_name, acc in per_class_acc.items():
                row[f"{label_name}_Accuracy"] = round(acc, 3)
            csv_rows.append(row)

            print(f"  Overall accuracy: {overall_acc:.3f}")

        txt.write("=" * 65 + "\n")
        txt.write("END OF EXPERIMENTS\n")
        txt.write("=" * 65 + "\n")

    #write summary csv
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"\nDone! Results saved to:")
    print(f"  {RESULTS_TXT}")
    print(f"  {RESULTS_CSV}")


if __name__ == "__main__":
    main()