import numpy as np

from hmm_inference import (
    NUM_FEATURES,
    NUM_STATES,
    STATE_NAMES,
    forward_backward,
    posterior_decode,
    state_indices_to_names,
)


# -----------------------------
# Adjustable fake-data constants
# -----------------------------

FAKE_NUM_EPOCHS = 200
FAKE_RANDOM_SEED = 42
FAKE_VARIANCE_VALUE = 0.02
FAKE_INITIAL_STATE = 0  # Wake


# Fake mean feature vectors for each sleep state.
# Each row is one hidden state.
# Each column is one feature:
# [delta_rel, theta_rel, alpha_rel, beta_rel]
FAKE_MEANS = np.array(
    [
        [0.10, 0.15, 0.40, 0.35],  # Wake
        [0.20, 0.45, 0.20, 0.15],  # N1
        [0.35, 0.35, 0.15, 0.15],  # N2
        [0.65, 0.20, 0.10, 0.05],  # N3
        [0.20, 0.40, 0.15, 0.25],  # REM
    ]
)


# Fake diagonal variances for each state and feature.
# Shape: (K, F)
FAKE_VARIANCES = np.full(
    (NUM_STATES, NUM_FEATURES),
    FAKE_VARIANCE_VALUE,
)


# Fake transition matrix used only to generate simulated data.
# A[i, j] = probability of moving from state i to state j.
FAKE_TRANSITION_MATRIX = np.array(
    [
        [0.80, 0.15, 0.03, 0.01, 0.01],  # Wake
        [0.05, 0.75, 0.15, 0.03, 0.02],  # N1
        [0.02, 0.05, 0.75, 0.15, 0.03],  # N2
        [0.02, 0.03, 0.10, 0.80, 0.05],  # N3
        [0.10, 0.05, 0.05, 0.05, 0.75],  # REM
    ]
)


# -----------------------------
# Fake data generation
# -----------------------------


def generate_fake_data(
    num_epochs=FAKE_NUM_EPOCHS,
    seed=FAKE_RANDOM_SEED,
    transition_matrix=FAKE_TRANSITION_MATRIX,
    means=FAKE_MEANS,
    variances=FAKE_VARIANCES,
):
    """
    Generate fake HMM observations for testing.

    This is only for debugging the HMM math.
    Real project data should come from feature_extraction.py.

    Returns:
        X:
            (T, F) fake observation matrix

        true_states:
            (T,) fake true hidden state sequence
    """

    rng = np.random.default_rng(seed)

    true_states = np.zeros(num_epochs, dtype=int)
    true_states[0] = FAKE_INITIAL_STATE

    for t in range(1, num_epochs):
        previous_state = true_states[t - 1]
        true_states[t] = rng.choice(
            NUM_STATES,
            p=transition_matrix[previous_state],
        )

    X = np.zeros((num_epochs, NUM_FEATURES))

    for t in range(num_epochs):
        state = true_states[t]

        X[t] = rng.normal(
            means[state],
            np.sqrt(variances[state]),
        )

        # Keep fake relative powers nonnegative
        X[t] = np.clip(X[t], 0, None)

        # Normalize so each fake epoch sums to 1, like relative band power
        X[t] /= X[t].sum()

    return X, true_states


# -----------------------------
# Test runner
# -----------------------------


def main():
    """
    Run a fake-data sanity check.

    This confirms that:
    - fake data generation works
    - forward_backward returns gamma, xi, and log likelihood
    - gamma rows sum to 1
    - xi transition matrices sum to 1
    """

    X, true_states = generate_fake_data()

    # Initial state probabilities.
    # For this fake test, we assume the recording starts in Wake.
    pi = np.zeros(NUM_STATES)
    pi[FAKE_INITIAL_STATE] = 1.0

    # Initial transition matrix for inference.
    # This is intentionally uniform to test whether the inference code runs.
    A_init = np.full(
        (NUM_STATES, NUM_STATES),
        1 / NUM_STATES,
    )

    gamma, xi, log_likelihood = forward_backward(
        X,
        A_init,
        FAKE_MEANS,
        FAKE_VARIANCES,
        pi,
    )

    predicted_states = posterior_decode(gamma)

    print("X shape:", X.shape)
    print("gamma shape:", gamma.shape)
    print("xi shape:", xi.shape)
    print("log_likelihood:", log_likelihood)

    print("gamma[0] sums to:", gamma[0].sum())
    print("xi[0] sums to:", xi[0].sum())

    print("First 5 predicted:", state_indices_to_names(predicted_states[:5]))
    print("First 5 true:     ", state_indices_to_names(true_states[:5]))


if __name__ == "__main__":
    main()
