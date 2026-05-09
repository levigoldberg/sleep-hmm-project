import numpy as np
from scipy.special import logsumexp
from constants import STATE_NAMES, EPSILON, MIN_VARIANCE


# -----------------------------
# Gaussian emission model
# -----------------------------


def gaussian_emission(x, means, variances):
    """
    Compute emission probabilities for one observation.

    Parameters:
        x:
            (F,) feature vector for one epoch

        means:
            (K, F) mean feature vector for each hidden state

        variances:
            (K, F) variance of each feature for each hidden state

    Returns:
        probs:
            (K,) array where probs[k] = P(x | state k)

    Notes:
        This assumes diagonal covariance, meaning each feature contributes
        independently to the emission probability.
    """

    variances = np.maximum(variances, MIN_VARIANCE)

    diff = x - means
    exponent = -0.5 * (diff**2) / variances
    coeff = 1.0 / np.sqrt(2 * np.pi * variances)

    probs = np.prod(coeff * np.exp(exponent), axis=1)

    return probs


# -----------------------------
# Forward algorithm
# -----------------------------


def forward_log(X, A, means, variances, pi):
    """
    Run the forward algorithm in log space.

    Parameters:
        X:
            (T, F) observation matrix

        A:
            (K, K) transition matrix

        means:
            (K, F) Gaussian mean vectors

        variances:
            (K, F) Gaussian variances

        pi:
            (K,) initial state probabilities

    Returns:
        log_alpha:
            (T, K) forward log-probabilities
    """

    T = X.shape[0]
    K = A.shape[0]

    log_alpha = np.zeros((T, K))
    log_A = np.log(A + EPSILON)

    log_emission = np.log(gaussian_emission(X[0], means, variances) + EPSILON)
    log_alpha[0] = np.log(pi + EPSILON) + log_emission

    for t in range(1, T):
        log_emission = np.log(gaussian_emission(X[t], means, variances) + EPSILON)

        for k in range(K):
            log_alpha[t, k] = log_emission[k] + logsumexp(
                log_alpha[t - 1] + log_A[:, k]
            )

    return log_alpha


# -----------------------------
# Backward algorithm
# -----------------------------


def backward_log(X, A, means, variances):
    """
    Run the backward algorithm in log space.

    Parameters:
        X:
            (T, F) observation matrix

        A:
            (K, K) transition matrix

        means:
            (K, F) Gaussian mean vectors

        variances:
            (K, F) Gaussian variances

    Returns:
        log_beta:
            (T, K) backward log-probabilities
    """

    T = X.shape[0]
    K = A.shape[0]

    log_beta = np.zeros((T, K))
    log_A = np.log(A + EPSILON)

    log_beta[T - 1] = 0.0

    for t in range(T - 2, -1, -1):
        log_emission_next = np.log(
            gaussian_emission(X[t + 1], means, variances) + EPSILON
        )

        for k in range(K):
            log_beta[t, k] = logsumexp(
                log_A[k, :] + log_emission_next + log_beta[t + 1]
            )

    return log_beta


# -----------------------------
# Xi: posterior transition probabilities
# -----------------------------


def compute_xi(X, A, means, variances, log_alpha, log_beta):
    """
    Compute xi, the posterior transition probabilities.

    Parameters:
        X:
            (T, F) array of observations

        A:
            (K, K) transition matrix where A[i, j] = P(state j at t+1 | state i at t)

        means:
            (K, F) mean feature vector for each state

        variances:
            (K, F) variance of each feature for each state

        log_alpha:
            (T, K) forward log-probabilities

        log_beta:
            (T, K) backward log-probabilities

    Returns:
        xi:
            (T - 1, K, K) array where
            xi[t, i, j] = P(state i at time t and state j at time t+1 | all observations)
    """

    T = X.shape[0]
    K = A.shape[0]

    xi = np.zeros((T - 1, K, K))
    log_A = np.log(A + EPSILON)

    for t in range(T - 1):
        emission_next = gaussian_emission(X[t + 1], means, variances)
        log_emission_next = np.log(emission_next + EPSILON)

        log_xi_t = (
            log_alpha[t, :, None]  # log P(reach state i at time t)
            + log_A  # log P(transition i -> j)
            + log_emission_next[None, :]  # log P(state j emits x[t+1])
            + log_beta[t + 1, None, :]  # log P(future observations from state j)
        )

        log_xi_t -= logsumexp(log_xi_t)
        xi[t] = np.exp(log_xi_t)

    return xi


# -----------------------------
# Forward-backward
# -----------------------------


def forward_backward(X, A, means, variances, pi):
    """
    Run forward-backward.

    Parameters:
        X:
            (T, F) observation matrix

        A:
            (K, K) transition matrix

        means:
            (K, F) Gaussian mean vectors

        variances:
            (K, F) Gaussian variances

        pi:
            (K,) initial state probabilities

    Returns:
        gamma:
            (T, K) posterior state probabilities

        xi:
            (T - 1, K, K) posterior transition probabilities

        log_likelihood:
            log probability of the full observation sequence
    """

    log_alpha = forward_log(X, A, means, variances, pi)
    log_beta = backward_log(X, A, means, variances)

    log_likelihood = logsumexp(log_alpha[-1])

    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    xi = compute_xi(X, A, means, variances, log_alpha, log_beta)

    return gamma, xi, log_likelihood


# -----------------------------
# Decoding helpers
# -----------------------------


def posterior_decode(gamma):
    """
    Choose the most likely state at each time step using gamma.

    Returns:
        states:
            (T,) array of predicted state indices
    """

    return np.argmax(gamma, axis=1)


def state_indices_to_names(states, state_names=STATE_NAMES):
    """
    Convert numeric state indices into readable state names.
    """

    return [state_names[state] for state in states]


def viterbi_decode(X, A, means, variances, pi):
    """
    Find the single most likely full sequence of hidden states.

    This is adapted from the XHMM-style Viterbi code:
    - dp_matrix stores the best path score ending in each state
    - backtrack_matrix stores which previous state gave that best score
    - the final path is recovered by walking backward

    Difference from XHMM:
    - each observation is a feature vector, not one number
    - emissions come from gaussian_emission()
    - log probabilities are used to avoid underflow
    """

    T = X.shape[0]  # number of epochs
    K = A.shape[0]  # number of hidden states

    # Convert probabilities to log probabilities.
    log_A = np.log(A + EPSILON)
    log_pi = np.log(pi + EPSILON)

    # dp_matrix[state, time] = best log probability of a path
    # that ends in this state at this time.
    dp_matrix = np.full((K, T), -np.inf, dtype="float64")

    # backtrack_matrix[state, time] = previous state that gave
    # the best path into this state.
    backtrack_matrix = np.full((K, T), 0, dtype=int)

    # Initialize the first epoch.
    first_emissions = gaussian_emission(X[0], means, variances)
    log_first_emissions = np.log(first_emissions + EPSILON)

    dp_matrix[:, 0] = log_pi + log_first_emissions

    # Fill the DP matrix from left to right.
    for t in range(1, T):
        current_emissions = gaussian_emission(X[t], means, variances)
        log_current_emissions = np.log(current_emissions + EPSILON)

        for state in range(K):
            best_val = -np.inf
            best_state = 0

            for prev_state in range(K):
                candidate = dp_matrix[prev_state, t - 1] + log_A[prev_state, state]

                if candidate > best_val:
                    best_val = candidate
                    best_state = prev_state

            dp_matrix[state, t] = best_val + log_current_emissions[state]
            backtrack_matrix[state, t] = best_state

    # Start from the best final state.
    hidden_states = np.zeros(T, dtype=int)
    current_state = int(np.argmax(dp_matrix[:, T - 1]))
    hidden_states[T - 1] = current_state

    # Walk backward through the backtrack matrix.
    for t in range(T - 1, 0, -1):
        previous_state = backtrack_matrix[current_state, t]
        hidden_states[t - 1] = previous_state
        current_state = previous_state

    return hidden_states
