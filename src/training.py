import numpy as np
from hmm_inference import forward_backward
from constants import (
    K,
    D,
    THRESHOLD,
    ITERATIONS,
    FEATURE_EXTRACTION_METHOD,
    SPLIT_RANDOM_SEED,
)


def initialize_training_params(
    transition_init="informed",
    emission_init="informed",
    feature_method=None,
):
    """
    Initialize the HMM before BW training.
    """

    if K != 3:
        raise ValueError(
            "This biological initialization is currently written for K = 3."
        )
    if feature_method is None:
        feature_method = FEATURE_EXTRACTION_METHOD
    # Recordings usually begin while the participant is awake.
    # Small nonzero values avoid impossible states.
    initial_prob = np.array([0.90, 0.09, 0.01])
    initial_prob = initial_prob / initial_prob.sum()

    # Rows = current state, columns = next state.
    #
    # State 0 = Wake
    # State 1 = NREM
    # State 2 = REM
    #
    # Self transitions should be high.
    if transition_init == "informed":
        Transition = np.array(
            Transition=np.array(
                [
                    [0.90, 0.05, 0.05],
                    [0.05, 0.90, 0.05],
                    [0.05, 0.05, 0.90],
                ]
            )
        )

    elif transition_init == "uniform":
        Transition = np.full((K, K), 1.0 / K)

    elif transition_init == "random":
        rng = np.random.default_rng(SPLIT_RANDOM_SEED)
        Transition = rng.dirichlet(np.ones(K), size=K)

    else:
        raise ValueError("transition_init must be 'informed', 'uniform', or 'random'.")

    # Choose starting Gaussian means and variances.
    if emission_init == "informed":
        if feature_method == "relative":
            means = np.array(
                means=np.array(
                    [
                        [0.15, 0.20, 0.35, 0.30],  # Wake: higher alpha/beta
                        [0.45, 0.30, 0.15, 0.10],  # NREM: higher delta/theta
                        [0.15, 0.35, 0.25, 0.25],  # REM: mixed frequency, lower delta
                    ]
                )
            )
            variances = np.full((K, D), 0.05)
        elif feature_method == "log relative":
            unlogged_means = np.array(
                relative_means=np.array(
                    [
                        [0.15, 0.20, 0.35, 0.30],
                        [0.45, 0.30, 0.15, 0.10],
                        [0.15, 0.35, 0.25, 0.25],
                    ]
                )
            )
            means = np.log10(unlogged_means+ 1e-12)
            variances = np.full((K, D), 0.75)

        else:
            raise ValueError("feature_method must be 'relative' or 'log relative'.")

    elif emission_init == "random":
        rng = np.random.default_rng(SPLIT_RANDOM_SEED)

        if feature_method == "relative":
            means = rng.dirichlet(np.ones(D), size=K)
            variances = np.full((K, D), 0.03)

        elif feature_method == "log relative":
            relative_means = rng.dirichlet(np.ones(D), size=K)
            means = np.log10(relative_means + 1e-12)
            variances = np.full((K, D), 0.5)

        else:
            raise ValueError("feature_method must be 'relative' or 'log relative'.")

    else:
        raise ValueError("emission_init must be 'informed' or 'random'.")

    return initial_prob, Transition, means, variances


def m_step_update(Features, gamma, xi):
    """
    Run the M step: count and update the intital probabilities, transition matrix, means and variances
    Input: Features, gamma (probability that time t is in state k) and xi (probability of transitioning from state i to state j between time t and t+1)
    Output: initial_prob, Transition, means, variances
    """
    T = len(Features)  # num of epochs

    # update starting probabilities
    initial_prob = []

    for k in range(K):
        initial_prob.append(gamma[0][k])

    initial_prob = np.array(initial_prob)

    # make a new transition matrix
    Transition = []

    for i in range(K):
        row = []

        # count how many times we leave state i
        denominator = 0
        for j in range(K):
            for t in range(T - 1):
                denominator += xi[t][i][j]

        # calculate the transition probability for each state
        for j in range(K):
            numerator = 0

            for t in range(T - 1):
                numerator += xi[t][i][j]

            row.append(numerator / (denominator + 1e-12))

        Transition.append(row)

    Transition = np.array(Transition)

    # update Gaussian means
    means = []

    # go through every hidden state
    for k in range(K):
        row = []

        # add up how much of all epochs belong to the state
        weight_sum = 0
        for t in range(T):
            weight_sum += gamma[t][k]

        # go through each feature and compute a weighted total
        for d in range(D):
            total = 0

            for t in range(T):
                total += gamma[t][k] * Features[t][d]

            row.append(total / (weight_sum + 1e-12))  # make it a soft count

        means.append(row)

    means = np.array(means)

    # update Gaussian variances
    variances = []

    # go through every hidden state
    for k in range(K):
        row = []

        # add up how much of all epochs belong to the state
        weight_sum = 0
        for t in range(T):
            weight_sum += gamma[t][k]

        for d in range(D):
            total = 0

            # difference = observed / means ^2
            for t in range(T):
                difference = Features[t][d] - means[k][d]
                total += gamma[t][k] * (difference**2)

            variance = total / (weight_sum + 1e-12)  # compute weighted variance

            # make sure variance isnt 0
            if variance < 1e-6:
                variance = 1e-6

            row.append(variance)

        variances.append(row)

    variances = np.array(variances)

    return initial_prob, Transition, means, variances


def m_step_update_multiple_sequences(feature_sequences, gammas, xis):
    """
    Run one M-step using all participants together.
    """

    # -----------------------------
    # Update initial probabilities
    # -----------------------------

    initial_prob = np.zeros(K)

    for gamma in gammas:
        initial_prob += gamma[0]

    initial_prob = initial_prob / np.sum(initial_prob)

    # -----------------------------
    # Update transition matrix
    # -----------------------------

    transition_counts = np.zeros((K, K))

    for xi in xis:
        # xi has shape (T - 1, K, K)
        # summing over time gives expected transitions i -> j
        transition_counts += np.sum(xi, axis=0)

    Transition = np.zeros((K, K))

    for i in range(K):
        row_sum = np.sum(transition_counts[i])

        if row_sum == 0:
            Transition[i] = np.full(K, 1 / K)
        else:
            Transition[i] = transition_counts[i] / row_sum

    # -----------------------------
    # Update Gaussian means
    # -----------------------------

    state_weights = np.zeros(K)
    weighted_feature_sums = np.zeros((K, D))

    for Features, gamma in zip(feature_sequences, gammas):
        T = len(Features)

        for t in range(T):
            for k in range(K):
                state_weights[k] += gamma[t][k]

                for d in range(D):
                    weighted_feature_sums[k][d] += gamma[t][k] * Features[t][d]

    means = np.zeros((K, D))

    for k in range(K):
        for d in range(D):
            means[k][d] = weighted_feature_sums[k][d] / (state_weights[k] + 1e-12)

    # -----------------------------
    # Update Gaussian variances
    # -----------------------------

    weighted_variance_sums = np.zeros((K, D))

    for Features, gamma in zip(feature_sequences, gammas):
        T = len(Features)

        for t in range(T):
            for k in range(K):
                for d in range(D):
                    difference = Features[t][d] - means[k][d]
                    weighted_variance_sums[k][d] += gamma[t][k] * (difference**2)

    variances = np.zeros((K, D))

    for k in range(K):
        for d in range(D):
            variance = weighted_variance_sums[k][d] / (state_weights[k] + 1e-12)

            if variance < 1e-6:
                variance = 1e-6

            variances[k][d] = variance

    return initial_prob, Transition, means, variances


def baum_welch_training_shell(
    feature_sequences,
    transition_init="informed",
    emission_init="informed",
    feature_method=None,
    return_initial_params=False,
):
    """
    Run Baum-Welch training across multiple participants.
    """

    initial_prob, Transition, means, variances = initialize_training_params(
        transition_init=transition_init,
        emission_init=emission_init,
        feature_method=feature_method,
    )

    # Save the starting parameters before Baum-Welch changes them.
    input_initial_prob = initial_prob.copy()
    input_transition = Transition.copy()
    input_means = means.copy()
    input_variances = variances.copy()

    log_likelihoods = []

    for iteration in range(ITERATIONS):

        gammas = []
        xis = []
        total_log_likelihood = 0.0

        # E-step for each participant.
        for Features in feature_sequences:
            gamma, xi, log_likelihood = forward_backward(
                Features,
                Transition,
                means,
                variances,
                initial_prob,
            )

            gammas.append(gamma)
            xis.append(xi)
            total_log_likelihood += log_likelihood

        # M-step across all participants.
        initial_prob, Transition, means, variances = m_step_update_multiple_sequences(
            feature_sequences,
            gammas,
            xis,
        )

        log_likelihoods.append(total_log_likelihood)
        print(f"Iteration {iteration + 1}, log likelihood: {total_log_likelihood}")

        # Stop if log likelihood stops changing much.
        if iteration > 0:
            previous = log_likelihoods[iteration - 1]
            current = log_likelihoods[iteration]
            change = abs(current - previous)

            if change < THRESHOLD:
                break

    if return_initial_params:
        return {
            "initial_prob": initial_prob,
            "transition": Transition,
            "means": means,
            "variances": variances,
            "log_likelihoods": log_likelihoods,
            "input_initial_prob": input_initial_prob,
            "input_transition": input_transition,
            "input_means": input_means,
            "input_variances": input_variances,
        }

    return initial_prob, Transition, means, variances, log_likelihoods
