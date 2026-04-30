import numpy as np
from hmm_inference import forward_backward
from constants import K, D, THRESHOLD, ITERATIONS


def initialize_training_params():
    """
    Intializing the starting prob, transition matrix, means and variance -- to be changed later just a placeholder
    """
    # Assuming starting prob is uniform, make inital prob matrix:
    initial_prob = []

    for k in range(K):
        initial_prob.append(1 / K)

    initial_prob = np.array(initial_prob)

    # starting assumption for transitions - every one has the same prob of transitioning
    Transition = []

    for i in range(K):
        row = []
        for j in range(K):
            row.append(1 / K)
        Transition.append(row)

    Transition = np.array(Transition)

    # random means to start
    means = []

    for k in range(K):
        row = []
        for d in range(D):
            row.append(np.random.random())
        means.append(row)

    means = np.array(means)

    # spread at beginning - all ones
    variances = []

    for k in range(K):
        row = []
        for d in range(D):
            row.append(1.0)
        variances.append(row)

    variances = np.array(variances)

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

    Instead of updating parameters from one participant at a time,
    this aggregates expected counts across every participant sequence.
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


def baum_welch_training_shell(feature_sequences):
    """
    Run Baum-Welch training across multiple participants.

    Each participant is treated as a separate observation sequence.
    The E-step runs forward-backward separately for each participant.
    The M-step aggregates expected counts across all participants.
    """

    initial_prob, Transition, means, variances = initialize_training_params()

    log_likelihoods = []

    for iteration in range(ITERATIONS):

        gammas = []
        xis = []
        total_log_likelihood = 0.0

        # E-step: run forward-backward separately for each participant
        for Features in feature_sequences:
            gamma, xi, log_likelihood = forward_backward(
                Features, Transition, means, variances, initial_prob
            )

            gammas.append(gamma)
            xis.append(xi)
            total_log_likelihood += log_likelihood

        # M-step: update once using all participants together
        initial_prob, Transition, means, variances = m_step_update_multiple_sequences(
            feature_sequences, gammas, xis
        )

        log_likelihoods.append(total_log_likelihood)

        print(f"Iteration {iteration + 1}, log likelihood: {total_log_likelihood}")

        if iteration > 0:
            previous = log_likelihoods[iteration - 1]
            current = log_likelihoods[iteration]
            change = abs(current - previous)

            if change < THRESHOLD:
                break

    return initial_prob, Transition, means, variances, log_likelihoods
