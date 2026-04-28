import numpy as np
from hmm_inference import forward_backward
import constants

def initialize_training_params():
    """
    Intializing the starting prob, transition matrix, means and variance -- to be changed later just a placeholder
    """
    #Assuming starting prob is uniform, make inital prob matrix:
    initial_prob = []

    for k in range(K):
        initial_prob.append(1 / K)

    initial_prob = np.array(initial_prob)

    #starting assumption for transitions - every one has the same prob of transitioning
    Transition = []

    for i in range(K):
        row = []
        for j in range(K):
            row.append(1 / K)
        Transition.append(row)

    Transition = np.array(Transition)

    #random means to start
    means = []

    for k in range(K):
        row = []
        for d in range(D):
            row.append(np.random.random())
        means.append(row)

    means = np.array(means)

    #spread at beginning - all ones
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
    T = len(Features) #num of epochs

    # update starting probabilities
    initial_prob = []

    for k in range(K):
        initial_prob.append(gamma[0][k])

    initial_prob = np.array(initial_prob)

    # make a new transition matrix
    Transition = []

    for i in range(K):
        row = []

        #count how many times we leave state i
        denominator = 0
        for j in range(K):
            for t in range(T - 1):
                denominator += xi[t][i][j]

        #calculate the transition probability for each state
        for j in range(K):
            numerator = 0

            for t in range(T - 1):
                numerator += xi[t][i][j]

            row.append(numerator / (denominator + 1e-12))

        Transition.append(row)

    Transition = np.array(Transition)

    # update Gaussian means
    means = []

    #go through every hidden state
    for k in range(K):
        row = []

        #add up how much of all epochs belong to the state
        weight_sum = 0 
        for t in range(T):
            weight_sum += gamma[t][k]

        #go through each feature and compute a weighted total 
        for d in range(D):
            total = 0

            for t in range(T):
                total += gamma[t][k] * Features[t][d]

            row.append(total / (weight_sum + 1e-12)) #make it a soft count

        means.append(row)

    means = np.array(means)

    # update Gaussian variances
    variances = []

    #go through every hidden state
    for k in range(K):
        row = []

        #add up how much of all epochs belong to the state
        weight_sum = 0
        for t in range(T):
            weight_sum += gamma[t][k]

        for d in range(D):
            total = 0

            #difference = observed / means ^2
            for t in range(T):
                difference = Features[t][d] - means[k][d]
                total += gamma[t][k] * (difference ** 2)

            variance = total / (weight_sum + 1e-12) #compute weighted variance

            #make sure variance isnt 0
            if variance < 1e-6:
                variance = 1e-6

            row.append(variance)

        variances.append(row)

    variances = np.array(variances)

    return initial_prob, Transition, means, variances


def baum_welch_training_shell(Features):
    """
    Calls forward backward for E step, calls both M and E step
    Input: Features
    Output: initial_prob, Transition, means, variances, log_likelihoods (training progress)
    """
    T = len(Features)


    #initialize the training parametetrs
    initial_prob, Transition, means, variances = initialize_training_params(K, D)

    
    log_likelihoods = []

    for iteration in range(ITERATIONS): #we can change this - max times it will run
        
        #E Step:
        gamma, xi, log_likelihood = forward_backward(Features, Transition, means, variances, initial_prob) 

        #M Step
        initial_prob, Transition, means, variances = m_step_update(Features, gamma, xi) 

        #store the score to see if we are improving
        log_likelihoods.append(log_likelihood)

        #once 2 rounds or more have been done, check to see if the score is improving or not
        if iteration > 0:
            previous = log_likelihoods[iteration - 1]
            current = log_likelihoods[iteration]
            change = abs(current - previous)


            if change < THRESHOLD: #if the change is super small stop
                break

    return initial_prob, Transition, means, variances, log_likelihoods