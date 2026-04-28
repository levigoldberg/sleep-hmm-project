import numpy as np
from scipy.special import logsumexp

#1: Set up states/fake data
    #States:
        # 0 = Wake: alpha and beta waves, fast
        # 1 = N1: Theta 
        # 2 = N2: Theta + some Delta
        # 3 = N3: Delta, slow, deep sleep
        # 4 = REM: Theta + Beta, dreaming, active 
    #Vector: [delta, theta, alpha, beta]
K = 5 #number of hidden states
F = 4 #number of features (delta, theta, alpha, beta)
STATE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']


TRUE_MEANS = np.array([
    [0.10, 0.15, 0.40, 0.35],   # Wake
    [0.20, 0.45, 0.20, 0.15],   # N1
    [0.35, 0.35, 0.15, 0.15],   # N2
    [0.65, 0.20, 0.10, 0.05],   # N3
    [0.20, 0.40, 0.15, 0.25],   # REM
])

TRUE_VARS = np.full((K, F), 0.02) #controls how spread out the observations are around the mean. 0.02 = standard deviation of ~0.14, which gives realistic looking scatter without the fake data being to noisy. 

def generate_fake_data(T =200, seed = 42): #42: just a random seed for reproducibility, so we get the same fake data every time we run the code
    #Returns: 
        #X: (T,4) array of observations (delta, theta, alpha, beta)
        #Z_true: (T,) array of true state indicies (for checking your work later)

    rng = np.random.default_rng(seed) #creates a random nnumber generater with a fixed starting point, so we get the same fake data every time we run the code

    #transition: mostly stay, but some chance to switch states
        #mostly stay: You don't flicker between Wake and N3 every 30 seconds — you stay in a stage for several minutes before transitioning. 
    A_true = np.array([ #generate fake data with these transition probabilities
        [0.80, 0.15, 0.03, 0.01, 0.01],  # Wake
        [0.05, 0.75, 0.15, 0.03, 0.02],  # N1
        [0.02, 0.05, 0.75, 0.15, 0.03],  # N2
        [0.02, 0.03, 0.10, 0.80, 0.05],  # N3
        [0.10, 0.05, 0.05, 0.05, 0.75],  # REM
    ])

    #start in Wake, just for testing forward-backward later
        #Wake:real sleep recoding starts when you put the electrodes on
    Z_true = np.zeros(T, dtype=int)
    for t in range(1, T):
        Z_true[t] = rng.choice(K, p=A_true[Z_true[t-1]])
    
    #sample observations from Gaussian around each state's mean
    X = np.zeros((T, F))
    for t in range(T):
        state = Z_true[t]
        X[t] = rng.normal(TRUE_MEANS[state], np.sqrt(TRUE_VARS[state]))
        X[t] = np.clip(X[t], 0, None)          #no negatives
        X[t] /= X[t].sum()                      #re-normalize to sum to 1
    
    return X, Z_true

#TEST

X,Z_true = generate_fake_data(T=200) #Z_true ground truth: just to make sure the forward-backward gets the right states

print("X shape:", X.shape) #200 epochs and 4 features each, confirm array was built correct
print("First 5 states:", [STATE_NAMES[z] for z in Z_true[:5]]) #True hidden state sequence (Z_true): ground truth we generate
print("First observation:", X[0].round(3)) #feature vector at t=0 (wake epoch) 




#2: Build emission model (Gaussians) -  In state k, what is the probability of observing this particular combination of delta, theta, alpha, beta (vector x)?
    #assume each states observation follow a Gaussian distribution use multivariate Gaussian probabilty formula (standard for continuous HMMs)
    #P(x | state=k) = P(delta | k) × P(theta | k) × P(alpha | k) × P(beta | k)
def gaussian_emission(x,means,variances):
    #Arguments:
        #X: (F,) vector of features (delta, theta, alpha, beta) at time t
        #means: (K,F) mean vector for each state
        #variances: (K,F) variance vector for each state
    #Returns:
        #probs: (K,) probablity of x under each state's Gaussian
    
    #Gaussian formula: (1/sqrt(2π var)) * exp(-0.5 * (x - mean)^2 / var)
    diff = x-means 
    exponent = -0.5 * (diff ** 2) / variances           
    coeff    = 1.0 / np.sqrt(2 * np.pi * variances)    

    #multiply across features (assumes independence)
    probs = np.prod(coeff * np.exp(exponent), axis = 1) #Axis = 1 collapses the columns (multiplys across the 4 features)
    return probs     

#TEST with first observation (true state = Wake)
#Expect Wake to score highest because X[0] has high alpha + beta, low delta
#which matches Wake's mean [0.10, 0.15, 0.40, 0.35]
probs = gaussian_emission(X[0], TRUE_MEANS, TRUE_VARS)
for name, p in zip(STATE_NAMES, probs):
    print(f"{name}: {p:.6f}")


#NOTE: both forward and backward are written with log space due to underflow or overflow respectively - functions w/o it below
    #underflow(forward):After 200 time steps of multiplying probabilities together, these numbers will get astronomically small and Python will round them to zero
    #overflow(backward): emission probabilities are bigger than 1, so multiplying 200 of those together will get astronomically large
    #instead of multiplying probabilities: a × b × c, we add their logs: log(a) + log(b) + log(c)
    #logsumexp handles the tricky case of summing in log space (replaces np.sum)

#3:Implement forward in log space - probability of seeing data up to time t
def forward_log(X, A, means, variances, pi):
    #Arguments:
        #X: (T,F) observation sequence
        #A: (K,K) transition matrix
        #means: (K,F) emission means
        #variances: (K,F) emission variances
        #pi: (K,) initial state distribution
    #Return:
        #log_alpha: (T,K) log forward probabilities

    T = X.shape[0] #gets the 200, number of time steps
    log_alpha = np.zeros((T, K)) #creates empty (200,5) array of zeros
    log_A = np.log(A)

    #initialize at t=0, log probability of starting in each state × probability of first observation
    log_emission = np.log(gaussian_emission(X[0], means, variances))  # (K,)
    log_alpha[0] = np.log(pi + 1e-300) + log_emission  # +1e-300 avoids log(0) for non-Wake states

    #Recursive forward in log space
    for t in range(1, T):
        log_emission = np.log(gaussian_emission(X[t], means, variances))  # (K,) how likely is observation at t under each state
        for k in range(K):
            #logsumexp replaces np.sum when in log space
            log_alpha[t, k] = log_emission[k] + logsumexp(log_alpha[t-1] + log_A[:, k])

    return log_alpha

#4: Implement backward pass in log space - probability of seeing all observations after time t, given state k at time t
    #β_t(k) = P(x_{t+1}, x_{t+2}, ..., x_T | z_t = k)
    #β_t(k) = Σ_j [ A[k,j] × P(x_{t+1} | z_{t+1}=j) × β_{t+1}(j) ] (Σ_j: sum over all states j you could go to next)
        #probability of all future observations given k = sum of every state j that it could go to next
    #Backward: A[k, :] — transitions out of k to everything
def backward_log(X, A, means, variances):
    #Arguments:
        #X: (T,F) observation sequence
        #A: (K,K) transition matrix
        #means: (K,F) emission means
        #variances: (K,F) emission variances
    #Returns:
        #log_beta: (T,K) log backward probabilities

    T = X.shape[0]
    log_beta = np.zeros((T, K))
    log_A = np.log(A)

    #Initialize at t=T-1 (last time step)
    log_beta[T-1] = 0.0  # log(1)=0: at t=T-1 there are no future observations, probability of all future observations is 1 (certain that nothing happens after the end)

    #Recursively go backward in log space
    for t in range(T-2, -1, -1):
        log_emission_next = np.log(gaussian_emission(X[t+1], means, variances))  # (K,)
        for k in range(K):
            log_beta[t, k] = logsumexp(log_A[k, :] + log_emission_next + log_beta[t+1])

    return log_beta

#TEST
pi = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  #always start in Wake
A_init = np.full((K, K), 1/K)  #uniform: every transition equally likely

log_alpha = forward_log(X, A_init, TRUE_MEANS, TRUE_VARS, pi)
log_beta = backward_log(X, A_init, TRUE_MEANS, TRUE_VARS)

print("log_alpha shape:", log_alpha.shape)         #should be (200, 5)
print("log_alpha[0]:", log_alpha[0].round(2))      #Wake should dominate, others -inf or very negative
print("log_beta[T-1]:", log_beta[199].round(2))    #should be [0, 0, 0, 0, 0] because log(1)=0 and at t=T-1 there are no future observation 
print("log_beta[0]:", log_beta[0].round(2))        #should be finite numbers becuase we're using a unfiform A_nit (every state looks equally likely when working backward)

#5: Combine forward/backward: Given the entire observation sequence, what's the probability of being in state k at time t
    #γ_t(k) = α_t(k) × β_t(k)  (normalized to sum to 1)

def forward_backward(X,A,means, variances, pi):
    #Arguments:
        #X: (T,F) observation sequence
        #A: (K,K) transition matrix
        #means: (K,F) emission means
        #variances: (K,F) emission variances
        #pi: (K,) initial state distribution
    #Return:
        #gamma: (T,K) posterior state probabilities (gamma[t, k] = P(z_t = k | all observations)

     log_alpha = forward_log(X, A, means, variances, pi)
     log_beta  = backward_log(X, A, means, variances)

     #add in log space = multiply in probability space
     #α_t(k) × β_t(k) - multiplying forward and backward together for every t and k simultaneously
     log_gamma = log_alpha + log_beta                          

    # normalize across states at each time step
    # (subtract logsumexp = divide by total probability)
     log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)  

    #convert back to normal probabilities
     gamma = np.exp(log_gamma)                                 # (T, K)
     return gamma

#TEST
gamma = forward_backward(X, A_init, TRUE_MEANS, TRUE_VARS, pi)

print("gamma shape:", gamma.shape)             #output table - should be (200, 5): 200 time steps x 5 steps
print("gamma[0]:", gamma[0].round(4))          #should sum to 1: at t=0, 100% certain that were in Wake
print("gamma[0] sums to:", gamma[0].sum())     #should be exactly 1.0, checks normalization step worked
# most likely state at each time step
predicted_states = np.argmax(gamma, axis=1)
print("First 5 predicted:", [STATE_NAMES[z] for z in predicted_states[:5]]) 
print("First 5 true:     ", [STATE_NAMES[z] for z in Z_true[:5]])



"""
#3: Implement forward pass - probability of seeing data up to time t
    #At each time t
        #1: predict - for each state k, sum up all the ways you could have arrived at k from any state at t-1
        #2: update - multiply by the emission probabilty of the current observation
        #α_t(k) = P(x_t | z_t=k) (emission) ×  Σ_j [ α_{t-1}(j) × A[j,k] (sum over all previous states) ]
    #Forward: A[:, k] — transitions into k from everything


def forward(X,A,means,variances,pi):
    #Arguments:
        #X: (T,F) observation sequence
        #A: (K,K) transition matrix
        #means: (K,F) emission means
        #variances: (K,F) emission variances
        #pi: (T,) initial state distribution
    #Return
        #alpha: (T,K) forward probabilities
    
    
    T = X.shape[0] #gets the 200, number of time steps
    alpha = np.zeros((T,K)) #creates empty (200,5) array of zeros 

    #initialize at t=0, probability of starting in each state × probability of first observation
    alpha[0] = pi * gaussian_emission(X[0], means,variances)

    #Recursive forward
    for t in range (1,T):
        emission = gaussian_emission(X[t], means, variances) #(K,) how likely is observation at t-1 under each state
        for k in range(K):
            #sum over all states j we could have came from
            alpha[t,k] = emission[k] * np.sum(alpha[t-1] * A[:k])
        
    return alpha 

#TEST: initialize a simple A to test with (we'll improve this later)
pi = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # always start in Wake
A_init = np.full((K, K), 1/K)  # uniform: every transition equally likely

alpha = forward(X, A_init, TRUE_MEANS, TRUE_VARS, pi)

print("alpha shape:", alpha.shape)        # should be (200, 5) (one row per time step, one col per state)
print("alpha[0]:", alpha[0].round(4))     # t=0: Wake should dominate becuase we told the model to start in wake (pi = [1, 0, 0, 0, 0])
print("alpha[1]:", alpha[1].round(4))     #A_init(transition is uniform), so wake is 0 and Rem is highest becuase it the probabilitiy of transitioning away from wake is = likely
    #Conclution: Forward algorithm is only as good as the A and emissin parameters


#4: Implement backward pass - probability of seeing all the observations after time t, given that state k at time t
    #β_t(k) = P(x_{t+1}, x_{t+2}, ..., x_T | z_t = k)
    #β_t(k) = Σ_j [ A[k,j] × P(x_{t+1} | z_{t+1}=j) × β_{t+1}(j) ] (Σ_j: sum over all states j you could go to next)
        #probability of all future observations given k = sum of every state j that it could go to next 
    #Backward: A[k, :] — transitions out of k to everything

def backward(X,A,means, variances):
    #Arguments:
        #X: (T,F) observation sequence
        #A: (K,K) transition matrix
        #means: (K,F) emission means
        #variances: (K,F) emission variances
    #Returns:
        #beta: (T,K) backward probabilities
    
    T = X.shape[0]
    beta = np.zeros((T,K))

    #Initialize at t =T-1 (last time step)
    beta[T-1] = 1.0 #At the last time step there are no future observations, probability of all future observations is 1 (certain that nothing happens after the end)

    #Recursively go backward
    for t in range (T-2 , -1, -1):
        emission_next = gaussian_emission(X[t+1], means, variances) #(K,)
        for k in range(K):
            beta[t,k] = np.sum(A[k, :] * emission_next * beta[t+1])
    return beta

beta = backward(X, A_init, TRUE_MEANS, TRUE_VARS)

print("beta shape:", beta.shape)      #should be (200, 5)
print("beta[T-1]:", beta[199])        #should be [1, 1, 1, 1, 1] becuase at t=T-1 there are no future observation 
print("beta[0]:", beta[0].round(4))   #
"""




#5: Combine forward/backward 
