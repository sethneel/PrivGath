import numpy as np
from scipy.stats import truncnorm


"""Module runs differentially private UCB Experiments with truncated gaussian noise. This includes 
an implementation of the counter mechanism https://eprint.iacr.org/2010/076.pdf
"""


def private_counter(T, epsilon):
    """Returns array of T representing sum of laplace noise added to means in epsilon d.p. private counter"""

    priv_noise = [0.0]*T
    eps_prime = epsilon/np.log2(T)
    digits = int(np.ceil(np.log2(T)))
    alpha_array = [0]*digits
    for j in range(1, T+1):
        """Update noise, stored in priv_noise"""
        # Get the binary expansion of j
        bin_j = '{0:b}'.format(j)
        # Get the first non-zero bit
        i = get_min_dex(bin_j)
        # Set all other alpha j < i to 0
        for l in range(i):
            alpha_array[l] = 0
        alpha_array[i] = np.random.laplace(loc=0, scale=np.log2(T)/eps_prime)  # Laplace noise
        # Noise added to the jth sum is the sum of all of the alphas with nonzero binary representation
        priv_noise[j-1] = np.sum([alpha_array[k] for k in range(min(digits, len(bin_j))) if bin_j[k] == '1'])

    return priv_noise


def get_min_dex(binary_string):
    """Get the min non-zero index in a binary string. Helper fn for priv counter."""
    ind = 0
    while ind < len(binary_string):
        if binary_string[ind] == '1':
            return ind
        ind += 1


def get_ucb(history = None):
    """Return the index of the arm with highest UCB."""
    if history is None:
        return None



def update_history(history, index, mus):
    """Pull arm index, update and return the history accordingly."""

    # pull arm i
    x_it = get_sample(mus[index])
    history[index][0] += x_it
    history[index][1] += 1
    return history


def get_means(gap=.1):
    """Return list of 1/gap means separated by gap."""

    means = []
    mu = 1
    while(mu > 0):
        means.append(mu)
        mu = mu-gap
    return means


def get_sample(mu):
    """Return sample from truncated normal in [-1,1] with mean mu."""

    # sample truncated normal in [-mu, 1-mu]
    mean_s = truncnorm.mean(-mu, 1-mu, 0, 1)
    s = truncnorm.rvs(-mu,1-mu)
    # get mean mu
    s = s - mean_s + mu
    return s


def ucb_bandit_run(time_horizon=500, gap=.1):
    """"Run UCB algorithm up to time_horizon with K arms of gap .1
        Return the history up to time_horizon
    """
    means = get_means(gap)
    K = len(means)
    # history at time 0
    history = {i: [0, 0] for i in range(K)}
    t = 1
    # Sample initial point from each arm
    while t <= K:
        history[t] = [get_sample(means[t-1], 1)]
        t += 1
    # Run UCB Algorithm from t = K + 1 to t = time_horizon
    while t <= time_horizon:
        arm_pull = get_ucb(history)
        history = update_history(history, arm_pull, means)
        t += 1
    return history


def priv_ucb_bandit_run(time_horizon=500, gap=.1, epsilon=.1):
    """"Run epsilon-Private UCB algorithm w/ private counter
     up to time_horizon with K arms of gap .1. Return the history up to time_horizon.
    """



#
# Run experiment
#
# initialize parameters
num_digits = 10
T = np.power(2, num_digits)
