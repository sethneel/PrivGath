import numpy as np
from scipy.stats import truncnorm

"""Module runs UCB Experiments with truncated gaussian noise.
"""

def get_ucb(history = None):
    """Return the index of the arm with highest UCB."""


def update_history(history, index):
    """Pull arm index, update and return the history accordingly."""


def get_means(K=2, gap=.1):
    """Return list of K means separated by gap."""




def get_sample(mu):
    """Return sample from truncated normal in [-1,1] with mean mu."""

    # sample truncated normal in [-mu, 1-mu]
    mean_s = truncnorm.mean(-mu, 1-mu, 0, 1)
    s = truncnorm.rvs(-mu,1-mu)
    # get mean mu
    s = s - mean_s + mu
    return s

def ucb_bandit_run(time_horizon=500, K=2, gap=.1):
    """"Run UCB algorithm up to time_horizon with K arms of gap .1
        Return the history up to time_horizon
    """


    # history at time 0
    history = {i:[0,0] for i in range(K)}
    t = 1
    means = get_means(K,gap)
    # Sample initial point from each arm
    while t <= K:
        history[t] = [get_sample(means[t-1],1)]
        t +=1
    # Run UCB Algorithm from t = K + 1 to t = time_horizon
    while t <= time_horizon:
        arm_pull = get_ucb(history)
        history = update_history(history, arm_pull)
        t += 1
    return history
