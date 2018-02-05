import numpy as np
from scipy.stats import truncnorm
from operator import add
import matplotlib.pyplot as plt
import sys
"""Module runs differentially private UCB Experiments with truncated gaussian noise. This includes 
an implementation of the counter mechanism https://eprint.iacr.org/2010/076.pdf
"""


def private_counter(k, T, epsilon, sensitivity=2):
    """Returns array of T representing sum of laplace noise added to means in epsilon d.p. private counter"""
    priv_noises = dict((u, []) for u in range(k))
    for t in range(k):
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
            alpha_array[i] = np.random.laplace(loc=0, scale=sensitivity*np.log2(T)/eps_prime)  # Laplace noise
            # Noise added to the jth sum is the sum of all of the alphas with nonzero binary representation
            priv_noise[j-1] = np.sum([alpha_array[k] for k in range(min(digits, len(bin_j))) if bin_j[k] == '1'])
        priv_noises[t] = priv_noise
    return priv_noises


def get_min_dex(binary_string):
    """Get the min non-zero index in a binary string. Helper fn for priv counter."""
    ind = 0
    while ind < len(binary_string):
        if binary_string[ind] == '1':
            return ind
        ind += 1


def get_ucb(delta, history=None):
    """Return the index of the arm with highest UCB."""
    if history is None:
        return None
    K = len(history.keys())
    ucb_list = [history[i][0]/history[i][1] + np.sqrt(2*np.log(1/delta)/history[i][1]) for i in range(K)]
    ucb = np.argmax(ucb_list)
    return ucb


def update_history(history, index, mus):
    """Pull arm index, update and return the history accordingly."""

    # pull arm i
    x_it = get_sample(mus[index])
    history[index][0] += x_it
    history[index][1] += 1.0
    return history


def get_means(gap=.1):
    """Return list of 1/gap means separated by gap."""

    means = []
    mu = 1
    while mu > 0:
        means.append(mu)
        mu = mu-gap
    return means


def get_sample(mu):
    """Return sample from truncated normal in [0,1] with mean mu."""

    # sample truncated normal in [-mu, 1-mu]
    mean_s = truncnorm.mean(-mu, 1-mu, 0, 1)
    s = truncnorm.rvs(-mu,1-mu)
    # get mean mu
    s = s - mean_s + mu
    return s


def get_priv_ucb(delta, history, priv_noises, T, epsilon):
    if history is None:
        return None
    K = len(history.keys())
    gamma = K*np.power(np.log(T), 2)*np.log(K*T*np.log(T)*1.0/delta)*1.0/epsilon
    noisy_means = [(history[i][0] + priv_noises[i][int(history[i][1])])/history[i][1] for i in range(K)]
    ucb_list = [noisy_means[i] + np.sqrt(2*np.log(1/delta)/history[i][1]) + gamma/history[i][1] for i in range(K)]
    ucb = np.argmax(ucb_list)
    return ucb


def ucb_bandit_run(time_horizon=500, gap=.1):
    """"Run UCB algorithm up to time_horizon with K arms of gap .1
        Return the history up to time_horizon
    """
    arm_pulls = []
    means = get_means(gap)
    K = len(means)
    # history at time 0
    history = dict((i, [0, 0]) for i in range(K))
    t = 1
    # Sample initial point from each arm
    while t <= K:
        history[t-1] = [get_sample(means[t-1]), 1]
        arm_pulls.append(t-1)
        t += 1
    # Run UCB Algorithm from t = K + 1 to t = time_horizon
    while t <= time_horizon:
        arm_pull = get_ucb(.9, history)
        arm_pulls.append(arm_pull)
        history = update_history(history, arm_pull, means)
        t += 1
    return [history, arm_pulls]


def priv_ucb_bandit_run(time_horizon=500, delta=.95, gap=.1, epsilon=.1):
    """"Run epsilon-Private UCB algorithm w/ private counter
     up to time_horizon with K arms of gap .1. Return the history up to time_horizon.
    """
    arm_pulls = []
    means = get_means(gap)
    K = len(means)
    priv_noises = private_counter(K, time_horizon, epsilon, sensitivity=2)
    # history at time 0
    history = dict((i, [0, 0]) for i in range(K))
    t = 1
    # Sample initial point from each arm
    while t <= K:
        history[t-1] = [get_sample(means[t-1]), 1]
        arm_pulls.append(t - 1)
        t += 1
    # Run UCB Algorithm from t = K + 1 to t = time_horizon
    while t <= time_horizon:
        arm_pull = get_priv_ucb(delta, history, priv_noises, time_horizon, epsilon)
        arm_pulls.append(arm_pull)
        history = update_history(history, arm_pull, means)
        t += 1
    return [history, arm_pulls]


def compute_avg_pseudo_regret(arm_pulls, mus):
    """"Compute the cumulative average regret of bandit algorithm with history arm_pulls and means mus.
    """
    time = len(arm_pulls)
    pseudo_reward = [mus[arm_pulls[i]] for i in range(time)]
    cum_pseudo_reward = np.cumsum(pseudo_reward)
    opt_mean = np.max(mus)
    cum_opt_reward = [(i+1)*opt_mean for i in range(time)]
    cum_pseudo_regret = [1.0/(j+1)*(cum_opt_reward[j]-cum_pseudo_reward[j]) for j in range(time)]
    return cum_pseudo_regret


# Run bandit experiments, generate bias & regret plots
#
if __name__ == "__main__":
    num_digits, gap, nsims = sys.argv[1:]
    # initialize parameters
    num_digits = int(num_digits)
    T = np.power(2, num_digits)
    gap = float(gap)
    mus = get_means(gap)
    K = len(mus)
    cum_mu_hat = [0]*K
    n_sims = int(nsims)
    cum_av_regret = [0]*T

    # Get sample means up to time T
    # Average over n_sims iterations
    # Compute Bias

    for j in range(n_sims):
        bandit = ucb_bandit_run(time_horizon=T, gap=gap)
        H_T = bandit[0]
        mu_hat = [H_T[i][0]/H_T[i][1] for i in range(K)]
        cum_mu_hat = map(add, cum_mu_hat, mu_hat)
        arms_pulled = bandit[1]
        av_regret = compute_avg_pseudo_regret(arms_pulled, mus)
        cum_av_regret = map(add, cum_av_regret, av_regret)

    # Compute the bias.
    average_mu_hat = np.multiply(1.0/n_sims, cum_mu_hat)
    bias = map(add, average_mu_hat, np.multiply(-1.0, mus))
    av_av_regret = list(np.multiply(cum_av_regret, 1.0/n_sims))
    #  95% conf. lower bound for the bias (Hoeffding Inequality)
    w = np.sqrt(-1*np.log(.975/2)/(2.0*n_sims))

    print(bias)
    print('non-private bias: {}'.format(bias))
    print('confidence width for bias: {}'.format(w))

    # plot the bias vs the arm_mean (barplot with CI)
    bars = bias
    yer1 = [w]*len(bias)
    bar_width = gap
    r1 = [x + bar_width-gap for x in mus]
    plt_ucb = plt
    plt_ucb.bar(r1, bars, width=bar_width, color='green', edgecolor='black', yerr=yer1, capsize=7, label='bias')
    plt_ucb.xlabel('arm mean')
    plt_ucb.ylabel('bias')
    plt_ucb.title('UCB bias per arm')
    plt_ucb.savefig('private_ucb_bias.pdf')


    # Private Version
    cum_mu_hat = [0]*K
    eps = 1/np.sqrt(T*K)
    cum_av_priv_regret = [0]*T
    for j in range(n_sims):
        private_bandit = priv_ucb_bandit_run(time_horizon=T, delta=.95, gap=.1, epsilon=eps)
        H_T_private = private_bandit[0]
        mu_hat = [H_T_private[i][0]/H_T_private[i][1] for i in range(K)]
        cum_mu_hat = map(add, cum_mu_hat, mu_hat)
        arms_pulled = private_bandit[1]
        av_regret_priv = compute_avg_pseudo_regret(arms_pulled, mus)
        cum_av_priv_regret = map(add, av_regret_priv, cum_av_priv_regret)

    # Compute the bias.
    average_mu_hat = np.multiply(1.0/n_sims, cum_mu_hat)
    priv_bias = map(add, mus, np.multiply(-1.0, average_mu_hat))
    priv_av_av_regret = list(np.multiply(cum_av_priv_regret, 1.0/n_sims))
    w_priv = np.sqrt(-1*np.log(.975/2)/(2.0*n_sims))

    print('private bias: {}'.format(priv_bias))
    print('confidence width for bias: {}'.format(w_priv))
    print('max of private bias: {}'.format(np.max(np.abs(priv_bias))))


    # plot the bias vs the arm_mean (barplot with CI)
    bars = priv_bias
    yer1 = [w_priv]*len(priv_bias)
    bar_width = gap
    plt_ucb_priv = plt
    plt_ucb_priv.bar(mus, bars, width=bar_width, color='green', edgecolor='black', yerr=yer1, capsize=7, label='bias')
    plt_ucb_priv.xlabel('arm mean')
    plt_ucb_priv.ylabel('bias')
    plt_ucb_priv.title('Private UCB bias per arm: epsilon = {}'.format(np.round(eps, 2)))
    plt_ucb_priv.savefig('private_ucb_bias_eps_{}.pdf'.format(np.round(eps, 2)))

    # plot the regret over time
    plt.plot(av_av_regret)
    plt.plot(priv_av_av_regret)
    plt.title('cumulative regret: UCB vs. {}-private UCB'.format(np.round(eps, 2)))
    plt.xlabel('T')
    plt.ylabel('average cumulative regret')
    plt.savefig('regret_eps_{}.pdf'.format(np.round(eps, 2)))