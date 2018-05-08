from scipy import stats
import numpy as np
from operator import add
import matplotlib.pyplot as plt
import sys

"""Module runs linear UCB Experiments with bounded reward distributions. This includes 
Usage: python linear_bandits.py 5 5 .001 .99 500 10000
"""



def get_min_dex(binary_string):
    """Get the min non-zero index in a binary string. Helper fn for priv counter."""
    ind = 0
    while ind < len(binary_string):
        if binary_string[ind] == '1':
            return ind
        ind += 1

# assumption of normal noise
def get_lin_ucb(t, delta, lbda, contexts, history=None):
    """Return the index of the arm with highest UCB."""
    K = len(history.keys())
    est_payoffs = [np.dot(history[i][0], contexts[i]) for i in range(K)]
    if lbda == 0:
        sigma = [np.sqrt(np.dot(np.dot(history[i][1], np.transpose(contexts[i])))) for j in range(K)]
        ucbs = [stats.norm.interval(1.0-delta, loc=est_payoffs[i], sigma=sigma[i]) for i in range(K)]

    ci_scale = [np.sqrt(2*K*np.log(1.0/delta*(1 + history[i][3]/lbda))) for i in range(K)]
    widths = [matrix_norm(history[i][1], contexts[i])*ci_scale[i] for i in range(K)]
    ucb_list = map(add, est_payoffs, widths)
    ucb = np.argmax(ucb_list)
    return ucb


def matrix_norm(M, x):
    """Return x'Mx for vector x matrix M"""
    return np.float(np.dot(np.dot(M, x), x))


def update_history(history, index, contexts, betas):
    """Pull arm index, update and return the history accordingly."""

    x_t = contexts[index]
    d = len(x_t)
    history[index][4].append(x_t)
    beta_t = betas[index]
    V_t_inv = history[index][1]
    XTY = history[index][2]
    # pull arm i
    y_it = get_sample(beta_t, x_t)
    history[index][5].append(y_it)

    # update 1/(XTX + lambda I): Sherman Morrison Formula
    if history[index][3] <= d:
        x_list = history[index][4]
        XTX = np.zeros((d, d))
        for x in x_list:
            XTX += np.outer(x, x)
        V_t_inv = np.linalg.pinv(XTX)
        history[index][1] = V_t_inv
    else:
        v_inv_x = np.dot(V_t_inv, x_t)
        history[index][1] = V_t_inv - 1.0/(1.0 + matrix_norm(V_t_inv, x_t))*np.outer(v_inv_x, v_inv_x)

    # update XTY
    XY = np.multiply(y_it, x_t)
    history[index][2] = map(add, XTY, XY)
    # update beta hat
    history[index][0] = np.dot(history[index][1], history[index][2])
    # update counts
    history[index][3] += 1.0
    return history


def empty_history(d, K):
    """Return empty history, K arms dimension d.
    """
    start = [[0.0]*d, np.zeros((d, d)), np.zeros(d), 0, [], []]
    return dict((i, start[:]) for i in range(K))


def get_betas(d, k):
    """Return list of k, d-dimensional beta vectors"""
    betas = [[np.random.uniform(-1,1) for _ in range(d)] for _ in range(k)]
    norm_betas = [b/np.linalg.norm(b) for b in betas]
    return norm_betas


def get_sample(beta, x):
    """Return sample from beta*x + N(0,1)."""

    # return np.random.uniform(-1, 1) + np.dot(beta, x)
    return np.random.normal(0, 1) + np.dot(beta, x)


def gen_contexts(k, d):
    contexts = [[np.random.uniform(-1, 1) for _ in xrange(d)] for _ in xrange(k)]
    norm_contexts = [b/np.linalg.norm(b) for b in contexts]
    return norm_contexts


def ucb_bandit_run(K, d, lbda, delta, time_horizon=500):
    """"Run UCB algorithm up to time_horizon with K arms of gap .1
        Return the history up to time_horizon
    """
    betas = get_betas(d, k=K)
    # history at time 0
    history = empty_history(d, K)
    t = 1
    # Sample initial point from each arm
    while t <= K:
        contexts = gen_contexts(K, d)
        arm_pull = t-1
        history = update_history(history, arm_pull, contexts, betas)
        t += 1
    # Run UCB Algorithm from t = K + 1 to t = time_horizon
    while t <= time_horizon:
        contexts = gen_contexts(K, d)
        arm_pull = get_lin_ucb(t, delta, lbda, contexts, history=history)
        history = update_history(history, arm_pull, contexts, betas)
        t += 1
    return history, betas


# perform two-sided z-test for beta_ik == c_i, sigma = 1
def t_test_reg(hist_i, k, c_k):
    beta_hat = hist_i[0]
    XTX_inv = hist_i[1]
    b_k = beta_hat[k]
    sigma = XTX_inv[k, k]
    z_score = (b_k-c_k)/sigma
    p_value = 2.0*min(1-stats.norm.cdf(z_score), stats.norm.cdf(z_score))
    return p_value

if __name__ == "__main__":
    K, d, lbda, delta, T, n_sims = sys.argv[1:]
    K = int(K)
    d = int(d)
    lbda = float(lbda)
    delta = float(delta)
    T = int(T)
    p_values = []
    max_arm_bias = [0]*K
    n_sims = int(n_sims)
    for _ in range(n_sims):
        H_T = ucb_bandit_run(K, d, lbda, delta, T)
        hist = H_T[0]
        beta = H_T[1]
        most_pulls = np.max([hist[i][3] for i in range(K)])
        most_pulled = np.argmax([hist[i][3] for i in range(K)])
        least_pulled = np.argmin([hist[i][3] for i in range(K)])
        least_pulls = np.min([hist[i][3] for i in range(K)])
        print('least pulls: {}'.format(least_pulls))
        print('most pulls: {}'.format(most_pulls))
        print(hist[most_pulled][0])
        print(beta[most_pulled])
        k = np.argmax(np.abs(beta[most_pulled]))
        p_val = t_test_reg(hist[most_pulled], k, beta[most_pulled][k])
        p_values.append(p_val)
    fdr = np.mean([p < .05 for p in p_values])
    # plots: p-value histogram - should be uniformly distributed
    bins = 25
    plt.hist(p_values, bins=bins, orientation='horizontal', color='green')
    plt.axhline(.05, color='b', linestyle='dashed', linewidth=2)
    plt.title('p-value histogram: t-test with LinUCB')
    plt.savefig('p-value histogram')
    print('fdr: {}'.format(fdr))
