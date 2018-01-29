
import numpy as np
from operator import add
from scipy import stats
"""Module runs differentially private UCB Experiments with truncated gaussian noise. This includes 
an implementation of the counter mechanism https://eprint.iacr.org/2010/076.pdf

Variables: 
    - history[i] = (beta_hat, 1/(XTX + lambda I), XTY, N_it) for i in 1...K
    -
    -
"""



def get_min_dex(binary_string):
    """Get the min non-zero index in a binary string. Helper fn for priv counter."""
    ind = 0
    while ind < len(binary_string):
        if binary_string[ind] == '1':
            return ind
        ind += 1


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
    beta_t = betas[index]
    V_t_inv = history[index][1]
    XTY = history[index][2]
    # pull arm i
    y_it = get_sample(beta_t, x_t)

    # update 1/(XTX + lambda I): Sherman Morrison Formula
    if np.linalg.det(V_t_inv) == 0:
        V_t_inv = np.linalg.pinv(V_t_inv + np.outer(x_t, x_t))
        history[index][1] = V_t_inv
    else:
        v_inv_x = np.dot(V_t_inv, x_t)
        history[index][1] = V_t_inv - 1.0/(1 + matrix_norm(V_t_inv, x_t))*np.outer(v_inv_x, v_inv_x)
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
    start = [[0.0]*d, np.zeros((d, d)), np.zeros(d), 0]
    return dict((i, start[:]) for i in range(K))


def get_betas(d, k):
    """Return list of k, d-dimensional beta vectors"""
    return [[np.random.uniform(-1,1) for _ in range(d)] for _ in range(k)]


def get_sample(beta, x):
    """Return sample from beta*x + uniform."""

    return np.random.uniform(-1, 1) + np.dot(beta, x)


def gen_contexts(k, d):
    return [[np.random.uniform(-1, 1) for _ in xrange(d)] for _ in xrange(k)]


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


K = 5
d = 6
lbda = .00001
delta = .99
T = 50000
H_T = ucb_bandit_run(K, d, lbda, delta, T)
hist = H_T[0]
beta = H_T[1]
most_pulls = np.max([hist[i][3] for i in range(K)])
most_pulled = np.argmax([hist[i][3] for i in range(K)])
least_pulled = np.argmin([hist[i][3] for i in range(K)])
print('most pulls: {}'.format(most_pulls))
print(hist[most_pulled][0])
print(beta[most_pulled])
