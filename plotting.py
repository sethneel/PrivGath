import matplotlib.pyplot as plt
import ast
import numpy as np

d = 2
k = 10
N = 100
T = 10000

ucb_grievances, fairucb_grievances = [0.0 for _ in xrange(T)], [0.0 for _ in xrange(T)]
for n in range(N+1)[1:]:
    print n
    f = open('new_all_random_n_' + str(n) + '_d_' + str(d) + '_k_' + str(k) + '_T_' + str(T) + '.txt', 'r')
    lines = f.readlines()[0]
    ucb_new_grievances, fairucb_new_grievances = lines.split(']')[:2]
    ucb_new_grievances = ast.literal_eval(ucb_new_grievances + ']')
    fairucb_new_grievances = ast.literal_eval(fairucb_new_grievances + ']')
    ucb_grievances = [ucb_grievances[i] + ucb_new_grievances[i] for i in range(T)]
    fairucb_grievances = [fairucb_grievances[i] + fairucb_new_grievances[i] for i in range(T)]
    f.close()

ucb_cumulative = np.cumsum(ucb_grievances)
ucb_cumulative = [entry/N for entry in ucb_cumulative]
fairucb_cumulative = np.cumsum(fairucb_grievances)
fairucb_cumulative = [entry/N for entry in fairucb_cumulative]

plt.rcParams.update({'font.size': 20})
plt.plot(range(T), ucb_cumulative, label = "UCB", linewidth = 2, color = "firebrick")
# plt.plot(range(T), fairucb_cumulative, label = "FairUCB", linewidth = 2, color = "dodgerblue")
plt.title('Cumulative mistreatment in UCB')
plt.xlabel('# rounds')
plt.ylabel('Cumulative mistreatment')
# plt.legend(loc = "upper left", prop={'size':18})
plt.tight_layout()
# plt.savefig('both_mistreatment.png')
plt.savefig('ucb_mistreatment.png')

