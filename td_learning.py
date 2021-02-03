import numpy as np
import dill as pickle

from environment import Easy21
import utils 
import time

toc = time.time()

env = Easy21()
N0 = 100
actions = [0, 1]


def reset():
    Q = np.zeros((22, 11, len(actions)))
    NSA = np.zeros((22, 11, len(actions)))
    wins = 0

    return Q, NSA, wins

Q, NSA, wins = reset()
trueQ = pickle.load(open('Q.dill', 'rb'))
NS = lambda p, d: np.sum(NSA[p, d])

alpha = lambda p, d, a: 1/NSA[p, d, a]
eps = lambda p, d: N0 / (N0 + NS(p, d))

# policy improvement - by epsilon-greedy
def epsilonGreedy(p, d):
    
    if np.random.random() <= eps(p, d):  #explore
        action = np.random.choice(actions)
    else:
        action = np.argmax(Q[p, d, :])
    
    return action


episodes = int(1e4)
lmds = list(np.arange(11)/10)

# output
mselamdas = np.zeros((len(lmds), episodes))
finalMSE = np.zeros(len(lmds))

for idx, lmd in enumerate(lmds):

    Q, NSA, wins = reset()

    for episode in range(episodes):
        terminate = False
        E = np.zeros((22, 11, len(actions)))  # Eligibility trace
        SA = list() 

        p, d = env.initGame()
        a = epsilonGreedy(p, d)

        while not terminate:

            NSA[p, d, a] += 1
            SA.append([p, d, a])
            E[p, d, a] += 1   #update e.t(1): for visited (s,a) 

            p_next, d_next, r, terminate = env.step(p, d, a)

            if not terminate:  
                a_next = epsilonGreedy(p_next, d_next)  #SARSA(lamda): needs s,a,(r),s_next,**a_next**
                td_error = r + Q[p_next, d_next, a_next] - Q[p, d, a]
            else:
                td_error = r - Q[p, d, a]

            for (_p, _d, _a) in SA:
                Q[_p, _d, _a] += alpha(_p, _d, _a) * td_error * E[_p, _d, _a]  # backward view SARSA(lamda) : update all (past) Q(s, a) using e.t
                E[_p, _d, _a] *= lmd   #update e.t(2): for past (s,a)

            if not terminate:
                p, d, a = p_next, d_next, a_next

        if r == 1:
            wins += 1

        mse = np.sum(np.square(Q-trueQ)) / (21*10*2)

    #mse = np.sum(np.square(Q-trueQ)) / (21*10*2)
    #mselamdas[idx] = mse
    finalMSE[idx] = mse

    print("Lambda %.1f, Episode 10000, MSE %5.3f, Win prob %.3f" % (lmd, mse, wins / (episode + 1) ))
    print("--------")

tic = time.time()
print("Implementation of SARSA(lamda) done! Comsumed time: %fs" % (tic - toc))

utils.plotMseLamdas(finalMSE, lmds)



