import numpy as np
import dill as pickle

from environment import Easy21
import utils
import time

toc = time.time()

env = Easy21()
N0 = 100
actions = [0, 1]

trueQ = pickle.load(open('Q.dill', 'rb'))

alpha = 0.01
eps = 0.05


def reset():
    w = np.zeros((3*6*2, 1))
    wins = 0
    
    return w, wins

def epsilonGreedy(p, d):
    
    if np.random.random() <= eps:
        action = np.random.choice(actions)
    else:
        action = np.argmax([Q(p, d, a) for a in actions])
    
    return action

def features(p, d, a):
    feature = np.zeros((3, 6, 2))

    dealer = [[1, 4], [4, 7], [7, 10]]
    player = [[1,6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]

    dealer_feature = np.array([x[0] <= d <= x[1] for x in dealer])
    player_feature = np.array([x[0] <= p <= x[1] for x in player])
    
    for i in np.where(dealer_feature):
        for j in np.where(player_feature):
            feature[i, j, a] = 1

    return feature.reshape(1, -1)

allFeatures = np.zeros((22, 11, 2, 3*6*2))
for p in range(1, 22):
    for d in range(1, 11):
        for a in range(2):
            allFeatures[p-1, d-1, a] = features(p, d, a)


def Q(p, d, a):
    return np.dot(features(p, d, a), w)

def allQ():
    return ((allFeatures.reshape(-1, 3*6*2)).dot(w)).reshape(-1)



episodes = int(1e4)
lmds = list(np.arange(0, 11)/10)
mselamdas = np.zeros(len(lmds))
mse_history = np.zeros((len(lmds), episodes))

for idx, lmd in enumerate(lmds): 
    
    w, wins = reset()

    for episode in range(episodes):
        terminate = False

        E = np.zeros_like(w)

        p, d = env.initGame()
        a = epsilonGreedy(p, d)

        while not terminate:

            p_next, d_next, r, terminate = env.step(p, d, a)

            if not terminate:  
                a_next = epsilonGreedy(p_next, d_next)  #SARSA(lamda): needs s,a,(r),s_next,**a_next**
                td_error = r + Q(p_next, d_next, a_next) - Q(p, d, a)
            else:
                td_error = r - Q(p, d, a)

            E = lmd * E + features(p, d, a).reshape(-1, 1)
            
            w_grad = alpha * td_error * E
            w += w_grad

            if not terminate:
                p, d, a = p_next, d_next, a_next

        if r == 1:
            wins += 1

        mse = np.sum(np.square(allQ()-trueQ.ravel())) / (21*10*2)
        mse_history[idx, episode] = mse

    
    mselamdas[idx] = mse
    print("Lambda %.1f, Episode 10000, MSE %5.3f, Win prob %.3f" % (lmd, mse, wins / (episode + 1) ))
    print("--------")

tic = time.time()
print("Implementation of lfa done! Comsumed time: %fs" % (tic - toc))


utils.plotMseLamdas(mselamdas, lmds)
utils.plotMseEpisodesLambdas(mse_history)

        
            