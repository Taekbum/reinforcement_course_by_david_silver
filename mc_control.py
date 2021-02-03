import numpy as np
import dill as pickle

from environment import Easy21
import utils

printEvery = 10000
actions = [0, 1]

N0 = 100

Q = np.zeros((22, 11, len(actions)))  # 22 = 21 + bust / 11 = 10 + color
NSA = np.zeros((22, 11, len(actions)))
NS = lambda p, d: np.sum(NSA[p, d])   #state: (player, dealer)


alpha = lambda p, d, a: 1/NSA[p, d, a]
eps = lambda p, d: N0 / (N0 + NS(p, d))

# policy improvement - by epsilon-greedy
def epsilonGreedy(p, d):
    
    if np.random.random() <= eps(p, d):  #explore
        action = np.random.choice(actions)
    else:
        action = np.argmax(Q[p, d, :])
    
    return action


# for each episode, calculate Q val from curr policy -> policy improvement
episodes = int(1e6)  # hyperparm - can tune it!
meanReturn = 0
wins = 0

env = Easy21()

for episode in range(episodes):

    terminate = False
    SAR = list()  # state, action, reward
    p, d = env.initGame()

    while not terminate:
        
        a = epsilonGreedy(p, d)

        NSA[p, d, a] += 1
        p_next, d_next, r, terminate = env.step(p, d, a)
        SAR.append([p, d, a, r])
        
        p, d = p_next, d_next

    # update Q
    G = sum([sar[-1] for sar in SAR]) # use actual G_t, which is undiscounted sum of rewards until terminate
    for (p, d, a, _) in SAR:
        Q[p, d, a] += alpha(p, d, a) * (G - Q[p, d, a]) # mc update

    # actual running mean
    meanReturn = meanReturn + 1/(episode+1) * (G - meanReturn)

    if r == 1:       # win : reward at terminate of episode is 1 
        wins += 1
    
    #check if it's perform well
    if episode % printEvery == 0:
        print("Episode %i, Mean Return %.3f, Win prob %.2f" % ((episode + 1), meanReturn, wins / (episode + 1) ))
    

pickle.dump(Q, open('Q.dill', 'wb'))
_ = pickle.load(open('Q.dill', 'rb')) #sanity check

#plot answer
utils.plot(Q, [0,1])
