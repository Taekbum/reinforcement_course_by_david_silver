import numpy as np

class Easy21():

    def __init__(self):
        self.minCardValue, self.maxCardValue = 1, 10
        self.dealerUpperBound = 17
        self.gameLowerBound, self.gameUpperBound = 1, 21

    def actionSpace(self):
        return (0, 1)

    def initGame(self):  #start with black card
        return (np.random.randint(self.minCardValue, self.maxCardValue+1), np.random.randint(self.minCardValue, self.maxCardValue+1))

    def draw(self):  
        value = np.random.randint(self.minCardValue, self.maxCardValue+1)

        if np.random.random() <= 1/3:  #red card
            return -value
        else:  #black card
            return value

    def step(self, playerValue, dealerValue, action):
        assert action in [0, 1], "Action must be in [0, 1]"
        terminate = False

        if action == 0:  #hit
            playerValue += self.draw()

            if self.gameLowerBound <= playerValue <= self.gameUpperBound:
                reward = 0
            else:
                reward = -1
                terminate = True
        
        elif action == 1:  #stick
            terminate = True
            
            #if player sticks, dealer hits until reaches 17 or more
            while self.gameLowerBound <= dealerValue < self.dealerUpperBound:
                dealerValue += self.draw()
                
            if not (self.gameLowerBound <= dealerValue <= self.gameUpperBound) or (playerValue > dealerValue):
                reward = 1
            elif playerValue == dealerValue:
                reward = 0
            elif playerValue < dealerValue:
                reward = -1
            
        return playerValue, dealerValue, reward, terminate