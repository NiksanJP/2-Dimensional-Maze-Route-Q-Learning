import numpy as np
import pandas as pd
import time

class environment:
    def __init__(self):
        self.Width = 10
        self.Height = 10
        #agent Position
        self.agentPositionX = 0
        self.agentPositionY = 0
        
        #Goal Position
        self.goalPositionX = 5
        self.goalPositionY = 5
        
        #State of the game
        self.stateX = self.agentPositionX - self.goalPositionX
        self.stateY = self.agentPositionY - self.goalPositionY
        
        #Action
        self.action = ['left', 'right', 'up', 'down']
        
        #board
    
    def getState(self):
        #State of the game
        self.stateX = self.goalPositionX - self.agentPositionX
        self.stateY = self.goalPositionY - self.agentPositionY
        return [self.agentPositionX, self.agentPositionY]
    
    def printState(self):
        board = ["","","","","",""]
        for x in range(6):
            for y in range(6):
                if self.agentPositionX == x and self.agentPositionY == y:
                    board[x] += "O"
                elif self.goalPositionX == x and self.goalPositionY == y:
                    board[x] += "G"
                else:
                    board[x] += "-"
        for i in range(6):
            print(board[i])
        
    def environmentReact(self, action):
        Reward = 0
        if action == "up":
            if self.agentPositionX != 0:
                self.agentPositionX -= 1
        elif action == "down":
            if self.agentPositionX != 5:
                self.agentPositionX += 1
        elif action == "left":
            if self.agentPositionY != 0:
                self.agentPositionY -=1
        elif action == "right":
            if self.agentPositionY != 5:
                self.agentPositionY += 1
        
        if self.agentPositionX == self.goalPositionX and self.agentPositionY == self.goalPositionY:
            Reward = 1
        
        return Reward, self.getState()
    
    def reset(self):
        self.__init__()
        
class agent:
    def __init__(self):
        self.learningRate = 0.8
        self.discountRate = 0.85
        self.epsilon = 1.0
        self.actions = ['up', 'down', 'left','right']
        self.boardSize = [6,6]
        
        self.table = self.createTable()
        
        self.maxEpsilon = 1.0
        self.minEpsilon = 0.01
        self.decayRate = 0.005
        
    def createTable(self):
        table = pd.DataFrame(
            columns=self.actions, dtype=np.float64
        )
        return table
    
    def learn(self, state, action, reward, new_state):
        state = str(state)
        new_state = str(new_state)

        if action == 0:
            action = "up"
        elif action == 1:
            action = "down"
        elif action == 2:
            action = "right"
        elif action == 3:
            action = "left"
        
        self.checkObservationExistance(new_state)

        predict = self.table.loc[state, action]
        target = reward + self.epsilon * np.max(self.table.loc[new_state, :])
        self.table.loc[state, action] = self.table.loc[state, action] + self.learningRate*(target - predict)
        
    def getTable(self):
        return self.table
    
    def changeEpsilon(self, episdoe):
        self.epsilon = self.minEpsilon + ( self.maxEpsilon - self.minEpsilon) * np.exp(-self.decayRate * episdoe)
    
    def chooseAction(self, state):
        self.checkObservationExistance(state)
        state = str(state)
        if (np.random.uniform(0, 1) < self.epsilon) :
            action = np.random.choice(self.actions) 
            print("RANDOM ACTION = ", action)
            
        else:
            action = np.argmax(self.table.loc[state, :])
            if action == 0:
                action = "up"
            elif action == 1:
                action = "down"
            elif action == 2:
                action = "right"
            elif action == 3:
                action = "left"
            
            print("CHOOSEN ACTION = ", action)
        return action   
    
    def checkObservationExistance(self, observation):
        if str(observation) not in self.table.index:
            self.table = self.table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.table.columns,
                    name = str(observation)
                )
            )     
            
        
env = environment()
agent = agent()
total = 0
minsteps = 300
for episode in range(30):
    state = env.getState()
    if episode != 0:
        agent.changeEpsilon(episode)
    for steps in range(100):    
        interaction = 'Episode %s: total_steps = %s  TOTAL REWARDs = %s  Minimum Steps = %s' % (episode+1, steps, total, minsteps)
        print('\r{}'.format(interaction), end='')
        print()
        env.printState()
        
        action = agent.chooseAction(state)
        
        reward, new_state = env.environmentReact(action)
        
        print("REWARD : ", reward)
        agent.learn(state, action, reward, new_state)
        
        state = new_state
        time.sleep(0.005)
        print(agent.getTable())
        
        if reward == 1:
            env.reset()
            if minsteps > steps:
                minsteps = steps
            time.sleep(0.5)
            total +=1
            print('\r                                ', end='')
            break        