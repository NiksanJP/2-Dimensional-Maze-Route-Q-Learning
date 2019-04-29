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
        return [self.stateX, self.stateY]
    
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
        self.learningRate = 0.01
        self.discountRate = 0.9
        self.epsilon = 0.9
        self.actions = ['up', 'down', 'left','right']
        self.boardSize = [6,6]
        
        self.states = 26
        
        self.table = self.createTable()
        
    def createTable(self):
        table = pd.DataFrame(
            np.zeros((self.states, len(self.actions))),
            columns=self.actions
        )
        return table
    
    def learn(self, state, action, reward, new_state):
        state = state[0] * state[1]
        new_state = new_state[0] * new_state[1]
        action = action.index(action)
        predict = self.table.iloc[state, action]
        target = reward + self.epsilon * np.max(self.table.iloc[new_state, :])
        self.table.iloc[state, action] = self.table.iloc[state, action] + self.learningRate*(target - predict)
        
    def getTable(self):
        return self.table
    
    def chooseAction(self, state):
        table = self.getTable()
        b_state = state
        state = state[0] * state[1]
        state_actions = table.iloc[state, :]
        if (np.random.uniform() < self.epsilon) or ((state_actions == 0).all()):
            possibleAction = []
            if b_state[0] == 0:
                possibleAction =['up', 'down', 'right']
            if b_state[0] == 5:
                possibleAction =['up', 'down', 'left']
            if b_state[1] == 0:
                possibleAction =['down', 'left','right']
            if b_state[1] == 5:
                possibleAction =['up', 'left','right']
            
            try:
                action = np.random.choice(possibleAction)
            except:
                action = np.random.choice(self.actions)
                 
            print("RANDOM ACTION = ", action)
        else:
            action = np.argmax(self.table.iloc[state, :])
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
        
            
env = environment()
agent = agent()
total = 0
minsteps = 300
for episode in range(30):
    state = env.getState()
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
        