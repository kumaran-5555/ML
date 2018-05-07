import sys
import os
import numpy as np
import collections
import enum

import copy

class MDP:
    def __init__(self, states, actions, dynamics, policy, terminal):
        self.states = states
        self.actions = actions
        self.dynamics = dynamics
        self.terminal = terminal
        self.policy = policy
        self.value = collections.defaultdict(lambda: 0.0)

        # for every state, the distribution over actions should be valid         
        for s in self.states:
            if s == self.terminal:
                continue
            
            prob = 0.0
            for _, p in self.policy[s]:
                prob += p

            if prob != 1.0:
                raise ValueError('Invalid policy, for state {} total probability {} is not 1.0'.format(s, prob))

        # for every state, action, the distribution over terminal state and rewards should be valid
        for s in self.states:
            if s == self.terminal:
                    continue
                
            for a in self.actions:
                prob = 0.0
                for _,_,p in self.dynamics[(s, a)]:
                    prob += p
                if prob != 1.0:
                    raise ValueError('Invalid dynamics, for state {} action {} total probability {} is not 1.0'.format(s, a, prob))
        
    
    def policyEvaluation(self, gamma=0.999999):
        while True:
            delta = 0.0
            v = copy.deepcopy(self.value)
            for s in self.states:
                if s == self.terminal:
                    continue

                old = v[s]

                new = 0.0
                for action, actionProb in self.policy[s]:
                    for endState, reward, prob in self.dynamics[(s, action)]:
                        new += (actionProb * prob * (reward + gamma * v[endState]))
                
                self.value[s] = new

                delta = max(delta, abs(old-new))

            if delta < 1e-4:
                break
            

        return self.value



        

class Policy:
    def __init__(self, states, actions):
        self.dynamics = collections.defaultdict(list)
        self.states = states
        self.actions = actions

    def add(self, state, action, probability):
        if not isinstance(state, self.states):
            raise ValueError('Invalid state state {}, not in state Enum'.format(state))

        if not isinstance(action, self.actions):
            raise ValueError('Invalid action {}, not in action Enum'.format(action))

        currentProd = 0.0
        for _, p  in self.dynamics[state]:
            currentProd +=  p

        if currentProd + probability > 1.0:
            raise ValueError('Invalid probabilty, total is > 1.0')

        self.dynamics[state].append((action, probability))

    def __getitem__(self, key):
        return self.dynamics[key]

class Dynamics:
    def __init__(self, states, actions):
        self.dynamics = collections.defaultdict(list)
        self.states = states
        self.actions = actions

    def __getitem__(self, key):
        return self.dynamics[key]
        
    def add(self, startState, action, reward, endState, probability):
        if not isinstance(startState, self.states):
            raise ValueError('Invalid state state {}, not in state Enum'.format(startState))

        if not isinstance(endState, self.states):
            raise ValueError('Invalid state {},not in state Enum'.format(endState))

        if not isinstance(action, self.actions):
            raise ValueError('Invalid action {}, not in action Enum'.format(action))

        currentProd = 0.0
        for _, _, p  in self.dynamics[(startState, action)]:
            currentProd +=  p

        if currentProd + probability > 1.0:
            raise ValueError('Total probaiblity of start {}  action {} > 1.0'.format(startState, action))

        self.dynamics[(startState, action)].append((endState, reward, probability))


def simpleGrid():
    class States(enum.Enum):
        TERMINAL = 15
        ONE = 1
        TWO = 2
        THREE = 3
        FOUR = 4
        FIVE = 5
        SIX = 6
        SEVEN = 7 
        EIGHT = 8
        NINE = 9
        TEN = 10
        ELEVEN = 11
        TWELVE = 12
        THIRTEEN = 13
        FOURTEEN = 14

    class Actions(enum.Enum):
        LEFT = 1
        RIGHT = 2
        UP = 3
        DOWN = 4

    
    dyn = Dynamics(States, Actions)
    
    grid = []
    grid.append([15,1,2,3])
    grid.append([4,5,6,7])
    grid.append([8,9,10,11])
    grid.append([12,13,14,15])

    r = len(grid)
    c = len(grid[0])
    for i in range(r):
        for j in range(c):
            state = grid[i][j]
            if States(state) == States.TERMINAL:
                continue

            # right
            if i+1 >=c:
                dyn.add(States(state), Actions.RIGHT, -1, States(state), 1)
            else:
                dyn.add(States(state), Actions.RIGHT, -1, States(grid[i+1][j]), 1)
            # left 
            if i-1 < 0:
                dyn.add(States(state), Actions.LEFT, -1, States(state), 1)
            else:
                dyn.add(States(state), Actions.LEFT, -1, States(grid[i-1][j]), 1)

            # up
            if j-1 < 0:
                dyn.add(States(state), Actions.UP, -1, States(state), 1)
            else:
                dyn.add(States(state), Actions.UP, -1, States(grid[i][j-1]), 1)

            # up
            if j+1 >= r:
                dyn.add(States(state), Actions.DOWN, -1, States(state), 1)
            else:
                dyn.add(States(state), Actions.DOWN, -1, States(grid[i][j+1]), 1)

    policy = Policy(States, Actions)
    for s in States:
        for a in Actions:
            # equi probable actions
            probability =  1.0/ len(Actions)
            policy.add(s, a, probability)

    mdp = MDP(States, Actions, dyn, policy, States.TERMINAL)
    mdp.policyEvaluation()




    


if __name__ == '__main__':
    simpleGrid()




