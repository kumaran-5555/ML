import sys
import os
import numpy as np
import collections
import enum

import copy

class MDP:
    def __init__(self, states, actions, dynamics, policy, terminal, gamma):                
        self.states = states
        self.actions = actions
        self.dynamics = dynamics
        self.terminal = terminal
        self.policy = policy
        self.value = collections.defaultdict(lambda: 0.0)
        self.gamma = gamma

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

    def policyEvaluation(self, policy=None): 
        if policy is None:            
            policy = self.policy.actionDistrib()

        i = 0
        while True:
            i+=1
            delta = 0.0            
            for s in self.states:
                if s == self.terminal:
                    continue

                old = self.value[s]
                new = self.v(s, policy)

                self.value[s] = new
                delta = max(delta, abs(old-new))

            if delta < 1e-2:
                break            

        return self.value




    def v(self, state, policy):
        '''
            Evaluates v(s) for the current policy and value
        '''
        ret = 0.0
        for action, actionProb in policy[state]:
            for endState, reward, prob in self.dynamics[(state, action)]:
                ret += (actionProb * prob * (reward + self.gamma * self.value[endState]))
        
        return ret
        



    def q(self, state, action):
        '''
            Evaluates q(s,a) for the current value function
        '''
        ret = 0
        for endState, reward, prob in self.dynamics[(state, action)]:
            ret +=(prob * (reward + self.gamma * self.value[endState]))

        return ret



    def policyImprovement(self, policy):
        isStable = True
        newPolicy = {}
        for s in self.states:
            if s == self.terminal:
                continue

            old, _ = policy[s][0]
            qValue = []
            for a in self.actions:
                qValue.append((a, self.q(s, a)))

            qValue = sorted(qValue, key=lambda x: x[1], reverse=True)

            new = qValue[0][0]

            if old != new:
                isStable = False
                policy[s] = [(new,1.0)]

            newPolicy[s] = [(new,1.0)]

        return isStable


    def policyIteration(self):
        policy = self.policy.actionSample()
        while True:            
            # learn better value estimates by keeping policy constant
            self.policyEvaluation(policy=policy)

            # learn better policy by keeping value estimates constant
            isStable = self.policyImprovement(policy)

            if isStable:
                break

class Policy:
    def __init__(self, states, actions):
        self.prob = collections.defaultdict(list)
        self.action = collections.defaultdict(list)
        self.states = states
        self.actions = actions

    def add(self, state, action, probability):
        if state not in self.states:
            raise ValueError('Invalid state state {}, not in state Enum'.format(state))

        if action not in self.actions:
            raise ValueError('Invalid action {}, not in action Enum'.format(action))

        
        if sum(self.prob[state]) + probability > 1.0:
            raise ValueError('Invalid probabilty, total is > 1.0')

        self.prob[state].append(probability)
        self.action[state].append(action)

    def actionDistrib(self):
        '''
            return full action distribution for each state
        '''
        ret = {}
        for s in self.states:
            ret[s] = [(self.action[s][i], self.prob[s][i]) for i in range(len(self.action[s]))]
        return ret
    
    def actionSample(self):
        '''
            returns single action sampled from the distribution for each state
        '''
        ret = {}
        for s in self.states:
            ret[s] = [(np.random.choice(self.action[s], p=self.prob[s]), 1.0)]

        return ret

    def update(self, state, actions, probs):
        if sum(probs) != 1.0:
            raise ValueError('Invalid probabilty, total is != 1.0')

        self.action[state] = actions
        self.prob[state] = probs

    
    
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
            if j+1 >=c:
                dyn.add(States(state), Actions.RIGHT, -1, States(state), 1)
            else:
                dyn.add(States(state), Actions.RIGHT, -1, States(grid[i][j+1]), 1)
            # left 
            if j-1 < 0:
                dyn.add(States(state), Actions.LEFT, -1, States(state), 1)
            else:
                dyn.add(States(state), Actions.LEFT, -1, States(grid[i][j-1]), 1)

            # up
            if i-1 < 0:
                dyn.add(States(state), Actions.UP, -1, States(state), 1)
            else:
                dyn.add(States(state), Actions.UP, -1, States(grid[i-1][j]), 1)

            # down
            if i+1 >= r:
                dyn.add(States(state), Actions.DOWN, -1, States(state), 1)
            else:
                dyn.add(States(state), Actions.DOWN, -1, States(grid[i+1][j]), 1)

    policy = Policy(States, Actions)
    for s in States:
        for a in Actions:
            # equi probable actions
            probability =  1.0/ len(Actions)
            policy.add(s, a, probability)

    mdp = MDP(States, Actions, dyn, policy, States.TERMINAL, 0.9)
    #mdp.policyEvaluation()
    mdp.policyIteration()

    for i in range(r):
        for j in range(c):
            state = States(grid[i][j])
            print(mdp.value[state], ' ', end='')
        print()
        


        

    


if __name__ == '__main__':
    simpleGrid()




