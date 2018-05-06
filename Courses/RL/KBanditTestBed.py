import numpy
import os



class KBanditEnv:
    def __init__(self, k, qStarMean=0.0, qStarVariance=1.0, rewardVariance=1.0):
        self.k = k
        self.qStar = numpy.random.normal(qStarMean, qStarVariance, self.k)
        self.rewardVariance = rewardVariance

        self.stats = {}
        self.stats['NumOfOptimal'] = 0
        self.isLastActionOptimal = False
        
    
    def interaction(self, actionNo):
        if actionNo < 1 or actionNo > self.k:
            raise ValueError('Invalid action {}'.format(actionNo))

        
        if actionNo - 1 == numpy.argmax(self.qStar):
            self.stats['NumOfOptimal'] += 1
            self.isLastActionOptimal = True
        else:
            self.isLastActionOptimal = False


        return numpy.random.normal(self.qStar[actionNo-1], self.rewardVariance)

class NonStationaryKBanditEnv:
    def __init__(self, k, qStarMean=0.0, qStarVariance=1.0, rewardVariance=1.0, walkVariance=0.01):
        self.k = k
        self.qStar = numpy.random.normal(qStarMean, qStarVariance, self.k)
        self.rewardVariance = rewardVariance

        self.stats = {}
        self.stats['NumOfOptimal'] = 0
        self.isLastActionOptimal = False
        self.walkVariance = walkVariance
        
    
    def interaction(self, actionNo):
        if actionNo < 1 or actionNo > self.k:
            raise ValueError('Invalid action {}'.format(actionNo))

        
        if actionNo - 1 == numpy.argmax(self.qStar):
            self.stats['NumOfOptimal'] += 1
            self.isLastActionOptimal = True
        else:
            self.isLastActionOptimal = False


        reward = numpy.random.normal(self.qStar[actionNo-1], self.rewardVariance)

        # update
        self.qStar += numpy.random.normal(0, self.walkVariance, self.k)
        
        return reward




class SimpleAvgBanditAgent:
    def __init__(self, kBanditEnv, epsilon=0.0):
        self.env = kBanditEnv
        self.k = self.env.k
        self.qEst = numpy.zeros(self.k)
        self.eps = epsilon
        self.n = numpy.zeros(self.k)

        self.stats = {}
        self.stats['TotalRewards'] = 0
        self.stats['NumOfInteractions'] = 0

    
    def action(self):
        pExploit = numpy.random.random_sample()

        # exploit
        if pExploit < 1-self.eps:
            actionNo = numpy.argmax(self.qEst)           
        
        # explore
        else:
            actionNo = numpy.random.randint(0, self.k)

        reward = self.env.interaction(actionNo+1)
        self.n[actionNo] += 1
        # update qEst
        self.qEst[actionNo] = self.qEst[actionNo] + ((1.0/self.n[actionNo])* (reward - self.qEst[actionNo]))

        self.stats['TotalRewards'] += reward
        self.stats['NumOfInteractions'] += 1
        return reward

class WeightedAvgBanditAgent:
    def __init__(self, kBanditEnv, epsilon=0.0, alpha=0.1):
        self.env = kBanditEnv
        self.k = self.env.k
        self.qEst = numpy.zeros(self.k)
        self.eps = epsilon
        self.n = numpy.zeros(self.k)
        self.alpha = alpha

        if self.alpha < 0 or self.alpha > 1:
            raise ValueError('Invalid value range for alpha {}'.format(self.alpha))

        self.stats = {}
        self.stats['TotalRewards'] = 0
        self.stats['NumOfInteractions'] = 0

    
    def action(self):
        pExploit = numpy.random.random_sample()

        # exploit
        if pExploit < 1-self.eps:
            actionNo = numpy.argmax(self.qEst)           
        
        # explore
        else:
            actionNo = numpy.random.randint(0, self.k)

        reward = self.env.interaction(actionNo+1)
        self.n[actionNo] += 1
        # update qEst
        self.qEst[actionNo] = self.qEst[actionNo] + ((self.alpha)* (reward - self.qEst[actionNo]))

        self.stats['TotalRewards'] += reward
        self.stats['NumOfInteractions'] += 1
        return reward


outputDir = r'E:\\Temp\\'



def stationarySimple():
    problems = []
    n = 2000
    runs = 10000
    k = 10
    for i in range(n):
        env = KBanditEnv(k)
        agent = SimpleAvgBanditAgent(env, epsilon=0.1)
        problems.append((env, agent))
        
    output = open(os.path.join(outputDir, 'stationarySimple.tsv'), 'w')
    for i in range(runs):
        totalReward = 0.0
        totalOptimal = 0.0
        for problem in problems:
            env = problem[0]
            agent = problem[1]
            totalReward += agent.action()
            if env.isLastActionOptimal:
                totalOptimal += 1
            
        output.write('{}\t{}\t{}\n'.format(i, totalReward/n, totalOptimal/n))

def stationaryWeighted():
    problems = []
    n = 2000
    runs = 10000
    k = 10
    for i in range(n):
        env = KBanditEnv(k)
        agent = WeightedAvgBanditAgent(env, epsilon=0.1, alpha=0.1)
        problems.append((env, agent))
        
    output = open(os.path.join(outputDir, 'stationaryWeighted.tsv'), 'w')
    for i in range(runs):
        totalReward = 0.0
        totalOptimal = 0.0
        for problem in problems:
            env = problem[0]
            agent = problem[1]
            totalReward += agent.action()
            if env.isLastActionOptimal:
                totalOptimal += 1
            
        output.write('{}\t{}\t{}\n'.format(i, totalReward/n, totalOptimal/n))

def nonStationarySimple():
    problems = []
    n = 2000
    runs = 10000
    k = 10
    for i in range(n):
        env = NonStationaryKBanditEnv(k)
        agent = SimpleAvgBanditAgent(env, epsilon=0.1)
        problems.append((env, agent))
        
    output = open(os.path.join(outputDir, 'nonStationarySimple.tsv'), 'w')
    for i in range(runs):
        totalReward = 0.0
        totalOptimal = 0.0
        for problem in problems:
            env = problem[0]
            agent = problem[1]
            totalReward += agent.action()
            if env.isLastActionOptimal:
                totalOptimal += 1
            
        output.write('{}\t{}\t{}\n'.format(i, totalReward/n, totalOptimal/n))

def nonStationaryWeighted():
    problems = []
    n = 2000
    runs = 10000
    k = 10
    for i in range(n):
        env = NonStationaryKBanditEnv(k)
        agent = WeightedAvgBanditAgent(env, epsilon=0.1, alpha=0.1)
        problems.append((env, agent))
        
    output = open(os.path.join(outputDir, 'nonStationaryWeighted.tsv'), 'w')
    for i in range(runs):
        totalReward = 0.0
        totalOptimal = 0.0
        for problem in problems:
            env = problem[0]
            agent = problem[1]
            totalReward += agent.action()
            if env.isLastActionOptimal:
                totalOptimal += 1
            
        output.write('{}\t{}\t{}\n'.format(i, totalReward/n, totalOptimal/n))

if __name__ == '__main__':
    #stationarySimple()
    #nonStationarySimple()
    stationaryWeighted()
    nonStationaryWeighted()
    

