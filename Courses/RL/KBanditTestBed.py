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


def test():
    problems = []
    n = 2000
    runs = 10000
    k = 10
    for i in range(n):
        env = KBanditEnv(k)
        agent = SimpleAvgBanditAgent(env, epsilon=0.1)
        problems.append((env, agent))
        

    for i in range(runs):
        totalReward = 0.0
        totalOptimal = 0.0
        for problem in problems:
            env = problem[0]
            agent = problem[1]
            totalReward += agent.action()
            if env.isLastActionOptimal:
                totalOptimal += 1
            
        print('{}\t{}\t{}'.format(i, totalReward/n, totalOptimal/n))

if __name__ == '__main__':
    test()






         

    

            


    


        



    



    

