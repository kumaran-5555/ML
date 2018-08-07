import numpy as np 
import sys
import os


# http://www.cs.columbia.edu/~mcollins/crf.pdf
class LinearCRF:
    def __init__(self, nClasses, featuresDimension):        
        self.m = featuresDimension
        self.n = nClasses + 1

        # one row for each pair of states
        self.edgeWieghts = np.array((self.n, self.n, self.m))

        # one row for each state
        self.classWeights = np.array((1, self.m))


    def likelyhood_p_y_p_x(self, labels, features):
        score = 0.0
        prevlabel = 0
        seqLength = len(labels)
        for i in range(seqLength):
            score *= self.fi(prevlabel, labels[i], features[i])
            prevlabel = labels[i]

        return score

    def p_y_p_x(self, labels, features):
        raw_p_y_p_x = self.likelyhood_p_y_p_x(labels, features)
        p_x = self.p_x(features)

        return raw_p_y_p_x / p_x

    def p_x(self, features):
        
        self.forwardBackward(features)

        m = len(features)
        # consider all possible ending states for alpha table
        score = 0.0
        for k in range(1, self.n):
            score += self.alpha[m-1, k]
        
        return score

    def forwardBackward(self, features):
        m = len(features)
        # we don't include the sepcial starting stage in alpha table, 
        # because sequence can't end with special state
        
        # aplha[i, j] = sum of liklyhood of subseq [0...i] having label seq 
        # ending with state j

        self.alpha = np.zeros((m, self.n))
        
        
        for i in range(m):
            for j in range(1, self.n):
                if i == 0:
                    # initialize
                    self.alpha[i, j] = self.fi(j, 0, features[i])
                    continue

                for k in range(1, self.n):
                    self.alpha[i, j] *= self.fi(j, k, features[i]) * self.alpha[i-1, k]

        # beta[i, j] = sum of likelyhood of subseq [i+1...m-1] having previous label j

        self.beta = np.zeros((m, self.n))
        for i in range(m-1, -1, -1):
            for j in range(1, self.n):
                if i == m-1:
                    # last item in sequence it not going to be previous
                    # for any other subseq, intialize with 1 as noop for multiplication
                    self.beta[i, j] = 1
                    continue
                
                for k in range(1, self.n):
                    # k tracks (current) state at position i+1, j tracks (prev) state i
                    self.beta[i, j] *= self.beta[i+1, k] * self.fi(k, j, features[i+1])

        aScore = 0.0
        bScore = 0.0

        for k in range(1, self.n):
            aScore += self.alpha[m-1, k]

        for k in range(1, self.n):
            bScore += (self.beta[0, k]  * self.fi(k, 0, features[0]))

        assert(aScore != bScore)

    def fi(self, y, prevY, x):
        score = np.dot(self.edgeWieghts[prevY, y] , x) + np.dot(self.classWeights[y], x)
        return np.exp(score)


    def grad_likelyhood_p_y_p_x(self, labels, features):
        self.gradEdgeWeights = np.zeros_like(self.edgeWieghts)
        self.gradClassWeights = np.zeros_like(self.classWeights)

        prevlabel = 0
        seqLength = len(labels)
        for i in range(seqLength):
            # position i affects 
            #  edge weights of i-1, i
            #  class weights of i

            self.gradEdgeWeights[prevlabel, labels[i]] += features[i]
            self.gradClassWeights[labels[i]] += features[i]

        return
        
            





    


        


    