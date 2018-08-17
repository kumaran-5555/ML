import math
import numpy as np 
import sys
import os
from nltk.tag.util import untag
import nltk
import collections
import scipy
import time
import shutil
import pickle


# http://www.cs.columbia.edu/~mcollins/crf.pdf
class LinearCRF:
    def __init__(self, nClasses, featuresDimension):        
        self.d = featuresDimension
        self.n = nClasses + 1

        # one row for each pair of states
        #self.edgeWieghts = np.random.normal(size=(self.n, self.n, 1, self.d), scale=0.1)
        self.edgeWieghts = np.zeros((self.n, self.n, 1, self.d), dtype=np.float32)
        # one row for each state
        #self.classWeights = np.random.normal(size=(self.n, 1, self.d), scale=0.1)
        self.classWeights = np.zeros((self.n, 1, self.d), dtype=np.float32)
        self.alpha = None
        self.beta = None
        self.scores = None
        self.fiCache = {}
        self.sampleInCurrentBatch = 0
        
        self.gE = np.zeros_like(self.edgeWieghts)
        self.gW = np.zeros_like(self.classWeights)

    @staticmethod
    def dot(w, x):
        return np.sum(w[0][x.row])
        
    @staticmethod
    def sum(w, x):
        w[0][x.col] +=  x.data

    @staticmethod
    def sumMul(w, x, c):
        w[0][x.row] = np.add(w[0][x.row], c)

    @staticmethod
    def sum2(w, x):
        w[0][x.row] += x.data

    @staticmethod
    def logsumexp(a):
        b = a.max()
        return b + np.log((np.exp(a-b)).sum())

    def potentials(self, labels, features):
        m = len(labels)-1
        self.scores = np.zeros((m+1, self.n, self.n), dtype=np.float32)
        for i in range(1, m+1):
            if i == 1:
                a = 0
                for b in range(1, self.n):
                    self.scores[i, a, b] = self.fi(a, b, i, features)
                continue

            for a in range(1, self.n):
                for b in range(1, self.n):
                    self.scores[i, a, b] = self.fi(a, b, i, features)

        return

                





    def score(self, labels, features):        
        m = len(labels)-1
        score = 0.0
        for i in range(1, m+1):
            if i==1:
                score = self.scores[i, 0, labels[i]]
                continue

            score += self.scores[i, labels[i-1], labels[i]]
        return score

    def pOfYGivenX(self, labels, features):
        scoreOfYGivenX = self.score(labels, features)
        ZOfX = self.ZOfX(features)
        if scoreOfYGivenX > ZOfX:
            raise ValueError
        return scoreOfYGivenX - ZOfX

    def ZOfX(self, features):                
        m = len(features)-1
        # consider all possible ending states for alpha table
        return LinearCRF.logsumexp(self.alpha[1:,m])

    
    def forwardBackward(self, features):
        m = len(features)-1
        # we don't include the sepcial starting stage in alpha table, 
        # because sequence can't end with special state
        
        # aplha[i, j] = sum of liklyhood of subseq [0...i] having label seq 
        # ending with state j        
        self.alpha = np.zeros((self.n, m+1))
        self.alpha[0, 0] = 0
        
        for i in range(1, m+1):
            for s in range(1, self.n):
                if i == 1:
                    self.alpha[s, i] = self.scores[i, 0, s]
                    continue
                    
                # when we have to sum across sequences, we need to go to exp space
                # for each sequence and apply sum in the exp space and come back to log space
                # log(p1 + p2 + p3) is what we want from logP1, logP2, logP3
                # log(p1 + p2 + p3) = log(exp(logP1) + exp(logP2) + exp(logP3))
                self.alpha[s, i] = LinearCRF.logsumexp(self.alpha[1:, i-1] + self.scores[i, 1:, s])
                
        # beta[i, j] = sum of likelyhood of subseq [i+1...m-1] having previous label j

        
        self.beta = np.zeros((self.n, m+1))
        for i in range(m, 0, -1):
            for s in range(1, self.n):
                if i == m:
                    # last item in sequence it not going to be previous
                    # for any other subseq, intialize with 1 as noop for multiplication
                    self.beta[s, i] = 0
                    continue                                
                self.beta[s, i] = LinearCRF.logsumexp(self.beta[1:, i+1] + self.scores[i+1, s, 1:])
                    

    def fi(self, prevY, y, i, features):        
        x = features[i]        
        score = LinearCRF.dot(self.edgeWieghts[prevY, y], x) + LinearCRF.dot(self.classWeights[y], x)
        return score

    def loss(self, labels, features):
        return self.pOfYGivenX(labels, features)

    def learn(self, labels, features, learningRate=0.15, alpha=0.002, batchSize = 5):
        # add dummy place holders to make the indcies from 1...m
        # add a dummy rows to label and features
        labels = [None] + labels
        features = [None] + features

        self.potentials(labels, features)
        self.forwardBackward(features)
        

        self.sampleInCurrentBatch += 1
        
        # initialize gradients with gradients of L2 regualization
        if batchSize == 1:
            self.gW = alpha * self.classWeights
            self.gE = alpha * self.edgeWieghts
        else:
            self.gW += alpha * self.classWeights
            self.gE += alpha * self.edgeWieghts       

        self.updateGradientForFirstTerm(labels, features)
        self.updateGradientForSecondTerm(features)
        #self.updateGradientOfL2Penalty(alpha)
        
        
        
        l = self.loss(labels, features)
        p = np.exp(self.pOfYGivenX(labels, features))
        z = self.ZOfX(features)
        print(l, p, z)

        # we will update weights only once per batch
        if self.sampleInCurrentBatch == batchSize:
            # gradient descent
            np.add(self.edgeWieghts, (learningRate * self.gE), out=self.edgeWieghts)
            np.add(self.classWeights, (learningRate * self.gW), out=self.classWeights)
            if batchSize != 1:
                self.gE.fill(0.0)
                self.gW.fill(0.0)

            self.sampleInCurrentBatch = 0
        return l
        

    def updateGradientForFirstTerm(self, labels, features):
        m = len(features)-1
        for i in range(1, m+1):
            LinearCRF.sum2(self.gW[labels[i]], features[i])
            if i==1:
                LinearCRF.sum2(self.gE[0, labels[i]], features[i])
                continue
            LinearCRF.sum2(self.gE[labels[i-1], labels[i]], features[i])

    def updateGradientForSecondTerm(self, features):
        m = len(features) - 1        
        ZOfXScore = self.ZOfX(features)
        for i in range(1, m+1):
            if i==1:
                a = 0
                for b in range(1, self.n):
                    q_i_a_b = -1 * np.exp((self.alpha[a, i-1] + self.scores[i, a, b] + self.beta[b, i])  - ZOfXScore)
                    LinearCRF.sumMul(self.gW[b], features[i], q_i_a_b)
                    LinearCRF.sumMul(self.gE[a, b], features[i], q_i_a_b)
                continue

            for a in range(1, self.n):                
                for b in range(1, self.n):                    
                    
                    q_i_a_b = -1 * np.exp((self.alpha[a, i-1] + self.scores[i, a, b] + self.beta[b, i])  - ZOfXScore)
                    
                    LinearCRF.sumMul(self.gW[b], features[i], q_i_a_b)
                    
                    LinearCRF.sumMul(self.gE[a, b], features[i], q_i_a_b)

        

    def updateGradientOfL2Penalty(self, alpha):
        self.gW += alpha * self.classWeights
        self.gE += alpha * self.edgeWieghts


# we will use features from  https://nlpforhackers.io/crf-pos-tagger/
def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        #'is_first': index == 0,
        #'is_last': index == len(sentence) - 1,
        #'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        #'is_all_caps': sentence[index].upper() == sentence[index],
        #'is_all_lower': sentence[index].lower() == sentence[index],
        #'prefix-1': sentence[index][0],
        #'prefix-2': sentence[index][:2],
        #'prefix-3': sentence[index][:3],
        #'suffix-1': sentence[index][-1],
        #'suffix-2': sentence[index][-2:],
        #'suffix-3': sentence[index][-3:],
        #'prev_word': '' if index == 0 else sentence[index - 1],
        #'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        #'has_hyphen': '-' in sentence[index],
        #'is_numeric': sentence[index].isdigit(),
        #'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }

def vectorizeLabels(labels):
    labelNames = {}

    for row in labels:
        for l in row:
            if l in labelNames:
                continue
            
            labelNames[l] = len(labelNames)

    vectorizedLabels = []
    for row in labels:        
        vectorizedLabel = []
        for l in row:
            # we want ids of labels to start from 1
            vectorizedLabel.append(labelNames[l]+1)
        vectorizedLabels.append(vectorizedLabel)

    return vectorizedLabels, len(labelNames)

        

def vectorizeFeatures(data):
    featureDimesntion = collections.defaultdict(lambda : collections.defaultdict())

    for row in data:
        for index in row:
            for f,v in index.items():
                if isinstance(v, (float)):
                    raise ValueError('float values are not accepted  {} {}'.format(f, v))            
                if v in featureDimesntion[f]:
                    continue

                featureDimesntion[f][v] = len(featureDimesntion[f])

    offsets = {}
    offset = 0
    
    for f in featureDimesntion:
        offsets[f] = offset
        offset += len(featureDimesntion[f])
        

    vectorizedData = []
    for row in data:
        features = []
        for index in row:
            indexFeatures = []
            for f in featureDimesntion:
                if f not in index:
                    continue
                
                v = index[f]
                indexFeatures.append(offsets[f] + featureDimesntion[f][v])
            
            # transform to sparse vector 
            indexFeatures = scipy.sparse.coo_matrix(([1]*len(indexFeatures), ([0]*len(indexFeatures), indexFeatures)), shape=(1, offset), dtype=np.float32).T
            features.append(indexFeatures)

        vectorizedData.append(features)

    return vectorizedData, offset

            

def trainPosTagger():

    fileName = r'E:\Temp\treebank.pkl'
    if not os.path.exists(fileName):
        tagged_sentences = nltk.corpus.treebank.tagged_sents()
        # Split the dataset for training and testing        
        
        def transform_to_dataset(tagged_sentences):
            x, y = [], []
        
            for tagged in tagged_sentences:
                x.append([features(untag(tagged), index) for index in range(len(tagged))])
                y.append([tag for _, tag in tagged])
        
            return x, y
        
        x, y = transform_to_dataset(tagged_sentences)

        x, d = vectorizeFeatures(x)
        y, nClasses = vectorizeLabels(y)
        
        pickle.dump((x,y,d,nClasses), open(fileName, 'wb'))

    else:
        x, y, d, nClasses = pickle.load(open(fileName, 'rb'))

    cutoff = int(.75 * len(x))
    xTrain = x[:cutoff]
    yTrain = y[:cutoff]
    xTest = x[cutoff:]
    yTest = y[cutoff:]

    crf = LinearCRF(nClasses, d)
    nSamples = len(xTrain)
    
    for iter in range(1,20):
        lr = 0.05 /math.sqrt(iter)
        for i in range(nSamples):
            loss = crf.learn(yTrain[i], xTrain[i], learningRate=lr, alpha=0.0002, batchSize=5)
            
            
            


if __name__ == '__main__':
    trainPosTagger()

    exit(1)
    c = LinearCRF(3, 3)

    features = []
    labels = [1,2,3]
    for l in labels:
        features.append( scipy.sparse.coo_matrix(([1], ([0], [l-1])), shape=(1, 3), dtype=np.float32).T)

    while True:
        c.learn(labels, features)