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
        self.edgeWieghts = np.random.normal(size=(self.n, self.n, 1, self.d), scale=0.1)

        # one row for each state
        self.classWeights = np.random.normal(size=(self.n, 1, self.d), scale=0.1)
        self.alpha = None
        self.beta = None
        self.fiCache = {}
        
        self.gE = np.zeros_like(self.edgeWieghts)
        self.gW = np.zeros_like(self.classWeights)


    @staticmethod
    def dot(w, x):
        if w.shape[1] != x.shape[0]:
            raise ValueError('Dimension mismatch w {} x {}', w.shape, x.shape)

        score = 0.0
        for i in range(len(x.row)):      
            score +=  w[0][x.row[i]] * x.data[i]
        return score

    @staticmethod
    def sum(w, x):
        if w.shape[1] != x.shape[1]:
            raise ValueError('Dimension mismatch w {} x {}', w.shape, x.shape)
        for i in range(len(x.col)):
            w[0][x.col[i] ] += x.data[i]

    @staticmethod
    def sumMul(w, x, c):
        if w.shape[1] != x.shape[0]:
            raise ValueError('Dimension mismatch w {} x {}', w.shape, x.shape)

        for i in range(len(x.row)): 
            w[0][x.row[i] ] += (x.data[i] * c)
    
    @staticmethod
    def sum2(w, x):
        if w.shape[1] != x.shape[0]:
            raise ValueError('Dimension mismatch w {} x {}', w.shape, x.shape)

        for i in range(len(x.row)): 
            w[0][x.row[i] ] += (x.data[i])





    def score(self, labels, features):
        
        m = len(labels)-1
        score = 0.0
        for i in range(1, m+1):
            if i==1:
                score = self.fi(0, labels[i], i, features)
                continue

            score *= self.fi(labels[i-1], labels[i], i, features)
        return score

    def pOfYGivenX(self, labels, features):
        scoreOfYGivenX = self.score(labels, features)
        ZOfX = self.ZOfX(features)
        if (scoreOfYGivenX - ZOfX) > 1e-2:
            scoreOfYGivenX = self.score(labels, features)
            self.forwardBackward(features)
            raise ValueError
        return scoreOfYGivenX / ZOfX

    def ZOfX(self, features):                
        m = len(features)-1
        # consider all possible ending states for alpha table
        score = 0.0
        for s in range(1, self.n):
            score += self.alpha[s, m]                
        return score

    
    def forwardBackward(self, features):
        m = len(features)-1
        # we don't include the sepcial starting stage in alpha table, 
        # because sequence can't end with special state
        
        # aplha[i, j] = sum of liklyhood of subseq [0...i] having label seq 
        # ending with state j        
        self.alpha = np.zeros((self.n, m+1))
        self.alpha[0, 0] = 1 
        
        for i in range(1, m+1):
            for s in range(1, self.n):
                if i == 1:
                    self.alpha[s, i] += self.alpha[0, i-1] * self.fi(0, s, i, features)
                    continue
                for _s in range(1, self.n):                    
                    self.alpha[s, i] += self.alpha[_s, i-1] * self.fi(_s, s, i, features)
                
        # beta[i, j] = sum of likelyhood of subseq [i+1...m-1] having previous label j

        
        self.beta = np.zeros((self.n, m+1))
        for i in range(m, 0, -1):
            for s in range(1, self.n):
                if i == m:
                    # last item in sequence it not going to be previous
                    # for any other subseq, intialize with 1 as noop for multiplication
                    self.beta[s, i] = 1
                    continue
                
                for _s in range(1, self.n):
                    # k tracks (current) state at position i+1, j tracks (prev) state i
                    self.beta[s, i] += self.beta[_s, i+1] * self.fi(s, _s, i+1, features)

        

        '''
        aZOfXScore = 0.0
        bZOfXScore = 0.0

        for s in range(1, self.n):
            aZOfXScore += self.alpha[s, m]

        for s in range(1, self.n):
            bZOfXScore += (self.beta[s, 1]  * self.fi(0, s, 1, features))
        
        if abs(aZOfXScore-bZOfXScore) > 0.1:
            pass

       '''


    def fi(self, prevY, y, i, features):
        key = (prevY, y, i)
        if key in self.fiCache:
            return self.fiCache[key]

        x = features[i]
        
        score = LinearCRF.dot(self.edgeWieghts[prevY, y], x) + LinearCRF.dot(self.classWeights[y], x)
        score = np.exp(score)
        if score >  500:
            score = 500
        elif score < 1e-5:
            score = 1e-5
        self.fiCache[key] = score
        return score



    def loss(self, labels, features):
        return np.log(self.pOfYGivenX(labels, features))

    def learn(self, labels, features, learningRate=0.09, alpha=0.002):
        # add dummy place holders to make the indcies from 1...m
        # add a dummy rows to label and features
        labels = [None] + labels
        features = [None] + features
        self.alpha = None
        self.beta = None
        self.fiCache = {}
        

        self.forwardBackward(features)

        
        
        self.gE *= 0.0
        self.gW *= 0.0
        
        
        
        self.updateGradientForFirstTerm(labels, features)
        
        
        self.updateGradientForSecondTerm(features)
        
        self.updateGradientOfL2Penalty(alpha)
        
        
        l = self.loss(labels, features)
        p = self.pOfYGivenX(labels, features)
        z = self.ZOfX(features)
        print(l, p, z)

        
        # gradient descent        
        self.edgeWieghts += (learningRate * self.gE)
        self.classWeights += (learningRate * self.gW)
        

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
                    q_i_a_b = (self.alpha[a, i-1] * self.fi(a, b, i, features) * self.beta[b, i])  / ZOfXScore
                    LinearCRF.sumMul(self.gW[b], features[i], q_i_a_b)
                    LinearCRF.sumMul(self.gE[a, b], features[i], q_i_a_b)
                continue

            for a in range(1, self.n):                
                for b in range(1, self.n):                    
                    
                    q_i_a_b = (self.alpha[a, i-1] * self.fi(a, b, i, features) * self.beta[b, i])  / ZOfXScore
                    
                    LinearCRF.sumMul(self.gW[b], features[i], q_i_a_b)
                    
                    LinearCRF.sumMul(self.gE[a, b], features[i], q_i_a_b)

        

    def updateGradientOfL2Penalty(self, alpha):
        self.gW += alpha * self.classWeights
        self.gE += alpha * self.edgeWieghts


# we will use features from  https://nlpforhackers.io/crf-pos-tagger/
def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        #'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        #'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        #'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
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


    if not os.path.exists(r'E:\Temp\treebank.pkl'):
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
        
        pickle.dump((x,y,d,nClasses), open(r'E:\Temp\treebank.pkl', 'wb'))

    else:
        x, y, d, nClasses = pickle.load(open(r'E:\Temp\treebank.pkl', 'rb'))

    cutoff = int(.75 * len(x))
    xTrain = x[:cutoff]
    yTrain = y[:cutoff]
    xTest = x[cutoff:]
    yTest = y[cutoff:]

    crf = LinearCRF(nClasses, d)
    nSamples = len(xTrain)
    
    while True:
        i = np.random.choice(nSamples, 1)[0]
        i = 0
        loss = crf.learn(yTrain[i], xTrain[i])






if __name__ == '__main__':
    trainPosTagger()