import numpy as np 
import sys
import os
from nltk.tag.util import untag
import nltk
import collections
import scipy


# http://www.cs.columbia.edu/~mcollins/crf.pdf
class LinearCRF:
    def __init__(self, nClasses, featuresDimension):        
        self.d = featuresDimension
        self.n = nClasses + 1

        # one row for each pair of states
        self.edgeWieghts = np.random.normal(size=(self.n, self.n, self.d))

        # one row for each state
        self.classWeights = np.random.normal(size=(self.n, self.d))


    def score(self, labels, features):
        
        m = len(labels)-1
        score = 0.0
        for i in range(1, m+1):
            if i==1:
                score = self.fi(0, labels[i], features[i])
                continue

            score *= self.fi(labels[i-1], labels[i], features[i])
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
                    self.alpha[s, i] += self.alpha[0, i-1] * self.fi(0, s, features[i])
                    continue
                for _s in range(1, self.n):                    
                    self.alpha[s, i] += self.alpha[_s, i-1] * self.fi(_s, s, features[i])
        

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
                    self.beta[s, i] += self.beta[_s, i+1] * self.fi(s, _s, features[i+1])

        aZOfXScore = 0.0
        bZOfXScore = 0.0

        for s in range(1, self.n):
            aZOfXScore += self.alpha[s, m]

        for s in range(1, self.n):
            bZOfXScore += (self.beta[s, 1]  * self.fi(0, s, features[1]))
        
        if abs(aZOfXScore-bZOfXScore) > 0.1:
            pass

       

    def fi(self, prevY, y, x):
        score = np.dot(self.edgeWieghts[prevY, y] , x) + np.dot(self.classWeights[y], x)
        return np.exp(score)


    def loss(self, labels, features):
        return np.log(self.pOfYGivenX(labels, features))

    def learn(self, labels, features, learningRate=0.01, alpha=0.002):
        # add dummy place holders to make the indcies from 1...m
        labels = [None] + labels
        features = [None] + features
        self.alpha = None
        self.beta = None
        

        self.forwardBackward(features)

        self.gE = np.zeros_like(self.edgeWieghts)
        self.gW = np.zeros_like(self.classWeights)

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
            self.gW[labels[i]] += features[i]
            if i==1:
                self.gE[0, labels[i]] += features[i]
                continue
            self.gE[labels[i-1], labels[i]] += features[i]

    def updateGradientForSecondTerm(self, features):
        m = len(features) - 1        
        ZOfXScore = self.ZOfX(features)
        for i in range(1, m+1):
            if i==1:
                a = 0
                for b in range(1, self.n):
                    q_i_a_b = (self.alpha[a, i-1] * self.fi(a, b, features[i]) * self.beta[b, i])  / ZOfXScore
                    self.gW[b] += np.multiply(features[i], q_i_a_b)
                    self.gE[a, b] += np.multiply(features[i], q_i_a_b)
                continue

            for a in range(1, self.n):                
                for b in range(1, self.n):
                    q_i_a_b = (self.alpha[a, i-1] * self.fi(a, b, features[i]) * self.beta[b, i])  / ZOfXScore
                    self.gW[b] += np.multiply(features[i], q_i_a_b)
                    self.gE[a, b] += np.multiply(features[i], q_i_a_b)

        

    def updateGradientOfL2Penalty(self, alpha):
        self.gW += alpha * self.classWeights
        self.gE += alpha * self.edgeWieghts


# we will use features from  https://nlpforhackers.io/crf-pos-tagger/
def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }

def vectorizer(data):
    featureDimesntion = collections.defaultdict(lambda : collections.defaultdict())

    for row in data:
        for index in row:
            for f,v in index.items():
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

    return vectorizedData

            

def trainPosTagger():
    tagged_sentences = nltk.corpus.treebank.tagged_sents()
    # Split the dataset for training and testing
    cutoff = int(.75 * len(tagged_sentences))
    training_sentences = tagged_sentences[:cutoff]
    test_sentences = tagged_sentences[cutoff:]
    
    def transform_to_dataset(tagged_sentences):
        X, y = [], []
    
        for tagged in tagged_sentences:
            X.append([features(untag(tagged), index) for index in range(len(tagged))])
            y.append([tag for _, tag in tagged])
    
        return X, y
    
    X_train, y_train = transform_to_dataset(training_sentences)
    X_test, y_test = transform_to_dataset(test_sentences)

    X_train_vec = vectorizer(X_train)

    pass





if __name__ == '__main__':
    trainPosTagger()