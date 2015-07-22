import math
import numpy
import random
import pickle
import json
from sklearn import datasets
import DataSets
from sklearn import metrics
import DecisionTree
from sklearn import ensemble

class BinomialLoss():
    pass

class LeastSquaresError():
    def loss(self, dataY, prediction):
        '''
            
        '''
        return -2.0 * numpy.mean( dataY  * prediction - math.log(1 + math.exp(prediction)))

    def negativeGradient(self, dataY, prediction):
        return -2.0 * (dataY - (math.exp(prediction) / ( math.exp(prediction) + 1)))






class GradientBoostedTrees(object):

    """description of class"""


if __name__ == '__main__':
    b = ensemble.GradientBoostingClassifier(n_estimators=1)
    data = datasets.load_iris()
    x = data.data
    y = data.target

    for i in range(y.shape[0]):
        if y[i] != 0:
            y[i] = 1
    b.fit(x, y)
    print(b.init_.predict(x))

    pred = b.staged_decision_function(x)
    count=0
    for i in pred:

        print(i)
        count += 1
        if count >= 10:
            break


    print(pred.shape)

