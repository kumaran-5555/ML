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
import scipy



class BinomialLoss():
    def loss(self, dataY, prediction):
        '''
            L = -2.0 * (y * log(P)  + (1-y) * log(1-P))
            P = exp(prediction) / (1 + exp(prediction))

            L = -2.0 * (Y * prediction - log(1 + exp(prediction)))
        '''
        return -2.0 * numpy.mean( dataY  * prediction - scipy.log(1 + scipy.exp(prediction)))

    def negativeGradient(self, dataY, prediction):
        '''
         L' = -2 * y + 2 * (exp(prediction) / (1 + exp(prediction)))
         residual   = - 1* L'
                    =  2* (y - (exp(prediction) / (1 + exp(prediction))))
                    =  2 * (y - P)
        '''
        return 2.0 * (dataY - (scipy.exp(prediction) / ( scipy.exp(prediction) + 1)))


    def updateTerminalNode(self, tree, dataY, prediction, allTerminalNodes, currentNode, residual):
        '''
            this stage's estimator predicts one step in newtop-raphson method, which optimizes
            loss function L

            this stage tries to predict = -1 * (L'/ L''), and this added to previous stage
            L' = -2 * y + 2 * (exp(prediction) / (1 + exp(prediction)))
            -1 * L' = 2* (y - (exp(prediction) / (1 + exp(prediction))))
                    = 2 * (y -P)
            L'' = exp(prediction) / (1 + exp(prediction)) ** 2
                = P * (1 - P)
            residual =  Y - P; P = Y - residual, therefore, ingoring 2
            L' /L'' = 2 * (y - P) / (P * (1 - P))       , ingoring 2
                    =  residual / ( (Y - residual) * ( 1 - Y + residual))
        '''

        # take all samples ending at this node
        # gives an array of indices of training data which ends at currentNode 
        selectedTerminalNodes = numpy.where(allTerminalNodes == currentNode)[0]

        # selects residuals of selected training data rows
        selectedResidual = residual.take(selectedTerminalNodes, axis=0)
        selectedDataY = dataY.take(selectedTerminalNodes, axis=0)

        # numerator = -1 * L' = residual
        numerator = numpy.sum(selectedResidual)

        # denomator = L'' = ( (Y - residual) * ( 1 - Y + residual))
        denominator = numpy.sum( (selectedDataY - selectedResidual) * (1 - selectedDataY + selectedResidual))

        if denominator != 0:
            tree[leaf].nodeValue = numerator / denominator
        else:
            tree[leaf].nodeValue = 0.0










class GradientBoostedTreeClassifier(object):

     def __init__(self, dataX, dataY, nClasses, sampleWeight, minSamplesLeaf, minSamplesSplit, minWeightLeaf, maxDepth, maxEnsembles):
         self.dataX = dataX
         self.dataY = dataY
         self.nClasses = nClasses
         self.sampleWeight = sampleWeight
         self.minSamplesLeaf = minSamplesLeaf
         self.minWeightLeaft = minWeightLeaf
         self.maxDepth = maxDepth
         self.maxEnsembles = maxEnsembles
         self.ensembles = []
         
         self.nSamples = self.dataX.shape[0]

         # stage wise predictions
         self.predictions = numpy.zeros((self.nSamples, self.maxEnsembles))
             


    def fit(self):

        pass




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

