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

    def updateTerminalNodes(self, tree, dataX, dataY, residual, currentPrediction, learningRate=0.1):
        allTerminalNodes = tree.get_terminal_regions(dataX)

        for node in tree.treeDict:
            if tree.treeDict[node].isLeaf:
                self.updateTerminalNode(tree, dataY, allTerminalNodes, node, residual)

        # update the current prediction, i.e F_m(x) score to exsiting sum
        currentPrediction += (learningRate * tree.predict_value(dataX))




        

    def updateTerminalNode(self, tree, dataY, allTerminalNodes, currentNode, residual):
        '''
            this stage's estimator predicts one step in newtop-raphson method, which optimizes
            loss function L

            this stage tries to predict = -1 * (L'/ L''), and this added to previous stage
            L' = -2 * y + 2 * (exp(prediction) / (1 + exp(prediction)))
            -1 * L' = 2* (y - (exp(prediction) / (1 + exp(prediction))))
                    = 2 * (y -P)
        
                
            L'' = 2 * exp(prediction) / (1 + exp(prediction)) ** 2
                = 2 * (P * (1 - P))


            residual =  2.0 (Y - P)
            P = Y - 0.5 * residual, therefore


            -L' /L'' = 2 * (y - P) / (P * (1 - P))
                    =  residual / (2 * ( (Y - 0.5 * residual) * ( 1 - Y + 0.5 * residual)))
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
        denominator = 2.0 * numpy.sum( (selectedDataY - 0.5 * selectedResidual) * (1 - selectedDataY + 0.5 * selectedResidual))

        if denominator != 0:
            tree.treeDict[currentNode].nodeValue = numerator / denominator
        else:
            tree.treeDict[currentNode].nodeValue = 0.0










class GradientBoostedTreeClassifier:

    def __init__(self, dataX, dataY, nClasses, sampleWeight, minSamplesLeaf, minSamplesSplit, minWeightLeaf, maxDepth, maxEnsembles):
        if nClasses > 2:
            print('only binary classes are allowed')
            raise ValueError 
        
        self.dataX = dataX
        self.dataY = dataY
        self.nClasses = nClasses
        self.sampleWeight = sampleWeight
        self.minSamplesLeaf = minSamplesLeaf
        self.minWeightLeaf = minWeightLeaf
        self.minSamplesSplit = minSamplesSplit
        self.maxDepth = maxDepth
        self.maxEnsembles = maxEnsembles
        self.ensembles = []
         
        self.nSamples = self.dataX.shape[0]

        # stage wise predictions (n, maxEnsembles)
        self.predictions = numpy.zeros((self.nSamples, self.maxEnsembles))

        self.loss = BinomialLoss()
        self.currentPrediction = numpy.zeros((self.nSamples))

    def _initEstimator(self):
        posCount = 0
        for i in range(self.nSamples):
            if self.dataY[i] == 1:
                posCount += 1
        negCount = self.nSamples - posCount

        initialPredictions = numpy.zeros((self.nSamples))
        initialPredictions.fill(scipy.log(posCount /negCount))

        return initialPredictions



    def fit(self):
        '''
            fit all stages, stage 0 is fitted using initial esitmator
        '''
        
        self.currentPrediction = self._initEstimator()


        for i in range(1, self.maxEnsembles):
            loss = self.fitStage(i)


    def fitStage(self, i):

        residual = self.loss.negativeGradient(self.dataY, self.currentPrediction)

        # fit residual using dataX
        tree = DecisionTree.DecisionTreeBuilder(DecisionTree.Constants.Regression, self.dataX, residual, self.nClasses, \
            self.sampleWeight, self.minSamplesLeaf, self.minSamplesSplit, self.minWeightLeaf, self.maxDepth)

        tree.fit()
               
        # current predictions are updated in this function        
        self.loss.updateTerminalNodes(tree, self.dataX, self.dataY, residual, self.currentPrediction)

        # add the final tree to list of ensembles
        self.ensembles.append(tree)










if __name__ == '__main__':
    #b = ensemble.GradientBoostingClassifier(n_estimators=1)
    
    data = datasets.load_iris()
    x = data.data
    y = data.target

    for i in range(y.shape[0]):
        if y[i] != 0:
            y[i] = 1

    b = GradientBoostedTreeClassifier(x, y, 2, None, 2, 2, 2, None, 2)
    b.fit()

    print(b)


    '''
    print(b.init_.predict(x))

    pred = b.staged_decision_function(x)
    count=0
    for i in pred:

        print(i)
        count += 1
        if count >= 10:
            break


    print(pred.shape)

    '''