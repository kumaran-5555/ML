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

class LeastSquareLoss():
    def loss(self, dataY, prediction):
        '''
            L = (y -p)**2
        '''
        return numpy.square(dataY - prediction)

    def negativeGradient(self, dataY, prediction):
        '''
            L' = -2 * (y - p) 

            residual = -1 * L'
                     = 2 * (y - p)
        '''

        return 2 * (dataY - prediction)


    def updateTerminalNodes(self, tree, dataX, dataY, residual, currentPrediction, learningRate=0.1):
        '''
            residual fitting already minimizes loss function because 
            it y-p is what is predicted for residual, so no need 
            for terminal node updates 

        '''

        # update the prediction score
        # boosting
        currentPrediction += learningRate * tree.predict_value(dataX)
        


class BinomialLoss():
    def lossE(self, dataY, prediction):
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






class GradientBosstedTreeRegressor:

    def __init__(self, dataX, dataY, sampleWeight, minSamplesLeaf, minSamplesSplit, minWeightLeaf, maxDepth, maxEnsembles, learningRate):
        
        self.dataX = dataX
        self.dataY = dataY
        self.sampleWeight = sampleWeight
        self.minSamplesLeaf = minSamplesLeaf
        self.minWeightLeaf = minWeightLeaf
        self.minSamplesSplit = minSamplesSplit
        self.maxDepth = maxDepth
        self.maxEnsembles = maxEnsembles
        self.ensembles = []
        self.learningRate = learningRate

         
        self.nSamples = self.dataX.shape[0]

        # stage wise predictions (n, maxEnsembles)
        self.predictions = numpy.zeros((self.nSamples, self.maxEnsembles))

        self.loss = LeastSquareLoss()
        self.currentPrediction = numpy.zeros((self.nSamples))
        self.initialPrediction = None

    def _initEstimator(self):
        '''
            we predict mean value as the initial prediction
        '''

        totalY = 0
        for i in range(self.nSamples):
            totalY += self.dataY[i]



        initialPredictions = numpy.zeros((self.nSamples))
        initialPredictions.fill(numpy.mean(self.dataY))

        # remember initial prediction, we will re-use this when we
        # predict for test samples

        self.initialPrediction = numpy.mean(self.dataY)


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
        tree = DecisionTree.DecisionTreeBuilder(DecisionTree.Constants.Regression, self.dataX, residual, 0, \
            self.sampleWeight, self.minSamplesLeaf, self.minSamplesSplit, self.minWeightLeaf, self.maxDepth)

        tree.fit()
               
        # current predictions are updated in this function        
        self.loss.updateTerminalNodes(tree, self.dataX, self.dataY, residual, self.currentPrediction, learningRate=self.learningRate)

        # add the final tree to list of ensembles
        self.ensembles.append(tree)
        
    

    def predict_stages(self, dataX, startStage, endStage):
        '''
            predict stages from start to end and adds all predicted
            scores  with  learningRate used for training
        '''

        # initialize predictions
        
        score = numpy.zeros(dataX.shape[0])
        # fill with the intial prediction used for
        # training 
        score.fill(self.initialPrediction)
        
        for i in range(startStage, endStage):
            tree = self.ensembles[i]
            predictedScore = tree.predict_value(dataX)

            score += predictedScore * self.learningRate

        return score


    def predict_value(self, dataX):
        totalTrees = len(self.ensembles)
        score = self.predict_stages(dataX, 0, totalTrees)


        return score

    


class GradientBoostedTreeClassifier:

    def __init__(self, dataX, dataY, nClasses, sampleWeight, minSamplesLeaf, minSamplesSplit, minWeightLeaf, maxDepth, maxEnsembles, learningRate):
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
        self.learningRate = learningRate

         
        self.nSamples = self.dataX.shape[0]

        # stage wise predictions (n, maxEnsembles)
        self.predictions = numpy.zeros((self.nSamples, self.maxEnsembles))

        self.loss = BinomialLoss()
        self.currentPrediction = numpy.zeros((self.nSamples))
        self.initialPrediction = None


    def _initEstimator(self):
        posCount = 0
        for i in range(self.nSamples):
            if self.dataY[i] == 1:
                posCount += 1
        negCount = self.nSamples - posCount

        initialPredictions = numpy.zeros((self.nSamples))
        initialPredictions.fill(scipy.log(posCount /negCount))

        # remember initial prediction, we will re-use this when we
        # predict for test samples

        self.initialPrediction = scipy.log(posCount / negCount)


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
        self.loss.updateTerminalNodes(tree, self.dataX, self.dataY, residual, self.currentPrediction, learningRate=self.learningRate)

        # add the final tree to list of ensembles
        self.ensembles.append(tree)



    

    def predict_stages(self, dataX, startStage, endStage):
        '''
            predict stages from start to end and adds all predicted
            scores  with  learningRate used for training
        '''

        # initialize predictions
        
        score = numpy.zeros(dataX.shape[0])
        # fill with the intial prediction used for
        # training 
        score.fill(self.initialPrediction)
        
        for i in range(startStage, endStage):
            tree = self.ensembles[i]
            predictedScore = tree.predict_value(dataX)

            score += predictedScore * self.learningRate

        return score


    def predict_value(self, dataX):
        totalTrees = len(self.ensembles)
        score = self.predict_stages(dataX, 0, totalTrees)


        return score

    def predict_proba(self, dataX):
        totalTrees = len(self.ensembles)
        score = self.predict_stages(dataX, 0, totalTrees)

        proba = numpy.ones((score.shape[0], 2), dtype=numpy.float64)
        proba[:, 1] = 1.0 / (1.0 + numpy.exp(-score.ravel()))
        proba[:, 0] -= proba[:, 1]
        return proba


if __name__ == '__main__':
    #b = ensemble.GradientBoostingClassifier(n_estimators=1)
    
    '''
    data = datasets.load_iris()
    x = data.data
    y = data.target

    for i in range(y.shape[0]):
        if y[i] != 2:
            y[i] = 0
        else:
            y[i] = 1


    b = GradientBoostedTreeClassifier(x, y, 2, None, 2, 2, 2, None, 50, 0.1)
    b.fit()

    print(b)

    print(b.predict_value(x))
    '''

    data = datasets.load_boston()

    b = GradientBosstedTreeRegressor(data.data, data.target, None, 2, 2, 2, None, 50, 0.1)

    b.fit()


    print(b.predict_value(data.data))




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