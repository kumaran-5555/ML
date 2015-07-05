import numpy
import scipy
import os 
from collections import defaultdict
import random
import time
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics



class LinearRegression:
    def __init__(self, dataX, dataY, iterations=100, learningRate=0.0001):
        self.numSamples = dataX.shape[0]
        self.numParams = dataX.shape[1]
        self.dataX = dataX
        self.dataY = dataY
        self.iterations = iterations
        self.learningRate = learningRate


    @staticmethod
    def _loss(params, dataX, dataY):
        value = numpy.dot(dataX, params)
        #print(value.shape)
        ret = numpy.mean(numpy.square(dataY - value))
        return ret

    @staticmethod
    def _batchGradient(params, dataX, dataY):
        loss = (dataY - numpy.dot(dataX, params))
        # gradient theta1 = -2 x1 * (dataY - dataX * theta)
        ret =  -2 * (numpy.dot(dataX.T, loss)) / dataX.shape[0]
        return ret

    @staticmethod
    def _miniBatchGradient(params, dataX, dataY):
        # take 50 % samples
        numSamples = dataX.shape[0]
        newSamples = random.sample(range(numSamples), numSamples//2)
        newDataX = numpy.take(dataX, newSamples, axis = 0)
        newDataY = numpy.take(dataY, newSamples, axis = 0)

        loss = (newDataY - numpy.dot(newDataX, params))
        # gradient theta1 = -2 x1 * (dataY - dataX * theta)
        return -2 * (numpy.dot(newDataX.T, loss)) / newDataX.shape[0]

    @staticmethod
    def printParams(params):
        print(params)


    def fit(self):
        # initialize params to random
        self.modelParams = numpy.array([random.randint(-100,100) for i in range(self.numParams)])
        self.optimLogit  = scipy.optimize.fmin_bfgs(LinearRegression._loss, x0 = self.modelParams, \
            args = (self.dataX, self.dataY), gtol = 1e-5, fprime = LinearRegression._batchGradient, \
            callback = LinearRegression.printParams, maxiter = 300)
        pass

    def predict(self, dataX):
        ret =  numpy.dot(dataX, self.optimLogit)
        return ret


if __name__ == '__main__':
    d = sklearn.datasets.load_boston()

    m = LinearRegression(d.data, d.target)

    m.fit()
    print(m.optimLogit)
    m.predict(d.data)
    print(sklearn.metrics.mean_squared_error(d.target, m.predict(d.data)))

    m2 = sklearn.linear_model.LinearRegression()
    m2.fit(d.data, d.target)
    print(m2.coef_)
    print(sklearn.metrics.mean_squared_error(d.target, m2.predict(d.data)))





