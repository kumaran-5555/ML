import sys
import pickle
import datetime

from sklearn import linear_model
from sklearn import metrics
import numpy as np






class TrainModel(object):
    def __init__(self, modelFile, xTrain, yTrain, xCV, yCV, args):
        self.modelFile = modelFile
        self.args = args
        self.model = None
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.xCV = xCV
        self.yCV = yCV
        self.precitionsCV = None
        self.stats = {}


    def __str__(self, **kwargs):
        return str(self.stats)

    def save(self):

        if self.model == None:
            raise ValueError("self.model is not set, train() should set this value")

        with open(self.modelFile + '.' + self.__class__.__name__ + '.pkl', 'wb') as file:
            pickle.dump(self.model, file)

        self.stats['modelFile'] = self.modelFile + '.' + self.__class__.__name__ + '.pkl'

        with open(self.modelFile + '.' + self.__class__.__name__ + '.stats.txt', 'w',) as file:
            file.write(str(self.stats))


        return True

    def start(self):
        start = datetime.datetime.now()
        self.stats['startTime'] = start.strftime("%Y-%m-%d %H:%M:%S")
        self.stats['args'] = self.args

        self.train(**self.args)
        
        self.cvpredict()

        end = datetime.datetime.now()
        self.stats['endTime'] = end.strftime("%Y-%m-%d %H:%M:%S")
        self.stats['totalTime'] = str(end - start)
        self.save()





    def metric(self):
        raise NotImplementedError

    def cvpredict(self):
        
        self.precitionsCV = self.model.predict(self.xCV)

        self.stats['cvErrorBeforeFreeform'] = self.metric()

        self.precitionsCV = self.freeform(self.xCV, self.precitionsCV)

        self.stats['cvError'] = self.metric()

    def train(self, **kargs):
        raise NotImplementedError

    def freeform(self, x, predictions):
        return predictions
    
    def predict(self, xTest):
        '''
        for final measurement
        '''

        if self.model == None:
            self.model = pickle.load(open(self.modelFile, 'rb'))

        temp =  self.model.predict(xTest)
        return self.freeform(xTest, temp)




class SGDTrain(TrainModel):

    def __init__(self, modelFile, xTrain, yTrain, xCV, yCV, args):
        return super().__init__(modelFile, xTrain, yTrain, xCV, yCV, args)


    def train(self, **kargs):
        self.model = linear_model.SGDRegressor(**kargs)

        self.model.fit(self.xTrain, self.yTrain)


    def metric(self):
        return metrics.mean_squared_error(self.yCV, self.precitionsCV)




                

