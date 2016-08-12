import sys
import pickle
import datetime

from sklearn import linear_model
from sklearn import ensemble as skensemble
from sklearn import metrics
from sktlc import ensemble
import numpy as np
import xgboost as xgb

class TrainModel(object):
    def __init__(self, modelFile, xTrain, yTrain, xCV, yCV, test, finalPredictFunc, args):
        self.modelFile = modelFile
        self.args = args
        self.model = None
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.xCV = xCV
        self.yCV = yCV
        self.precitionsCV = None
        self.stats = {}
        self.test = test
        self.finalPredictFunc = finalPredictFunc
        self.outputFile = open(self.modelFile + '.' + self.__class__.__name__ + '.txt', 'w')



    def strStats(self):
        return str.format('{}\t{}\t{}\t{}\t{}\t{}\t{}', self.stats['args'], self.stats['cvError'], \
            self.stats['cvErrorBeforeFreeform'], self.stats['startTime'], self.stats['endTime'], \
            self.stats['totalTime'], self.stats['modelFile'])



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

        self.predict()

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
    
    def predict(self):
        '''
        for final measurement
        '''

        self.stats['totalFinalPredictions'] = self.finalPredictFunc(self.test, self.model, self.outputFile)





class SGDTrain(TrainModel):

    def __init__(self, modelFile, xTrain, yTrain, xCV, yCV, args):
        return super().__init__(modelFile, xTrain, yTrain, xCV, yCV, args)


    def train(self, **kargs):
        self.model = linear_model.SGDRegressor(**kargs)

        self.model.fit(self.xTrain, self.yTrain)


    def metric(self):
        return metrics.mean_squared_error(self.yCV, self.precitionsCV)


class Ridge(TrainModel):

    def __init__(self, modelFile, xTrain, yTrain, xCV, yCV, args):
        return super().__init__(modelFile, xTrain, yTrain, xCV, yCV, args)


    def train(self, **kargs):
        self.model = linear_model.Ridge(**kargs)

        self.model.fit(self.xTrain, self.yTrain)


    def metric(self):
        return metrics.mean_squared_error(self.yCV, self.precitionsCV)


class TreeRegressor(TrainModel):

    def __init__(self, modelFile, xTrain, yTrain, xCV, yCV, args):
        return super().__init__(modelFile, xTrain, yTrain, xCV, yCV, args)


    def train(self, **kargs):
        self.model = ensemble.auto_TlcFastForestRegression.TlcFastForestRegression(**kargs)

        self.model.fit(self.xTrain, self.yTrain)


    def metric(self):
        return metrics.mean_squared_error(self.yCV, self.precitionsCV)
                

class AdaBoost(TrainModel):

    def __init__(self, modelFile, xTrain, yTrain, xCV, yCV, args):
        return super().__init__(modelFile, xTrain, yTrain, xCV, yCV, args)


    def train(self, **kargs):
        self.model = skensemble.AdaBoostRegressor(**kargs)

        self.model.fit(self.xTrain, self.yTrain)


    def metric(self):
        return metrics.mean_squared_error(self.yCV, self.precitionsCV)


class XGB(TrainModel):

    def train(self, **kargs):
        self.model = xgb.sklearn.XGBRegressor(**kargs)
        self.model.fit(self.xTrain, self.yTrain,\
            verbose=True, early_stopping_rounds=10,\
            eval_metric='rmse', eval_set=[tuple((self.xCV, self.yCV))])

    def metric(self):
        return metrics.mean_squared_error(self.yCV, self.precitionsCV)


    def metric2(self, pred, true):
        return metrics.mean_squared_error(true, pred)

        
    
       