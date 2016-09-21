import sys
import pickle
import datetime
import math
from sklearn import linear_model
from sklearn import ensemble as skensemble
from sklearn import metrics
from sktlc import ensemble
import numpy as np
import xgboost as xgb

class TrainModel(object):
    def __init__(self, prefix, xTrain, yTrain, xCV, yCV, test, finalPredictFunc, args):
        self.prefix = prefix + '.' + self.__class__.__name__
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
        self.outputFile = open(self.prefix + '.txt', 'w')        
        self.statsFile = open(self.prefix + '.stats.txt', 'w')
        self.modelFile = open(self.prefix + '.pkl', 'wb')





    def strStats(self):
        return str.format('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}', self.stats['args'], self.stats['cvError'], \
            self.stats['cvErrorBeforeFreeform'], self.stats['startTime'], self.stats['endTime'], \
            self.stats['totalTime'], self.stats['modelFile'], self.stats['trainShape'])



    def save(self):

        if self.model == None:
            raise ValueError("self.model is not set, train() should set this value")

        
        pickle.dump(self.model, self.modelFile)

        self.stats['modelFile'] = self.prefix + '.pkl'

        self.modelFile.flush()
        
        self.statsFile.write(str(self.stats))
        self.statsFile.flush()


        return True

    def start(self):
        start = datetime.datetime.now()
        self.stats['startTime'] = start.strftime("%Y-%m-%d %H:%M:%S")
        self.stats['args'] = self.args
        self.stats['trainShape'] = self.xTrain.shape

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
        
        self.precitionsCV = self.model.predict(self.xCV.values)

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

        self.stats['totalFinalPredictions'] = self.finalPredictFunc(self)





class SGDTrain(TrainModel):

    def __init__(self, modelFile, xTrain, yTrain, xCV, yCV, args):
        return super().__init__(modelFile, xTrain, yTrain, xCV, yCV, args)


    def train(self, **kargs):
        self.model = linear_model.SGDRegressor(**kargs)

        self.model.fit(self.xTrain, self.yTrain)


    def metric(self):
        return math.sqrt(metrics.mean_squared_error(self.yCV, self.precitionsCV))


class Ridge(TrainModel):

    def __init__(self, modelFile, xTrain, yTrain, xCV, yCV, args):
        return super().__init__(modelFile, xTrain, yTrain, xCV, yCV, args)


    def train(self, **kargs):
        self.model = linear_model.Ridge(**kargs)

        self.model.fit(self.xTrain, self.yTrain)


    def metric(self):
        return math.sqrt(metrics.mean_squared_error(self.yCV, self.precitionsCV))


class TreeRegressor(TrainModel):

    def __init__(self, modelFile, xTrain, yTrain, xCV, yCV, args):
        return super().__init__(modelFile, xTrain, yTrain, xCV, yCV, args)


    def train(self, **kargs):
        self.model = ensemble.auto_TlcFastForestRegression.TlcFastForestRegression(**kargs)

        self.model.fit(self.xTrain, self.yTrain)


    def metric(self):
        return math.sqrt(metrics.mean_squared_error(self.yCV, self.precitionsCV))
                

class AdaBoost(TrainModel):

    def __init__(self, modelFile, xTrain, yTrain, xCV, yCV, args):
        return super().__init__(modelFile, xTrain, yTrain, xCV, yCV, args)


    def train(self, **kargs):
        self.model = skensemble.AdaBoostRegressor(**kargs)

        self.model.fit(self.xTrain, self.yTrain)


    def metric(self):
        return math.sqrt(metrics.mean_squared_error(self.yCV, self.precitionsCV))


class XGB(TrainModel):

    def train(self, **kargs):
        self.model = xgb.sklearn.XGBRegressor(**kargs)

        self.model.fit(self.xTrain, self.yTrain,\
            verbose=True, early_stopping_rounds=10,\
            eval_metric='rmse', eval_set=[tuple((self.xCV, self.yCV))])

    def metric(self):
        return math.sqrt(metrics.mean_squared_error(self.yCV, self.precitionsCV))
        


    def metric2(self, pred, true):
        return metrics.mean_squared_error(true, pred)

        
class XGBClassifier(TrainModel):
    def train(self, **kargs):
        self.model = xgb.sklearn.XGBClassifier(**kargs)

        fmap = {}
        
        for f in self.xTrain.columns.values:
            fmap['f{}'.format(len(fmap))] = f


        self.model.fit(self.xTrain.values, self.yTrain.values,\
            verbose=True, early_stopping_rounds=10,\
            eval_metric='auc', eval_set=[tuple((self.xCV.values, self.yCV.values))])


        importance = {}
        for k,v in self.model.booster().get_fscore().items():
            importance[fmap[k]] = v

        fscore = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        self.statsFile.write(self.prefix  + '\t' + str(fscore))
        self.statsFile.flush()

    def metric(self):
        return metrics.auc(self.yCV.values, self.precitionsCV, reorder=True)
    
