import os
import datetime
from multiprocessing import Process
from multiprocessing import Queue 
from collections import defaultdict 
from sklearn import preprocessing
import json
import gc
import numpy as np

import pickle
import hashlib
from copy import deepcopy
import random




class FeaturePrunner:
    def __init__(self, dataDir, outputDir, args, trainer, finalPredictFunc, pruneOrder):
        self.dataDir = dataDir
        self.outputDir = outputDir        
        self.args = args        
        self.trainer = trainer
        self.xTrain = None
        self.yTrain = None
        self.xCV = None
        self.yCV = None
        self.params = list(args.keys())        
        # func(test, model, file)
        self.finalPredictFunc = finalPredictFunc                
        self.pruneOrder = pruneOrder
        self.badFeatures = []
        self.bestScore = None


        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)

        self.statusFile = open(self.outputDir + 'report.tsv', 'a')

    @staticmethod
    def _getHash():
        return hashlib.md5(str(random.randint(0, 1000000)).encode('ascii')).hexdigest()
    


    def train(self):

        
        print('STS: Loading data..')
        train = pickle.load(open(self.dataDir + 'train.pkl', 'rb'))
        cv = pickle.load(open(self.dataDir + 'cv.pkl', 'rb'))
        test = pickle.load(open(self.dataDir + 'test.pkl', 'rb'))

        
        # we make train and cv have same columns

        
        cols = [i for i in train.columns.values.tolist()if i not in ['label']]
        self.xTrain = train[cols]
        self.yTrain = train['label']

        cols = [i for i in cv.columns.values.tolist()if i not in ['label']]
        self.xCV = cv[cols]
        self.yCV = cv['label']
        self.test = test

        # run with all features to get best score
        hash = FeaturePrunner._getHash()
        m = self.trainer(self.outputDir + hash, self.xTrain, self.yTrain, self.xCV, self.yCV, self.test, self.finalPredictFunc, self.args)
        m.start()
        self.bestScore = m.stats['return']
        self.statusFile.write(m.strStats() + '\t' + str(self.xTrain.columns.values.tolist()) + '\n')
        self.statusFile.flush()

        
        for f in self.pruneOrder:
            hash = FeaturePrunner._getHash()

            print('Trying to prune feature {}',format(f))
            cols = [c for c in self.xTrain.columns.values if c not in self.badFeatures + [f]]
            temp = self.xTrain[cols]

            m = self.trainer(self.outputDir + hash, temp , self.yTrain, self.xCV, self.yCV, self.test, self.finalPredictFunc, self.args)
            m.start()
            
            self.statusFile.write(m.strStats() + '\t' + str(self.xTrain.columns.values.tolist()) + '\n')
            self.statusFile.flush()

            score = m.stats['return']

            if score >= self.bestScore:
                print("Found bad feature score {} bestScore {} feature {}".format(score, self.bestScore, f))
                self.bestScore = score
                self.badFeatures.append(f)
            else:
                print("Could not remove feature score {} bestScore {} feature {}".format(score, self.bestScore, f))


        return

        


