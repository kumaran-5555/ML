import os
import datetime
from multiprocessing import Process
from multiprocessing import Queue 
from collections import defaultdict 
from sklearn import preprocessing
import json

import numpy as np

import pickle
import hashlib
from copy import deepcopy
import random



def _thread(trainer, outputDir, xTrain, yTrain, xCV, yCV, test, finalPredictFunc, reqQ, respQ):

    while True:
        params = json.loads(reqQ.get())
        if '_CMD' in params and params['_CMD'] == 'KILL':
            break

        hash = params['_HASH']        

        del params['_HASH']
        


        m = trainer(outputDir + hash, xTrain, yTrain, xCV, yCV, test, finalPredictFunc, params)
        m.start()
        
        respQ.put(m.strStats())
    return 0


class Orchestrator:
    def __init__(self, dataDir, outputDir, args, trainer, finalPredictFunc, resetData=False, threads=2, debug=False,\
        getData = None, exceptCols = []):
        self.dataDir = dataDir
        self.outputDir = outputDir
        self.resetData = resetData
        self.args = args
        self.threads = threads
        self.trainer = trainer
        self.xTrain = None
        self.yTrain = None
        self.xCV = None
        self.yCV = None
        self.params = list(args.keys())
        self.debug = debug
        self.finalPredictFunc = finalPredictFunc
        self.getData = getData
        self.exceptCols = exceptCols



        self.sweep = []
        self.reqQ = Queue()
        self.respQ = Queue()
        self.processes = []
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)

        self.statusFile = open(self.outputDir + 'report.tsv', 'a')

    @staticmethod
    def _getHash():
        return hashlib.md5(str(random.randint(0, 1000000)).encode('ascii')).hexdigest()
    
    def _processArgs(self, i, temp):
        if i >= len(self.params):
            # add model key param
            temp['_HASH'] = Orchestrator._getHash()
            self.sweep.append(temp.copy())
            return

        k = self.params[i]

        for v in self.args[k]:
            temp[k] = v
            self._processArgs(i+1, temp)


    def train(self):

        # get data and persist
        if self.resetData or not os.path.exists(self.dataDir + 'train.pkl') \
            or not os.path.exists(self.dataDir + 'cv.pkl') or not os.path.exists(self.dataDir + 'test.pkl'):
            self.getData(self.dataDir)




        
        print('STS: Loading data..')
        train = pickle.load(open(self.dataDir + 'train.pkl', 'rb'))
        cv = pickle.load(open(self.dataDir + 'cv.pkl', 'rb'))
        test = pickle.load(open(self.dataDir + 'test.pkl', 'rb'))

        cols = [i for i in train.columns.names if i not in self.exceptCols]

        train = train[cols]
        cv = cv[cols]
        test = test[cols]

        cols = [i for i in train.columns.names if i not in ['label']]

        self.xTrain = train[cols]
        self.yTrain = train['label']
        self.xCV = cv[cols]
        self.yCV = cv['label']
        self.test = test

        # do dymanic preproc
        print('STS: Preprocessing data..')
        self.xTrain, self.yTrain, self.xCV, self.yCV, self.test = self.preproc(self.xTrain, self.yTrain, self.xCV, self.yCV, self.test)


        # get sweep list
        print('STS: Generatic sweep params..')
        self._processArgs(0, defaultdict(lambda : None))
        print('STS: Total sweep params ', len(self.sweep))

        if self.debug:
            for p in self.sweep:
                hash = p['_HASH']
                del p['_HASH']

                m = self.trainer(self.outputDir + hash, self.xTrain, self.yTrain, self.xCV, self.yCV, self.test, self.finalPredictFunc, p)
                m.start()
                self.statusFile.write(m.strStatus() + '\t' + str(self.test.columns.names) + '\n')
                self.statusFile.flush()

            return

        # start threads
        print('STS: Creating threads {}...', self.threads)
        for i in range(self.threads):
            p = Process(target=_thread, args=(self.trainer, self.outputDir, self.xTrain, self.yTrain, self.xCV, self.yCV, self.test, self.finalPredictFunc, self.reqQ, self.respQ))
            p.start()
            self.processes.append(p)


        # assign work

        busyThreads = 0


        for params in self.sweep:
            if busyThreads >= self.threads:
                self.statusFile.write(self.respQ.get() + '\t' + str(self.test.columns.names) + '\n')
                self.statusFile.flush()
                busyThreads -= 1

            # there is thread availble here
            print('STS: started training for ', params)
            self.reqQ.put(json.dumps(params, sort_keys=True))
            busyThreads += 1



        print('STS: Done with all params...')
        while busyThreads:
            self.statusFile.write(self.respQ.get() + '\t' + str(self.test.columns.names) + '\n')
            self.statusFile.flush()
            busyThreads -= 1

        print('STS: Killing all threads...')
        for i in range(self.threads):        
            self.reqQ.put(json.dumps({'_CMD' :'KILL'}))

        for p in self.processes:
            p.join()

    def getData(self, train, cv):
        raise NotImplementedError


    def preproc(self, xTrain, yTrain, xCV, yCV, test):
        return xTrain, yTrain, xCV, yCV, test












