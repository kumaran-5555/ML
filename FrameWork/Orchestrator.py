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



def _thread(trainer, outputDir, xTrain, yTrain, xCV, yCV, reqQ, respQ):

    while True:
        params = json.loads(reqQ.get())
        if '_CMD' in params and params['_CMD'] == 'KILL':
            break

        hash = params['_HASH']        

        del params['_HASH']
        


        m = trainer(outputDir + hash, xTrain, yTrain, xCV, yCV, params)
        m.start()
        
        respQ.put(str(m.stats))
    return 0


class Orchestrator:
    def __init__(self, dataDir, outputDir, args, trainer, resetData=False, threads=2):
        self.dataDir = dataDir
        self.outputDir = outputDir
        self.resetData = resetData
        self.args = args
        self.threads = 2
        self.trainer = trainer
        self.xTrain = None
        self.yTrain = None
        self.xCV = None
        self.yCV = None
        self.params = list(args.keys())
        self.sweep = []
        self.reqQ = Queue()
        self.respQ = Queue()
        self.processes = []
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)

        self.statusFile = open(self.outputDir + 'report.tsv', 'w')

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
            or not os.path.exists(self.dataDir + 'cv.pkl'):
            print('STS: Resetting data..')
            train, cv = self.getData(self.dataDir + 'train.tsv', \
                self.dataDir + 'cv.tsv')


            self.xTrain = train[:,1:]
            self.yTrain = train[:,0]
            self.xCV = cv[:,1:]
            self.yCV = cv[:,0]

            print('STS: Persisting data..')
            # persist
            with open(self.dataDir + 'xTrain.pkl', 'wb') as file:
                pickle.dump(self.xTrain, file)

            with open(self.dataDir + 'yTrain.pkl', 'wb') as file:
                pickle.dump(self.yTrain, file)

            with open(self.dataDir + 'xCV.pkl', 'wb') as file:
                pickle.dump(self.xCV, file)

            with open(self.dataDir + 'yCV.pkl', 'wb') as file:
                pickle.dump(self.yCV, file)

        else:
            print('STS: Loading data..')
            train = pickle.load(open(self.dataDir + 'train.pkl', 'rb'))
            cv = pickle.load(open(self.dataDir + 'cv.pkl', 'rb'))

            self.xTrain = train[:,1:]
            self.yTrain = train[:,0]
            self.xCV = cv[:,1:]
            self.yCV = cv[:,0]


        # do dymanic preproc
        print('STS: Preprocessing data..')
        self.xTrain, self.yTrain, self.xCV, self.yCV = self.preproc(self.xTrain, self.yTrain, self.xCV, self.yCV)


        # get sweep list
        print('STS: Generatic sweep params..')
        self._processArgs(0, defaultdict(lambda : None))
        print('STS: Total sweep params ', len(self.sweep))

        # start threads
        print('STS: Creating threads {}...', self.threads)
        for i in range(self.threads):
            p = Process(target=_thread, args=(self.trainer, self.outputDir, self.xTrain, self.yTrain, self.xCV, self.yCV, self.reqQ, self.respQ))
            p.start()
            self.processes.append(p)


        # assign work

        busyThreads = 0

        for params in self.sweep:
            if busyThreads >= self.threads:
                self.statusFile.write(self.respQ.get() + '\n')
                self.statusFile.flush()
                busyThreads -= 1

            # there is thread availble here
            print('STS: started training for ', params)
            self.reqQ.put(json.dumps(params))
            busyThreads += 1



        print('STS: Done with all params...')
        while busyThreads:
            self.statusFile.write(self.respQ.get() + '\n')
            self.statusFile.flush()
            busyThreads -= 1

        print('STS: Killing all threads...')
        for i in range(self.threads):        
            self.reqQ.put({'_CMD' :'KILL'})

        for p in self.processes:
            p.join()

    def getData(self, train, cv):
        raise NotImplementedError


    def preproc(self, xTrain, yTrain, xCV, yCV):
        return xTrain, yTrain, xCV, yCV






class GrupoOrchestrator(Orchestrator):
    

    def getData(self, train, cv):
        
        cv = np.genfromtxt(cv, \
            delimiter='\t', usecols=(13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33), comments="m:", filling_values=1)

        train = np.loadtxt(train, \
            delimiter='\t', usecols=(13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33), comments="m:")

        return train, cv


    def preproc(self, xTrain, yTrain, xCV, yCV):
        
        scale = preprocessing.StandardScaler()
        xTrain = xTrain[:,[3,10,11,12,13,14,15,16,17,18,19]]
        xCV = xCV[:,[3,10,11,12,13,14,15,16,17,18,19]]

        xTrain = scale.fit_transform(xTrain)
        xCV = scale.transform(xCV)

        return xTrain, yTrain, xCV, yCV





