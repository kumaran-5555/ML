from multiprocessing import Process
from multiprocessing import Queue 
import sklearn.svm
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

import sys
import numpy
import scipy.sparse
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score


def measureModel(YTest, YPred):
    return 1


def trainGBT(requestsQ, responsesQ):
    while True:
        args = requestsQ.get()
        if args[0] == 'KILL':
            break

        vectors = args[1]     
        # expected in the order of learningRate, maxTrees, minSplitSize, maxDepth
        hyperparams = args[2]

        model =   GradientBoostingClassifier(learning_rate=hyperparams[0], n_estimators=hyperparams[1], min_samples_split=hyperparams[2], max_depth=hyperparams[3])
        
        model.fit(vectors['Xtrain'], vectors['Ytrain'])
        score = accuracy_score(vectors['Ytest'], model.predict(vectors['Xtest']))        
        responsesQ.put((model, score), True)

    return 0

def ThreadedGBTSweep(learningRateVec, numTreesVec, minSamplesToSplitVec, maxDepthVec, Xtrain, Ytrain, Xtest, Ytest, statusFile):
    
    
    threads = 5
    requestsQ = Queue()
    responsesQ = Queue()
    busyThreads = 0 
    grid = []
    params = {}
    params['Xtrain'] = Xtrain
    params['Xtest'] = Xtest
    params['Ytrain'] = Ytrain
    params['Ytest'] = Ytest
    bestModel = None
    bestScore = -1
    

    for l in learningRateVec:
        for t in numTreesVec:
            for m in minSamplesToSplitVec:
                for d in maxDepthVec:
                    grid.append([l,t,m,d])
    processes = []
    print('creating processes ...')
    for i in range(threads):
        p = Process(target=trainGBT, args=(requestsQ, responsesQ))
        p.start()
        processes.append(p)
    print('grid sweeping ...')
    for hyperparams in grid:
        if busyThreads >= threads:

            status = responsesQ.get()
            if status[-1] > bestScore:
                bestScore = status[-1]
                bestModel = status[0]
            print('result  ',str(status))
            statusFile.write(str(status).replace('\n','').replace('\r','')+'\n')
            statusFile.flush()
            busyThreads -= 1
        # there is thread availble here
        print('started training for ',str(hyperparams))
        requestsQ.put(('CONT',params, hyperparams))
        busyThreads += 1

    print('done with all grid ...', busyThreads)
    while busyThreads:
        status = responsesQ.get()
        if status[-1] > bestAuc:
            bestAuc = status[-1]
            bestModel = status[0]
        statusFile.write(str(status).replace('\n','').replace('\r','')+'\n')
        statusFile.flush()
        busyThreads -= 1

    print('killing all processes ...')
    for i in range(threads):        
        requestsQ.put(('KILL',params))

    for p in processes:
        p.join()
    
    statusFile.close()
    return bestModel, bestAuc







