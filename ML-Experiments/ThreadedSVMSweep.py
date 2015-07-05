from multiprocessing import Process
from multiprocessing import Queue 
import FeatureHasherWithoutLabel
import sklearn.svm
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
import sys
import numpy
import scipy.sparse
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score


def trainSvm(requestsQ, responsesQ):
    while True:
        args = requestsQ.get()
        if args[0] == 'KILL':
            break

        vectors = args[1]        
        m = sklearn.svm.SVC(C=args[2], cache_size = 1000, probability=True, class_weight = {'1': args[3], '0': 1.0}, gamma=args[4])
        m.fit(vectors['Xtrain'], vectors['Ytrain'])
        auc = roc_auc_score(numpy.array(vectors['Ytest'], dtype=float), m.predict_proba(vectors['Xtest'])[:,1])
        responsesQ.put((m, auc), True)
    return 0

def threadedSVMSweep(Cvec, Wvec, Gvec, Xtrain, Ytrain, Xtest, Ytest, statusFile):
    
    
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
    bestAuc = -1
    

    for c in Cvec:
        for w in Wvec:
            for g in Gvec:
                grid.append([c,w,g])
    processes = []
    print('creating processes ...')
    for i in range(threads):
        p = Process(target=trainSvm, args=(requestsQ, responsesQ))
        p.start()
        processes.append(p)
    print('grid sweeping ...')
    for p in grid:
        if busyThreads >= threads:

            status = responsesQ.get()
            if status[-1] > bestAuc:
                bestAuc = status[-1]
                bestModel = status[0]
            print('result  ',str(status))
            statusFile.write(str(status).replace('\n','').replace('\r','')+'\n')
            statusFile.flush()
            busyThreads -= 1
        # there is thread availble here
        print('started training for ',str(p))
        requestsQ.put(('CONT',params, p[0], p[1], p[2]))
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







