import TrainModel
import Orchestrator

from sklearn import preprocessing
from sklearn import linear_model
import numpy as np
from GrupoData import GrupoData

def output(test, model, file):
    # remove id column
    test = test[:,:-1]
    pred = model.predict(test)
    pred.apply(np.expm1)

    file.write('id,Demanda_uni_equil\n')

    for i in range(pred.shape[0]):
        file.write('{},{}\n', i, pred[i])

    file.flush()

    return pred.shape[0]






class GrupoOrchestrator(Orchestrator.Orchestrator):
    

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

    

class GrupoOrchestratorNoScale(Orchestrator.Orchestrator):
    

    def getData(self, train, cv):
        
        cv = np.genfromtxt(cv, \
            delimiter='\t', usecols=(13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33), comments="m:", filling_values=1)

        train = np.loadtxt(train, \
            delimiter='\t', usecols=(13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33), comments="m:")

        return train, cv



def SGD():
    p = {}
    p['alpha'] = [1,0.75, 0.5, 0.45, 0.4, 0.25, 0.1]
    m = TrainModel.SGDTrain
    o = GrupoOrchestrator(r'E:\Git\ML\Kaggle_Grupo\Data\DataV1\\', 
                                       r'E:\Git\ML\Kaggle_Grupo\Data\OutputSGD\\', 
                                       p, TrainModel.SGDTrain, output, resetData=False)

    o.train()

def Ridge():
    p = {}
    p['alpha'] = [100, 10, 1,0.75, 0.5, 0.45, 0.4, 0.25, 0.1, 0.01, 0.001, 0.0001]
    
    o = GrupoOrchestrator(r'E:\Git\ML\Kaggle_Grupo\Data\DataV1\\', 
                                       r'E:\Git\ML\Kaggle_Grupo\Data\OutputRidge\\', 
                                       p, TrainModel.Ridge, output, resetData=False)

    o.train()


def Tree():
    p = {}
    p['mil'] = [500, 250, 100, 50, 25, 10]
    p['nl'] = [ 100, 75, 50, 25, 10, 5]
    p['ff'] = [1]
    p['mb'] = [512, 256, 1024]

    
    o = GrupoOrchestratorNoScale('E:\Git\ML\Kaggle_Grupo\Data\DataV1\\', 
                                       r'E:\Git\ML\Kaggle_Grupo\Data\OutputTree\\', 
                                       p, TrainModel.TreeRegressor, output, resetData=False)

    o.train()

def ada():
    p = {}
    #p['base_estimator'] = [linear_model.SGDRegressor]
    p['n_estimators'] = [250, 150, 100, 50]
    p['learning_rate'] = [0.1, 0.25, 0.5, 1]
    p['loss'] = ['square']

    o = GrupoOrchestratorNoScale('E:\Git\ML\Kaggle_Grupo\Data\DataV1\\', 
                                       r'E:\Git\ML\Kaggle_Grupo\Data\OutputAda\\', 
                                       p, TrainModel.AdaBoost, output, resetData=False)

    o.train()


def xgb():
    p = {}
    #p['base_estimator'] = [linear_model.SGDRegressor]
    p['learning_rate'] = [0.1, 0.05]
    p['max_depth'] = [10, 15]
    p['n_estimators'] = [75, 125, 150]
    p['objective'] = ['reg:linear']
    p['colsample_bytree'] = [0.8]
    p['silent'] = [False]
    p['subsample'] = [0.8]
    o = GrupoOrchestratorNoScale('E:\Git\ML\Kaggle_Grupo\Data\DataV2\\', 
                                       r'E:\Git\ML\Kaggle_Grupo\Data\OutputXGB\\', 
                                       p, TrainModel.XGB, output, resetData=False, threads=1, debug=False)

    

    o.train()



if __name__ == '__main__':
    #SGD()
    #Tree()
    #Ridge()
    #ada()
    #xgb()

    #c = GrupoData(r'E:\\Git\ML\Kaggle_Grupo\Data\DataV2\\')
    #c.process()

    xgb()








