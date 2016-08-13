import TrainModel
import Orchestrator

from sklearn import preprocessing
from sklearn import linear_model
import numpy as np
import pandas as pd
from GrupoData import GrupoData

def output(test, model, file):
    # remove id column
    
    test['label'] = test['label'].astype(int)
    
    

    week10 = test[test['Semana']==10]
    



    week11 = test[test['Semana']==11]
    
    week10['pred'] = np.expm1(model.predict(week10.values[:,:-1]))
    file.write('id,Demanda_uni_equil\n')
    temp = week10[['label', 'pred']]
    temp.to_csv(file, index=False, delimiter=',', header=False)

    week10['Semana'] = week10['Semana'] + 1

    week10 = week10[['Cliente_ID', 'Producto_ID', 'Semana', 'pred']]

    week10 = week10.groupby(by=['Cliente_ID', 'Producto_ID', 'Semana'], as_index=False).mean()

    

    week11 = pd.merge(week11, week10, on=['Cliente_ID', 'Producto_ID', 'Semana'], how='left')

    week11['l1'] = week11['pred']
    del week11['pred']

    week11['pred'] = np.expm1(model.predict(week11.values[:,:-1]))

    temp = week11[['label', 'pred']]
    temp.to_csv(file, index=False, delimiter=',', header=False)

    file.flush()

    return test.shape[0]






class GrupoOrchestrator(Orchestrator.Orchestrator):
    

    
    
    def preproc(self, xTrain, yTrain, xCV, yCV, test):
        
        scale = preprocessing.StandardScaler()

        xTrain = scale.fit_transform(xTrain)
        xCV = scale.transform(xCV)

        temp = test.values[:,:-1]

        scale.transform(temp, copy=False)

        return xTrain, yTrain, xCV, yCV, test

    

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
    o = GrupoOrchestrator(r'E:\Git\ML\Kaggle_Grupo\Data\DataV4\\', 
                                       r'E:\Git\ML\Kaggle_Grupo\Data\OutputSGD2\\', 
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
    p['learning_rate'] = [0.1]
    p['max_depth'] = [10]
    p['n_estimators'] = [150]
    p['objective'] = ['reg:linear']
    p['colsample_bytree'] = [0.7]
    p['silent'] = [False]
    p['subsample'] = [0.85]
    o = GrupoOrchestratorNoScale('E:\Git\ML\Kaggle_Grupo\Data\DataV5\\', 
                                       r'E:\Git\ML\Kaggle_Grupo\Data\OutputXGB2\\', 
                                       p, TrainModel.XGB, output, resetData=False, threads=1, debug=False)

    

    o.train()



if __name__ == '__main__':
    #SGD()
    #Tree()
    #Ridge()
    #ada()
    #xgb()

    c = GrupoData(r'E:\\Git\ML\Kaggle_Grupo\Data\DataV5\\')
    c.process()

    xgb()
    #SGD()








