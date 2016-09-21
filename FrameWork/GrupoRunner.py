import TrainModel
import Orchestrator

from sklearn import preprocessing
from sklearn import linear_model
import numpy as np
import pandas as pd
#from GrupoData import GrupoData
import pandas as pd
import numpy as np
import re
import random
import pickle
import os
import math

brands = {}

class GrupoData:
    def __init__(self, outputDir):
        self.outputDir = outputDir
        if not os.path.exists(self.outputDir):
            os.mkdir(self.outputDir)


    

    @staticmethod
    def extractBrand(name):
        global brands
        pattern = r'^[A-Z]+$'

        fields = name.split(' ')

        m = re.match(pattern, fields[len(fields)-2])

        if fields[len(fields)-2] not in brands:            
            brands[fields[len(fields)-2]] = len(brands)+1

        return brands[fields[len(fields)-2]]

    @staticmethod 
    def extractPieces(name):
        pattern = r'.*\b([0-9]+)p\b.*'

        m = re.match(pattern, name)

        if m is None:
            return 0
        return int(m.group(1))

    @staticmethod
    def extractVolumn(name):
        patternVol = r'.*\b([0-9]+)ml\b.*'

        m = re.match(patternVol, name)
        if m is None:
            return 0

        return int(m.group(1))

    @staticmethod
    def extractProductWeight(name):
        patternGrams = r'.*\b([0-9]+)g\b.*'
        patternKilos = r'.*\b([0-9]+)Kg\b.*'



        m = re.match(patternGrams, name)

        if m is None:
            m = re.match(patternKilos, name)
        else:
            return int(m.group(1))

        if m is None: 
            return 0

        return int(m.group(1))* 1000


    @staticmethod
    def varianceLag(row):
        lags = []
        if row['l1'] != np.nan:
            lags.append(row['l1'])

        if row['l2'] != np.nan:
            lags.append(row['l2'])

        if row['l3'] != np.nan:
            lags.append(row['l3'])

        if row['l4'] != np.nan:
            lags.append(row['l4'])

        if row['l5'] != np.nan:
            lags.append(row['l5'])


        return np.var(lags)


    @staticmethod
    def newProduct(row):
        if row['l1'] == np.nan and row['l2'] == np.nan and row['l3'] == np.nan and \
            row['l4'] == np.nan and row['l5'] == np.nan:
            return 1

        return 0

        


    def process(self):
        dtypes_test = {'Semana': np.int8, 'Agencia_ID': np.int16, 'Canal_ID': np.int8, 'Producto_ID': np.uint16}
        dtypes_train = {'Semana': np.int8, 'Agencia_ID': np.int16, 'Canal_ID': np.int8, 'Producto_ID': np.uint16,
               'Venta_uni_hoy': np.uint16, 'Dev_uni_proxima': np.int32, 'Demanda_uni_equil': np.int16}



        print('STATUS: Reading input..')
        train = pd.read_csv(r'E:\Git\ML\Kaggle_Grupo\Data\train.csv', dtype = dtypes_train, \
            usecols = ['Semana', 'Producto_ID', 'Ruta_SAK', 'Agencia_ID', 'Cliente_ID', 'Demanda_uni_equil'])#, nrows =10000)

        train['target'] = train['Demanda_uni_equil']
        train['istest'] = 0

        test =  pd.read_csv(r'E:\Git\ML\Kaggle_Grupo\Data\test.csv', dtype = dtypes_test, usecols = ['id', 'Semana', 'Producto_ID', 'Ruta_SAK', 'Agencia_ID', 'Cliente_ID']\
            )#, nrows =10000)


        
        test['target'] = 0
        test['istest'] = 1

        data = pd.concat((train, test))

        print('STATUS: train shape {} test shape {}', train.values.shape, test.values.shape)

        del train, test

        temp = data[data['istest']==0][['Producto_ID', 'Cliente_ID', 'Semana', 'target']]        
        temp2  = temp.groupby(by=['Producto_ID', 'Cliente_ID', 'Semana'], as_index=False).mean()
        
        del temp

        selected = data[data['Semana'] >= 8]
        #print(selected[:20])

        numOfClients = data[data['istest']==0].groupby(by=['Agencia_ID']).Cliente_ID.nunique()
        selected['numOfClients'] = selected['Agencia_ID'].map(numOfClients)
        print('STATUS: merging numOfClients')
        del numOfClients

        numOfProducts = data[data['istest']==0].groupby(by=['Cliente_ID']).Producto_ID.nunique()
        selected['numOfProducts'] = selected['Cliente_ID'].map(numOfProducts)
        print('STATUS: merging numOfProducts')
        del numOfProducts

        # l1
        temp2['Semana'] = temp2['Semana'] + 1
        temp2.rename(columns={'target': 'l1'}, inplace=True)
        #selected = pd.merge(selected, temp2,  on=['Producto_ID', 'Cliente_ID', 'Semana'], how='left')
        selected['l1'] = 0

        # l2
        temp2['Semana'] = temp2['Semana'] + 1
        temp2.rename(columns={'l1': 'l2'}, inplace=True)
        selected = pd.merge(selected, temp2,  on=['Producto_ID', 'Cliente_ID', 'Semana'], how='left')

        # l3
        temp2['Semana'] = temp2['Semana'] + 1
        temp2.rename(columns={'l2': 'l3'}, inplace=True)
        selected = pd.merge(selected, temp2, on=['Producto_ID', 'Cliente_ID', 'Semana'], how='left')

        # l3
        temp2['Semana'] = temp2['Semana'] + 1
        temp2.rename(columns={'l3': 'l4'}, inplace=True)
        selected = pd.merge(selected, temp2, on=['Producto_ID', 'Cliente_ID', 'Semana'], how='left')

        # l3
        temp2['Semana'] = temp2['Semana'] + 1
        temp2.rename(columns={'l4': 'l5'}, inplace=True)
        selected = pd.merge(selected, temp2, on=['Producto_ID', 'Cliente_ID', 'Semana'], how='left')

        del temp2


        #selected['lagSum'] = selected['l1'] + selected['l2'] + selected['l3'] + selected['l4'] + selected['l5']
        #selected['lagAvg'] = selected['lagSum'] / 5

        
        print('STATUS: Reading products...')

        products = pd.read_csv(r'E:\Git\ML\Kaggle_Grupo\Data\producto_tabla.csv', dtype={ 'Producto_ID': np.uint16, 'NombreProducto': np.str})

        products['weight'] = products['NombreProducto'].apply(GrupoData.extractProductWeight)
        products['piece'] = products['NombreProducto'].apply(GrupoData.extractPieces)
        products['brand'] = products['NombreProducto'].apply(GrupoData.extractBrand)
        products['volumn'] = products['NombreProducto'].apply(GrupoData.extractVolumn)
        
        #pidCluster = pd.read_csv(r'E:\Git\ML\Kaggle_Grupo\Data\productsCluster.csv', dtype={ 'Producto_ID': np.uint16, 'cluster': np.uint16}, \
        #    usecols=['Producto_ID', 'cluster'])

        #selected = pd.merge(selected, pidCluster, on=['Producto_ID'], how='left')
        

        print('STATUS: merging product specific features')
        selected = pd.merge(selected, products, on=['Producto_ID'], how='left')
        del selected['NombreProducto']

        '''
        #selected['weightPerPiece'] = selected['weight']/selected['piece']
        #selected.replace(to_replace=np.inf, value=0, inplace=True)


        


        
        pca = (data[data['istest']==0].groupby(by=['Producto_ID', 'Cliente_ID', 'Agencia_ID'], as_index=False))['target'].mean()
        pca.rename(columns={'target': 'pcamean'},inplace=True)        
        selected = pd.merge(selected, pca, on=['Producto_ID', 'Cliente_ID', 'Agencia_ID'], how='left')
        print('STATUS: merging pca')
        del pca

        pa = (data[data['istest']==0].groupby(by=['Producto_ID', 'Agencia_ID'], as_index=False))['target'].mean()
        pa.rename(columns={'target': 'pamean'},inplace=True)
        selected = pd.merge(selected, pa, on=['Producto_ID', 'Agencia_ID'], how='left')
        print('STATUS: merging pa')
        del pa

        p = (data[data['istest']==0].groupby(by=['Producto_ID'], as_index=False))['target'].mean()
        p.rename(columns={'target': 'pmean'},inplace=True)
        selected = pd.merge(selected, p, on=['Producto_ID'], how='left')
        print('STATUS: merging p')
        del p

        
        # impute missing values
        selected.ix[pd.isnull(selected['pamean']), 'pamean'] = selected['pmean']
        selected.ix[pd.isnull(selected['pcamean']), 'pcamean'] = selected['pamean']

        selected.ix[pd.isnull(selected['l1']), 'l1'] = selected['pcamean']
        selected.ix[pd.isnull(selected['l1']), 'l1'] = selected['pamean']
        selected.ix[pd.isnull(selected['l1']), 'l1'] = selected['pmean']

        selected.ix[pd.isnull(selected['l2']), 'l2'] = selected['pcamean']
        selected.ix[pd.isnull(selected['l2']), 'l2'] = selected['pamean']
        selected.ix[pd.isnull(selected['l2']), 'l2'] = selected['pmean']

        selected.ix[pd.isnull(selected['l3']), 'l3'] = selected['pcamean']
        selected.ix[pd.isnull(selected['l3']), 'l3'] = selected['pamean']
        selected.ix[pd.isnull(selected['l3']), 'l3'] = selected['pmean']

        selected.ix[pd.isnull(selected['l4']), 'l4'] = selected['pcamean']
        selected.ix[pd.isnull(selected['l4']), 'l4'] = selected['pamean']
        selected.ix[pd.isnull(selected['l4']), 'l4'] = selected['pmean']

        selected.ix[pd.isnull(selected['l5']), 'l5'] = selected['pcamean']
        selected.ix[pd.isnull(selected['l5']), 'l5'] = selected['pamean']
        selected.ix[pd.isnull(selected['l5']), 'l5'] = selected['pmean']
        '''

        # add frequency features
        nPid = selected.groupby(by=['Producto_ID','Semana'], as_index=False).size().groupby(level=0).mean()
        selected['nPid'] = selected['Producto_ID'].map(nPid)
        

        nAgen = selected.groupby(by=['Agencia_ID','Semana'], as_index=False).size().groupby(level=0).mean()
        selected['nAgen'] = selected['Agencia_ID'].map(nAgen)


        nRuka = selected.groupby(by=['Ruta_SAK','Semana'], as_index=False).size().groupby(level=0).mean()
        selected['nRuka'] = selected['Ruta_SAK'].map(nRuka)

        nClient = selected.groupby(by=['Cliente_ID','Semana'], as_index=False).size().groupby(level=0).mean()
        selected['nClient'] = selected['Cliente_ID'].map(nClient)

        #nClientPid = selected.groupby(by=['Cliente_ID','Producto_ID','Semana'], as_index=False).size().to_frame(name='nClientPid').reset_index()
        #nClientPid = nClientPid
        #selected = pd.merge(selected, nClientPid[['Cliente_ID','Producto_ID']], on=['Cliente_ID','Producto_ID'], how='left')
        #selected['nClient'] = selected['Cliente_ID'].map(nClient

        temp = selected[['l1','l2','l3','l4','l5']]
        temp = temp.fillna(0)

        selected['lagVar'] = np.var(temp, axis=1)
        selected['newProduct'] = np.sum(temp, axis=1) == 0

        selected['newProduct'].replace(False, 0, inplace=True)
        selected['newProduct'].replace(True, 1, inplace=True)

        # from week 9, hold some data for cv
        cv = random.sample(selected[selected['Semana']==9].index.tolist(), 30000)

        selected['label'] = selected['target'].apply(np.log1p)
        del selected['Demanda_uni_equil']
        del selected['target']

        cvdata = selected.loc[cv]
        selected = selected.drop(cv)

        train = selected[selected['istest']==0]
        test = selected[selected['istest']==1]        
        test['label'] = test['id']

        del train['id']
        del test['id']
        del cvdata['id']
        del train['istest']
        del test['istest']
        del cvdata['istest']

        # save the frames

        train[:10000].to_csv(self.outputDir + 'train.tsv', sep='\t', encoding='utf-8', index=False)
        test[:10000].to_csv(self.outputDir + 'test.tsv', sep='\t', encoding='utf-8', index=False)
        cvdata[:10000].to_csv(self.outputDir + 'cv.tsv', sep='\t', encoding='utf-8', index=False)

        pickle.dump(train, open(self.outputDir + 'train.pd.pkl', 'wb'))
        pickle.dump(test, open(self.outputDir + 'test.pkl', 'wb'))
        pickle.dump(cvdata, open(self.outputDir + 'cvdata.pd.pkl', 'wb'))

        # numpy pkl
        pickle.dump(train.values, open(self.outputDir + 'train.pkl', 'wb')) 
        pickle.dump(test.values, open(self.outputDir + 'test.np.pkl', 'wb'))
        pickle.dump(cvdata.values, open(self.outputDir + 'cv.pkl', 'wb'))


        print(products.columns)
        print(data.columns)

        del train, selected, test, cvdata


def output(modelObj):
    test = modelObj.test
    file = modelObj.outputFile
    model = modelObj.model

    # remove id column
    
    test['label'] = test['label'].astype(int)
    
    

    week10 = test[test['Semana']==10]
    



    week11 = test[test['Semana']==11]
    
    week10['pred'] = np.expm1(model.predict(week10.values[:,:-1]))
    file.write('id,Demanda_uni_equil\n')
    temp = week10[['label', 'pred']]
    temp.to_csv(file, index=False, delimiter=',', header=False)
    '''
    week10['Semana'] = week10['Semana'] + 1

    
    week10 = week10[['Cliente_ID', 'Producto_ID', 'Semana', 'pred']]

    week10 = week10.groupby(by=['Cliente_ID', 'Producto_ID', 'Semana'], as_index=False).mean()

    

    week11 = pd.merge(week11, week10, on=['Cliente_ID', 'Producto_ID', 'Semana'], how='left')
    week11['l1'] = week11['pred']
    del week11['pred']
    
    


    temp = week11[['l1','l2','l3','l4','l5']]
    temp = temp.fillna(0)

    week11['lagVar'] = np.var(temp, axis=1)
    week11['newProduct'] = np.sum(temp, axis=1) == 0

    week11['newProduct'].replace(False, 0, inplace=True)
    week11['newProduct'].replace(True, 1, inplace=True)
    '''

    #week11['lagSum'] = week11['l1'] + week11['l2'] + week11['l3'] + week11['l4'] + week11['l5']
    #week11['lagAvg'] = week11['lagSum'] / 5

    week11['pred'] = np.expm1(model.predict(week11.values[:,:-1]))

    temp = week11[['label', 'pred']]
    temp.to_csv(file, index=False, delimiter=',', header=False)

    file.flush()

    return test.shape[0]



def ensembler(listOfFiles, outputFile):
    output = []
    outputFile = open(r'E:\Git\ML\Kaggle_Grupo\Data\OutputXGB2\\'+outputFile, 'w')
    outputFile.write( 'id,Demanda_uni_equil\n')

    for f in listOfFiles:
        output.append(pd.read_csv(r'E:\Git\ML\Kaggle_Grupo\Data\OutputXGB2\\'+f, dtype={ 'id' : np.str, 'Demanda_uni_equil' : np.float32}))

    combined = pd.DataFrame()

    combined['id'] = output[0]['id']
    combined['output'] = output[0]['Demanda_uni_equil']
    
    for i in range(1,len(output)):
        combined['output'] += output[i]['Demanda_uni_equil']


    combined['output'] /= len(output)
    combined.to_csv(outputFile, index=False, delimiter=',', header=False)






        


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
    p['n_estimators'] = [200,150]
    p['max_depth'] = [15,10]    
    p['objective'] = ['reg:linear']
    p['colsample_bytree'] = [0.7, 0.8]
    p['silent'] = [False]
    p['subsample'] = [0.85]
    o = GrupoOrchestratorNoScale('E:\Git\ML\Kaggle_Grupo\Data\DataV13\\', 
                                       r'E:\Git\ML\Kaggle_Grupo\Data\OutputXGB2\\', 
                                       p, TrainModel.XGB, output, resetData=False, threads=1, debug=False)

    

    o.train()



if __name__ == '__main__':
    #SGD()
    #Tree()
    #Ridge()
    #ada()
    #xgb()

    #c = GrupoData(r'E:\\Git\ML\Kaggle_Grupo\Data\DataV13\\')
    #c.process()

    #xgb()
    ensembler(["a8e6a51174ef36fe5b3cbd8550ddcf05.XGB.txt","d37bfca59aa216f026743150d37832de.XGB.txt","94f2471816b6d3a6c93d206efa30c434.XGB.txt","49d3b4db2721b0aa94033cd2bbbdde43.XGB.txt","91c30519a1534b0bbd486b53030a2fe8.XGB.txt","e4ee58f125b7394597886ae36399f60a.XGB.txt","cc87a19643b55e16b96d9a383b0b2f78.XGB.txt","9c0ae91de519fde8d1fd871bb4fa3520.XGB.txt","af0f3df616d22dcee3bf8e2ed28d820c.XGB.txt","8d3141657c1c6295a85990445285382b.XGB.txt","f042acffd562cdc9fa2d680ce267d661.XGB.txt","58976c7e9ee3e80437ddbf3ddae8980d.XGB.txt"], 'ensemble1.txt')
    #SGD()








