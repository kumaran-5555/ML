import pandas as pd
import numpy as np
import re
import random
import pickle


brands = {}

class GrupoData:
    def __init__(self, outputDir):
        self.outputDir = outputDir


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
        selected = pd.merge(selected, temp2,  on=['Producto_ID', 'Cliente_ID', 'Semana'], how='left')

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

        
        print('STATUS: Reading products...')

        products = pd.read_csv(r'E:\Git\ML\Kaggle_Grupo\Data\producto_tabla.csv', dtype={ 'Producto_ID': np.uint16, 'NombreProducto': np.str})

        products['weight'] = products['NombreProducto'].apply(GrupoData.extractProductWeight)
        products['piece'] = products['NombreProducto'].apply(GrupoData.extractPieces)
        products['brand'] = products['NombreProducto'].apply(GrupoData.extractBrand)

        print('STATUS: merging product specific features')
        data = pd.merge(data, products, on='Producto_ID', how='left')

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
        pickle.dump(test, open(self.outputDir + 'test.pd.pkl', 'wb'))
        pickle.dump(cvdata, open(self.outputDir + 'cvdata.pd.pkl', 'wb'))

        # numpy pkl
        pickle.dump(train.values, open(self.outputDir + 'train.pkl', 'wb')) 
        pickle.dump(test.values, open(self.outputDir + 'test.pkl', 'wb'))
        pickle.dump(cvdata.values, open(self.outputDir + 'cv.pkl', 'wb'))


        print(products.columns)
        print(data.columns)

        












