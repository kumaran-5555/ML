import TrainModel
import Orchestrator
import FeaturePrunning
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics


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
import gc

config = set()


def output(modelObj):


    pass




def processNumCat(outputDir):
    global cols
    global config
    global usecols

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    
    if 'basic' in config:
        train = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\train_basic.pkl','rb'))
        numeric = ['Id', 'L0_S0_F10', 'L0_S0_F10_max', 'L0_S0_F16', 'L0_S0_F18', 'L0_S0_F18_max', 'L0_S0_F2', 'L0_S0_F20', 'L0_S0_F20_max', 'L0_S0_F22', 'L0_S0_F2_max', 'L0_S10_F219', 'L0_S10_F219_max', 'L0_S10_F244', 'L0_S10_F244_max', 'L0_S10_F259', 'L0_S10_F259_max', 'L0_S11_F294', 'L0_S11_F294_max', 'L0_S12_F350', 'L0_S12_F350_max', 'L0_S13_F356', 'L0_S13_F356_max', 'L0_S14_F370', 'L0_S14_F370_max', 'L0_S14_F374', 'L0_S14_F374_max', 'L0_S15_F403', 'L0_S15_F403_max', 'L0_S15_F418', 'L0_S15_F418_max', 'L0_S17_F433', 'L0_S17_F433_max', 'L0_S18_F439', 'L0_S18_F439_max', 'L0_S19_F455', 'L0_S19_F455_max', 'L0_S1_F28', 'L0_S1_F28_max', 'L0_S23_F619', 'L0_S23_F671', 'L0_S23_F671_max', 'L0_S2_F44', 'L0_S2_F44_max', 'L0_S2_F60', 'L0_S2_F60_max', 'L0_S3_F100', 'L0_S3_F100_max', 'L0_S5_F114', 'L0_S5_F116', 'L0_S5_F116_max', 'L0_S6_F122', 'L0_S6_F122_max', 'L0_S6_F132', 'L0_S6_F132_max', 'L0_S7_F138', 'L0_S7_F138_max', 'L0_S9_F165', 'L0_S9_F165_max', 'L1_S24_F1565', 'L1_S24_F1565_max', 'L1_S24_F1569', 'L1_S24_F1571', 'L1_S24_F1581', 'L1_S24_F1604', 'L1_S24_F1604_max', 'L1_S24_F1632', 'L1_S24_F1632_max', 'L1_S24_F1647', 'L1_S24_F1647_max', 'L1_S24_F1667', 'L1_S24_F1672', 'L1_S24_F1695', 'L1_S24_F1695_max', 'L1_S24_F1723', 'L1_S24_F1723_max', 'L1_S24_F1778', 'L1_S24_F1778_max', 'L1_S24_F1838', 'L1_S24_F1838_max', 'L1_S24_F1842', 'L1_S24_F1844', 'L1_S24_F1846', 'L1_S24_F1846_max', 'L2_S26_F3036', 'L2_S26_F3036_max', 'L2_S26_F3047', 'L2_S26_F3047_max', 'L2_S26_F3062', 'L2_S26_F3062_max', 'L2_S26_F3069', 'L2_S26_F3073', 'L2_S26_F3073_max', 'L2_S26_F3113', 'L2_S26_F3117', 'L2_S26_F3121', 'L2_S26_F3121_max', 'L2_S27_F3129', 'L2_S27_F3129_max', 'L2_S27_F3133', 'L2_S27_F3133_max', 'L2_S27_F3140', 'L2_S27_F3140_max', 'L2_S27_F3144', 'L2_S27_F3144_max', 'L2_S27_F3210', 'L2_S27_F3210_max', 'L3_S29_F3321', 'L3_S29_F3324', 'L3_S29_F3327', 'L3_S29_F3327_max', 'L3_S29_F3330', 'L3_S29_F3330_max', 'L3_S29_F3336', 'L3_S29_F3336_max', 'L3_S29_F3342', 'L3_S29_F3342_max', 'L3_S29_F3351', 'L3_S29_F3351_max', 'L3_S29_F3354', 'L3_S29_F3370', 'L3_S29_F3373', 'L3_S29_F3373_max', 'L3_S29_F3376', 'L3_S29_F3382', 'L3_S29_F3382_max', 'L3_S29_F3407', 'L3_S29_F3407_max', 'L3_S29_F3412', 'L3_S29_F3430', 'L3_S29_F3436', 'L3_S29_F3458', 'L3_S29_F3461', 'L3_S29_F3461_max', 'L3_S29_F3467', 'L3_S29_F3479', 'L3_S29_F3479_max', 'L3_S30_F3494', 'L3_S30_F3494_max', 'L3_S30_F3504', 'L3_S30_F3524', 'L3_S30_F3544', 'L3_S30_F3544_max', 'L3_S30_F3554', 'L3_S30_F3554_max', 'L3_S30_F3564', 'L3_S30_F3574', 'L3_S30_F3574_max', 'L3_S30_F3604', 'L3_S30_F3609', 'L3_S30_F3609_max', 'L3_S30_F3689', 'L3_S30_F3689_max', 'L3_S30_F3704', 'L3_S30_F3754', 'L3_S30_F3754_max', 'L3_S30_F3759', 'L3_S30_F3769', 'L3_S30_F3769_max', 'L3_S30_F3804', 'L3_S30_F3804_max', 'L3_S30_F3809', 'L3_S30_F3809_max', 'L3_S32_F3850', 'L3_S32_F3850_max', 'L3_S33_F3855', 'L3_S33_F3855_max', 'L3_S33_F3857', 'L3_S33_F3857_max', 'L3_S33_F3859', 'L3_S33_F3859_max', 'L3_S33_F3861', 'L3_S33_F3863', 'L3_S33_F3865', 'L3_S33_F3865_max', 'L3_S34_F3876', 'L3_S34_F3878', 'L3_S34_F3882', 'L3_S34_F3882_max', 'L3_S35_F3889', 'L3_S36_F3920', 'L3_S36_F3920_max', 'L3_S38_F3952', 'L3_S38_F3952_max', 'L3_S38_F3956', 'L3_S38_F3960', 'L3_S40_F3980', 'L3_S40_F3986', 'L3_S41_F4016', 'L3_S43_F4080']
        #train = train[numeric + ['label']]
   
        


    if 'date' in config:
        date = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\train_minmax.2.pkl','rb'))
        train = pd.merge(train, date, on=['Id'])
        del date

    if 'leak' in config:
        leak = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\train_leak2.pkl','rb'))
        train = pd.merge(train, leak, on=['Id'])
        del leak

    if 'dups' in config:
        dups = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\train_dup2.pkl','rb'))
        train = pd.merge(train, dups, on=['Id'])

        #train['MeanResponse2'] = train['MeanResponse'] * train['SortedIdDiff']
        del dups

    if 'motoki' in config:
        motoki = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\train_motoki.pkl','rb'))
        
        cols = set(motoki.columns.values)
        cols = set(train.columns.values).intersection(cols)
        cols = [f for f in motoki.columns.values if f not in cols] + ['Id']
        motoki = motoki[cols]
        train = pd.merge(train, motoki, on=['Id'])
        del motoki

    if 'motoki2' in config:
        motoki = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\train_motoki2.pkl','rb'))
        
        cols = set(motoki.columns.values)
        cols = set(train.columns.values).intersection(cols)
        cols = [f for f in motoki.columns.values if f not in cols] + ['Id']
        motoki = motoki[cols]
        train = pd.merge(train, motoki, on=['Id'])
        del motoki

    if 'inter' in config:
        inter = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\train_inter.pkl','rb'))
        train = pd.merge(train, inter, on=['Id'])
        del inter

    if 'qcut' in config:
        temp = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\train_qcut.pkl','rb'))
        train = pd.merge(train, temp, on=['Id'])
        del temp

    if 'type' in config:
        temp = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\train_type.pkl','rb'))
        train = pd.merge(train, temp, on=['Id'])
        del temp

    if 'inter4' in config:
        temp = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\train_inter4.pkl','rb'))
        train = pd.merge(train, temp, on=['Id'])
        del temp

    if 'nextprev' in config:
        temp = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\train_nextprev.pkl','rb'))
        train = pd.merge(train, temp, on=['Id'])
        del temp

    if 'magic2' in config:
        temp = pd.read_csv(r'E:\Git\ML\Kaggle_Bosch\Data\magic2.csv')
        train = pd.merge(train, temp, on=['Id'])

    if 'char' in config:
        temp = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\train_char.pkl','rb'))
        train = pd.merge(train, temp, on=['Id'])
        del temp

    if 'prevnextId' in config:
        temp = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\prevnextId.pkl','rb'))
        train = pd.merge(train, temp, on=['Id'])
        del temp

    if 'time1' in config:
        temp = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\time1.pkl','rb'))
        train = pd.merge(train, temp, on=['Id'])
        del temp
    if 'prevnextnum' in config:
        temp = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\prevnextnum.pkl','rb'))
        train = pd.merge(train, temp, on=['Id'])
        del temp

    if 'numcount' in config:
        temp = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\numcount.pkl','rb'))
        train = pd.merge(train, temp, on=['Id'])
        del temp

    if 'zscale' in config:
        temp = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\zscale.pkl','rb'))
        train = pd.merge(train, temp, on=['Id'])
        del temp
    
    gc.collect()

    
    
    train[colsnum] += 2
    train.fillna(0, inplace=True)
    

    cv = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\cv2.pkl', 'rb'))

    cvdata = train.loc[cv]
    train= train.drop(cv)


    del train['Id']

    if 'basic' in config:
        test = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\test_basic.pkl','rb'))
        #test = test[numeric]

    if 'date' in config:
        date = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\test_minmax.2.pkl','rb'))
        test = pd.merge(test, date, on=['Id'])
        del date

    if 'leak' in config:
        leak = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\test_leak2.pkl','rb'))
        test = pd.merge(test, leak, on=['Id'])
        del leak

    if 'dups' in config:
        dups = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\test_dup2.pkl','rb'))
        test = pd.merge(test, dups, on=['Id'])
        #test['MeanResponse2'] = test['MeanResponse'] * test['SortedIdDiff']
        del dups

    if 'motoki' in config:
        motoki = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\test_motoki.pkl','rb'))

        cols = set(motoki.columns.values)
        cols = set(test.columns.values).intersection(cols)
        cols = [f for f in motoki.columns.values if f not in cols] + ['Id']
        motoki = motoki[cols]

        test = pd.merge(test, motoki, on=['Id'])
        del motoki

    if 'motoki2' in config:
        motoki = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\test_motoki2.pkl','rb'))

        cols = set(motoki.columns.values)
        cols = set(test.columns.values).intersection(cols)
        cols = [f for f in motoki.columns.values if f not in cols] + ['Id']
        motoki = motoki[cols]

        test = pd.merge(test, motoki, on=['Id'])
        del motoki

    if 'inter' in config:
        inter = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\test_inter.pkl','rb'))
        test = pd.merge(test, inter, on=['Id'])
        del inter

    if 'qcut' in config:
        temp= pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\test_qcut.pkl','rb'))
        test = pd.merge(test, temp, on=['Id'])
        del temp

    if 'type' in config:
        temp= pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\test_type.pkl','rb'))
        test = pd.merge(test, temp, on=['Id'])
        del temp
    if 'inter4' in config:
        temp= pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\test_inter4.pkl','rb'))
        test = pd.merge(test, temp, on=['Id'])
        del temp

    if 'nextprev' in config:
        temp= pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\test_nextprev.pkl','rb'))
        test = pd.merge(test, temp, on=['Id'])
        del temp

    if 'magic2' in config:
        temp = pd.read_csv(r'E:\Git\ML\Kaggle_Bosch\Data\magic2.csv')
        test = pd.merge(test, temp, on=['Id'])

    if 'char' in config:
        temp= pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\test_char.pkl','rb'))
        test = pd.merge(test, temp, on=['Id'])
        del temp

    if 'prevnextId' in config:
        temp= pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\prevnextId.pkl','rb'))
        test = pd.merge(test, temp, on=['Id'])
        del temp

    if 'time1' in config:
        temp= pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\time1.pkl','rb'))
        test = pd.merge(test, temp, on=['Id'])
        del temp

    
    if 'prevnextnum' in config:
        temp= pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\prevnextnum.pkl','rb'))
        test = pd.merge(test, temp, on=['Id'])
        del temp

    if 'numcount' in config:
        temp= pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\numcount.pkl','rb'))
        test = pd.merge(test, temp, on=['Id'])
        del temp

    if 'zscale' in config:
        temp= pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\zscale.pkl','rb'))
        test = pd.merge(test, temp, on=['Id'])
        del temp

    test['label'] = test['Id']
    del test['Id']


    test[colsnum] += 2
    test.fillna(0, inplace=True)
    

    return train, cvdata, test





class Coin10Orchestrator(Orchestrator.Orchestrator):
    def __init__(self, dataDir, outputDir, args, trainer, finalPredictFunc, resetData = False, threads = 2, debug = False, getData = None, exceptCols = [], selectCols = None):
        return super().__init__(dataDir, outputDir, args, trainer, finalPredictFunc, resetData, threads, debug, getData, exceptCols, selectCols)


def xgb1():
    p = {}
    #p['base_estimator'] = [linear_model.SGDRegressor]
    p['learning_rate'] = [0.01]
    p['n_estimators'] = [200,300]
    p['max_depth'] = [11,12,13]    
    p['objective'] = ['binary:logistic']
    p['colsample_bytree'] = [0.5, 0.7]
    p['silent'] = [False]
    p['subsample'] = [0.85, 1]
    p['base_score'] = [0.003, 0.002]

    
    p['learning_rate'] = [0.02]
    p['n_estimators'] = [300]
    p['max_depth'] = [11]
    p['objective'] = ['binary:logistic']
    p['colsample_bytree'] = [0.95]
    p['silent'] = [False]
    p['subsample'] = [1]
    p['base_score'] = [0.003]

    
    
    
    #usecols = usecols
    config.add('basic')
    config.add('date')
    config.add('leak')
    config.add('dups')
    #config.add('motoki')
    #config.add('motoki2')
    config.add('inter')
    config.add('magic2')
    config.add('char')
    #config.add('qcut')
    #config.add('type')
    config.add('inter4')
    #config.add('nextprev')
    config.add('prevnextId')
    config.add('time1')
    config.add('prevnextnum')
    #config.add('numcount') #OVERFITS
    config.add('zscale')

    o = BoschOrchestrator(r'E:\Git\ML\Kaggle_Bosch\Data\WithMagic18\\',
                                       r'E:\Git\ML\Kaggle_Bosch\Data\OutputXGBWithMagic18\\', 
                                       p, TrainModel.XGBClassifier, output, 
                                       resetData=False, threads=1, debug=True, 
                                       getData=processNumCat, selectCols=None, exceptCols=[])



    
    o.train()



if __name__ == '__main__':  
    xgb1()







