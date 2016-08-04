import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics
from sklearn import ensemble
import gc
from sktlc import ensemble as tlcensemble




import pickle




print("Here")

'''
cvtest = np.genfromtxt(r"E:\Git\ML\Kaggle_Grupo\Data\cvtest49.tsv", \
    delimiter='\t', usecols=(13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28), comments="m:", missing_values=0)

pickle.dump(cvtest, open(r"E:\Git\ML\Kaggle_Grupo\Data\cvtest49.pkl", 'wb'))
print(cvtest.shape)

train = np.loadtxt(r"E:\Git\ML\Kaggle_Grupo\Data\trainfinal.tsv", \
    delimiter='\t', usecols=(13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28), comments="m:")
gc.collect()


print(train.shape)

pickle.dump(train, open(r"E:\Git\ML\Kaggle_Grupo\Data\trainfinal.pkl", 'wb'))
'''

train = pickle.load(open(r"E:\Git\ML\Kaggle_Grupo\Data\trainfinal.2.pkl", 'rb'))
cvtest = pickle.load(open(r"E:\Git\ML\Kaggle_Grupo\Data\cvtest1.2.pkl", 'rb'))
test = pickle.load(open(r"E:\Git\ML\Kaggle_Grupo\Data\test.2.pkl", 'rb'))


#[2,11,12,13,14,15,16,17,18,19,20]
xTrain = train[:,[4,11,12,13,14,15,16,17,18,19,20]]
yTrain = train[:,0]

xTest = cvtest[:,[4,11,12,13,14,15,16,17,18,19,20]]
yTest = cvtest[:,0]

xFinal = test[:,[4,11,12,13,14,15,16,17,18,19,20]]
id = test[:,0]




preproc = preprocessing.StandardScaler()
print("Started preprocessing")

xTrain = preproc.fit_transform(xTrain)
xTest = preproc.transform(xTest)
xFinal = preproc.transform(xFinal)







def TLCFastTree():
    ft = tlcensemble.auto_TlcFastTreeRegression.TlcFastTreeRegression(mb=512)

    ft.fit(xTrain, yTrain)
    print(metrics.mean_squared_error(yTest, ft.predict(xTest)))


def SGD():
    for a in [0.45, 1, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.15]:
    #for a in [100, 50, 30, 20, 10, 7, 5, 2, 1]:
        print("Started training for ", a)
        lr = linear_model.SGDRegressor(alpha=a)
        
    
        lr.fit(xTrain, yTrain)
        print("Started measurement")
        pred = lr.predict(xTest)
        print(metrics.mean_squared_error(yTest, pred))
   
        for i in range(pred.shape[0]):
            if xTest[i,0] > pred[i] + 0.5:
                pred[i] = xTest[i, 0]

        print(metrics.mean_squared_error(yTest, pred))
        
        pred = lr.predict(xFinal)

        for i in range(pred.shape[0]):
            if xFinal[i,0] > pred[i] + 0.5:
                pred[i] = xFinal[i, 0]
        
        pred = lr.predict(xFinal)

        final = np.stack((id.T, np.expm1(pred).T)).T

        np.savetxt(r"E:\Git\ML\Kaggle_Grupo\Data\submit6.tsv", final,delimiter=',', fmt="%d,%f")


        break

SGD()
#TLCFastTree()




