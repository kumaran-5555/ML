import FeatureHasherWithoutLabel
import sklearn.svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
import sys
import numpy
import scipy.sparse
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
import ThreadedSVMSweep


if __name__ == '__main__':
    comment='''
    v1 - weight 1:2, one chunk negative with 0.02% from all negative
    '''

    print('Creating vectors ...')
    projectName = 'auto_researcher'
    projectName = 'financial_service'
    iteration ='intermediate2'
    iteration ='intermediate'

    Xp,Yp,id = FeatureHasherWithoutLabel.process(r"D:\Hack\OneMLDisplayAdsHackathon\\"+ projectName + r"\\" + iteration + r"\combined_train.train.pos.dat")
    Xp2,Yp2,id = FeatureHasherWithoutLabel.process(r"D:\Hack\OneMLDisplayAdsHackathon\\"+ projectName + r"\\" + iteration + r"\combined_train.test.pos.dat")
    Xn,Yn,id = FeatureHasherWithoutLabel.process(r"D:\Hack\OneMLDisplayAdsHackathon\\"+ projectName + r"\\" + iteration + r"\combined_train.train.neg_1.dat")
    Xn2,Yn2,id = FeatureHasherWithoutLabel.process(r"D:\Hack\OneMLDisplayAdsHackathon\\"+ projectName + r"\\" + iteration + r"\combined_train.train.neg_2.dat")
    Xn3,Yn3,id = FeatureHasherWithoutLabel.process(r"D:\Hack\OneMLDisplayAdsHackathon\\"+ projectName + r"\\" + iteration + r"\combined_train.train.neg_3.dat")
    Xn4,Yn4,id = FeatureHasherWithoutLabel.process(r"D:\Hack\OneMLDisplayAdsHackathon\\"+ projectName + r"\\" + iteration + r"\combined_train.train.neg_4.dat")
    Xtest,Ytest,id = FeatureHasherWithoutLabel.process(r"D:\Hack\OneMLDisplayAdsHackathon\\"+ projectName + r"\\" + iteration + r"\combined_train.test.dat")

    Xsubtest,Ysubtest, IDsubtest = FeatureHasherWithoutLabel.process(r"D:\Hack\OneMLDisplayAdsHackathon\\"+ projectName + r"\\" + iteration + r"\combined_test.norm.dat")
    Xsubtrain,Ysubtrain, IDsubtrain = FeatureHasherWithoutLabel.process(r"D:\Hack\OneMLDisplayAdsHackathon\\"+ projectName + r"\\" + iteration + r"\combined_train.norm.dat")
    outputFile= open(r"D:\Hack\OneMLDisplayAdsHackathon\\"+ projectName + r"\\" + iteration + r"\scorev7.txt", 'w')
    #trainOut= open(r"D:\Hack\OneMLDisplayAdsHackathon\auto_researcher\intermediate\final.trainscores.tsv", 'w')

    Xtrain_ = scipy.sparse.vstack((Xp,Xn,Xn2,Xp2))
    #Xtrain_ = scipy.sparse.vstack((Xp,Xn,Xn2))
    Ytrain_ = numpy.hstack((Yp,Yn,Yn2,Yp2))
    #Ytrain_ = numpy.hstack((Yp,Yn,Yn2))
    Xtrain,Ytrain = shuffle(Xtrain_,Ytrain_)

    print('Training vectors Done. train shape {0} test shape {0}'.format(Xtrain.shape, Xtest.shape))
    print('Training modesl ...')


    c = [2**i for i in range(-5, 4)]
    w = [i/4 for i in range(8, 20)]
    g = [2**i for i in range(-10, 3)]
    
    c = [4]
    w = [3]
    g = [0.00390625]
    c = [0.125]
    w = [1.0]
    g = [0.03125]
    
    output = open(r"D:\Hack\sweep.tmp.txt", 'w')
    model, auc = ThreadedSVMSweep.threadedSVMSweep(c, w, g, Xtrain, Ytrain, Xtest, Ytest, output)
    print(model, auc)


    
    
    print('Predicting test scores ...')
    
    scores = model.predict_proba(Xsubtest)[:,1]
    print(IDsubtest.shape, scores.shape)
    for i in range(len(scores)):
        outputFile.write('{0},{1}\n'.format(IDsubtest[i], scores[i]))

    print('Predicting train scores ...')

    scores = model.predict_proba(Xsubtrain)[:,1]
    for i in range(len(scores)):
        outputFile.write('{0},{1}\n'.format(IDsubtrain[i], scores[i]))

    outputFile.close()
    
    