#!/cygdrive/c/Users/serajago/Downloads/WinPython-64bit-3.4.3.3/python-3.4.3.amd64/python.exe

import sklearn.feature_extraction
import sys
import numpy
import os.path
import pickle

hasher = sklearn.feature_extraction.FeatureHasher(n_features=2**15, input_type='dict')


# file has 4 cols, label\trowId\tk1:v1\tk2:v2\tkn:vn
labels = []
rowIds = []


def generateRows(inputFile): 
    global labels
    global rowIds       
    labels = []
    rowIds = []
    count = 0
    for l in inputFile:
        count += 1
        if count % 1000 ==0:
            print('Vectorized ', count, ' rows...') 
        tempDict = {}
        fields = l[:-1].split('\t')
        try:
            temp = float(fields[0])
            labels.append(fields[0])
            rowIds.append(fields[1])

            for i in range(2, len(fields)):
                try:
                    k = fields[i].split(':')[0]
                    v = fields[i].split(':')[1]
                    tempDict[k] = float(v)
                except Exception:
                    continue
            yield tempDict
        except Exception:
            print('ignore row due to exception', inputFile.name, '<', l, '>')
            continue

def process(inputFileName):
    pickleFileName = inputFileName + '.pkl'
    if not os.path.isfile(pickleFileName):
        print('pickle doesn\'t exists for {0}, creating new'.format(inputFileName ))
        inputFile = open(inputFileName, 'r', encoding='utf-8')
        hashedFeatures = hasher.transform(generateRows(inputFile))
        print('features shape ', hashedFeatures.shape)
        labelsArray = numpy.array(labels)
        rowIdsArray = numpy.array(rowIds)
        
        print('storing objects ...')
        pickle.dump(hashedFeatures, open(pickleFileName, 'wb'))
        pickle.dump(labelsArray, open(pickleFileName +'lbl', 'wb'))
        pickle.dump(rowIdsArray, open(pickleFileName +'ids', 'wb'))
    else:
        print('pickle exists for {0}, extracting from it'.format(inputFileName ))
        hashedFeatures = pickle.load(open(pickleFileName, 'rb'))
        labelsArray = pickle.load(open(pickleFileName +'lbl', 'rb'))
        rowIdsArray = pickle.load(open(pickleFileName +'ids', 'rb'))
        print(hashedFeatures.shape)
        print(labelsArray.shape)
        print(rowIdsArray.shape)
        
        
        
        

    return hashedFeatures,labelsArray,rowIdsArray

