#!/usr/bin/python
__author__ = 'serajago'

import sys
import sklearn
import sklearn.datasets
import sklearn.tree
import sklearn.ensemble
import sklearn.cross_validation
import sklearn.metrics
import numpy
import sklearn.linear_model

f = open("E:\\Projects\\DifForAggre\\Training\\v1\\en_inTrainDif.txt.features.output.txt.v1.txt", 'r', encoding='utf-8')

trainingData = []
trainingLabel = []
trainingQuery = []
count = 0
for l in f:
    count += 1

    l = l.replace('\r','').replace('\n','')
    fields = l.split('\t')
    if count == 1:
        numFields = len(fields)
        continue
    if len(fields) != numFields:
        continue
    if fields[1] != '0' and fields[1] != '1':
        continue

    trainingLabel.append(fields[1])
    trainingQuery.append(fields[0])
    trainingData.append(fields[2:])


#print(trainingData)
numData = numpy.array(trainingData, dtype=float)
numLabel = numpy.array(trainingLabel, dtype=int)

print(numData.shape)
print(numLabel.shape)

model = sklearn.tree.DecisionTreeClassifier()
model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=250, max_leaf_nodes=20, min_samples_split=5)
#model = sklearn.ensemble.AdaBoostClassifier(base_estimator=sklearn.tree.DecisionTreeClassifier(), n_estimators=200)
#model = sklearn.svm.SVC()

#model = sklearn.tree.DecisionTreeRegressor()
#model = sklearn.linear_model.LogisticRegression()
model.fit(numData, numLabel)
#print(sklearn.cross_validation.cross_val_score(model, numData, numLabel))



f = open("E:\\Projects\\DifForAggre\\Training\\v1\\eset.txt.features.txt.v1.txt", 'r', encoding='utf-8')

trainingData = []
trainingLabel = []
trainingQuery = []
sampleWeight = []
count = 0
for l in f:
    count += 1

    l = l.replace('\r','').replace('\n','')
    fields = l.split('\t')
    if count == 1:
        numFields = len(fields)
        continue
    if len(fields) != numFields:
        continue
    if fields[1] != '0' and fields[1] != '1':
        continue

    if fields[1] == '0':
        sampleWeight.append(4.685)
    else:
        sampleWeight.append(1)


    trainingLabel.append(fields[1])
    trainingQuery.append(fields[0])
    trainingData.append(fields[2:])


#print(trainingData)
numData = numpy.array(trainingData, dtype=float)
numLabel = numpy.array(trainingLabel, dtype=int)

scores = numpy.array(model.predict_proba(numData))[:,1]
p,r,t = sklearn.metrics.precision_recall_curve(numLabel, scores, sample_weight=sampleWeight)
for i in range(len(t)):
    print(p[i],'\t',r[i],'\t',t[i])
sys.exit(0)















