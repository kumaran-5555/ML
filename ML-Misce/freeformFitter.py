#!/usr/bin/python
__author__ = 'serajago'

import sys

f = open('E:\\Projects\\DifForAggre\\Training\\v3\\eset.txt.features.output.txt.freeform.txt', 'r',encoding='utf-8')

isFirst = True
data = []

for l in f:
    if isFirst:
        isFirst = False
        continue

    fields = l.split('\t')
    try:
        data.append([fields[0],int(fields[1]),float(fields[2]), float(fields[3]), float(fields[4]), 0])
    except Exception:
        continue
#m:RawQuery      m:Label PBADIFMeta      PBANegativeIntent       NameGrammarV2s
boost = 0
negativeIntentThreshold = 0
tp = 0
fp = 0
fn = 0
difThreshold = 0.50
weight = 4.685
tn = 0

while boost < 1:
    while negativeIntentThreshold < 1:
        for d in data:
            if d[3] > negativeIntentThreshold and d[4] > 0:
                d[-1] = d[2] + boost
            else:
                d[-1] = d[2]

            if d[-1] > difThreshold and d[1] == 1:
                tp += 1
            elif d[-1] <= difThreshold and d[1] == 1:
                fn += 1
            elif d[-1] > difThreshold and d[1] == 0:
                fp += 4.685
            else:
                tn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print ("%.2f\t%.2f\t%.2f\t%.2f\t%d\t%d\t%d\t%d"%(boost,negativeIntentThreshold,  precision, recall, tp, fn, tn, fp))

        tp = 0
        fp = 0
        fn = 0
        tn = 0
        negativeIntentThreshold += 0.05
    boost += 0.05
    negativeIntentThreshold = 0








