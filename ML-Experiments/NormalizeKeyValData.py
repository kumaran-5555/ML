#!/usr/bin/python3

#import sklearn.feature_extraction
import sys
#import numpy

if len(sys.argv) != 2:
    print('Usage : {0} <inputFile>'.format(sys.argv[0]))
    sys.exit(1)

inputFile = open(sys.argv[1], 'r', encoding='utf-8')

def generateRows():
    features = []
    prevKey = ''
    
    for l in inputFile:
        fields = l[:-1].split('\t')
        label = fields[0]        
        fields = fields[1:]
        if prevKey != fields[0] and prevKey != '':
            yield '\t'.join(features)
            features = []
            prevKey = fields[0]
            features.append(label)
            features.append(prevKey)
            features.append(fields[1]+':'+fields[2])
            
        else:
            if prevKey == '':
                prevKey = fields[0]
                features.append(label)
                features.append(prevKey)

            features.append(fields[1]+':'+fields[2])

for line in generateRows():
    print(line)

