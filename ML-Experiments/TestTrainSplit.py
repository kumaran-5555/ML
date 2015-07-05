#!/usr/bin/python3
import sys
import random
import os

if len(sys.argv) != 4:
    print('Usage: {0} <inputFile> <testPercentage> <numOfRows> '.format(sys.argv[0]))
    print(sys.argv)
    sys.exit(1)

    

samples = []
numSamples = int(sys.argv[3])
file = open(sys.argv[1], 'r')

indices = [i for i in range(numSamples)]
testPercentage = float(sys.argv[2])
#print('started sampling ...')
testIndices = random.sample(range(numSamples), int(numSamples * testPercentage))
trainIndices = []
#print('test samples ', len(testIndices))


testIndicesDict = {}
for i in testIndices:
    testIndicesDict[i] = 1

for i in range(numSamples):
    if i not in testIndicesDict:
        trainIndices.append(i)
#print('train samples ', len(trainIndices))
testIndices = sorted(testIndices)
trainIndices = sorted(trainIndices)
lenTest = len(testIndices)
lenTrain = len(trainIndices)

count = 0
cursor = 0
for l in file:
    if not count % 2000:
        #print('Processed ', count, ' rows...')
        pass
    if cursor < lenTest and testIndices[cursor] == count:
        sys.stdout.write(l)
        cursor += 1
    count += 1

file.seek(0, 0)

count = 0
cursor = 0
for l in file:
    if not count % 2000:
        #print('Processed ', count, ' rows...')
        pass
    if cursor < lenTrain and trainIndices[cursor] == count:
        sys.stderr.write(l)
        cursor += 1
    count += 1

