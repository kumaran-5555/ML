#!/usr/bin/python3
import sys
import random
import os

if len(sys.argv) != 3:
    print('Usage: {0} <inputFile> <samplePercentage>'.format(sys.argv[0]))
    sys.exit(1)



samples = []
numSamples = 0
with open(sys.argv[1], 'r') as file:
    for l in file:
        numSamples += 1
        samples.append(l)
indices = [i for i in range(numSamples)]
testPercentage = float(sys.argv[2])
testIndices = random.sample(indices, int(numSamples * testPercentage))

for i in testIndices:
    sys.stdout.write(samples[i])

