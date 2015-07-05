#!/usr/bin/python3
import sys
import random

if len(sys.argv) != 2:
    print('Usage: {0} <inputFileName>'.format(sys.argv[0]))
    sys.exit(2)

file = open(sys.argv[1], 'r', encoding='utf-8')

for l in file:
    fields = l[:-1].split('\t')
    # count without id and label
    numFeatures = len(fields) - 2
    # create pesudo ID
    fields[1] = fields[1] + '_p'
    idxToRemove = random.randint(0, numFeatures-1)
    fields.remove(fields[idxToRemove+2])
    print('\t'.join(fields))

    


