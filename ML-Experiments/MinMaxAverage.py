#!/usr/bin/python3

from collections import defaultdict
import sys
import os



if len(sys.argv) != 3:
		print('Usage: {0} <inputTestFile inputTrainFile>'.format(sys.argv[0]))
		sys.exit(1)

inputTestFile = open(sys.argv[1], 'r')
inputTrainFile = open(sys.argv[2], 'r')

featureMin = defaultdict(lambda :float('inf'))
featureMax = defaultdict(lambda :float('-inf'))


for l in inputTestFile:
		fields = l[:-1].split('\t')
		k = fields[2]
		v = float(fields[3])
		if featureMin[k] > v:
				featureMin[k] = v
		if featureMax[k] < v:
				featureMax[k] = v


for l in inputTrainFile:
		fields = l[:-1].split('\t')
		k = fields[2]
		v = float(fields[3])
		if featureMin[k] > v:
				featureMin[k] = v
		if featureMax[k] < v:
				featureMax[k] = v


for k in featureMin:
		avg = (featureMin[k] + featureMax[k] ) / 2
		if not avg:
				avg = 1e-10
		print('{0}\t{1}\t{2}\t{3}'.format(k, featureMin[k], featureMax[k], avg))



