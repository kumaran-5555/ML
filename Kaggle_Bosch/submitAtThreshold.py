#!/usr/bin/python3
import pandas as pd
import numpy as np
import sys

def update(fileName, threshold):
	threshold = float(threshold)
	scores = pd.read_csv(fileName, 'rb')
	scores['Response'] =  0
	scores['Response'][scores['prob'] > 1] = 1
	scores[['Id','Response']].to_csv(fileName+str(threshold), delimiter=',', header=False, index=False)



if __name__=='__main__':
	if len(sys.args) != 3:
		print("Usage: {0} <inputFile> <threshold>", sys.args[0])
		sys.exit(1)
	update(sys.args[1], sys.args[2])




	

