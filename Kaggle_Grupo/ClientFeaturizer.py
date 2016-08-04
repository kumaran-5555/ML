from sklearn.feature_extraction import FeatureHasher

import sys
import Kaggle_Grupo

import numpy as np
import sktlc.linear_model.auto_TlcOLSLinearRegression





class ClientFeaturizer:
    def __init__(self, inputFile, outputFile, hashWidth, encodeWidth, numberOfFeatures):
        self.inputFile = open(inputFile, 'r', encoding='utf-8')

        self.outputFile = open(outputFile, 'w', encoding='utf-8')

        self.hashWidth = hashWidth
        self.encodeWidth = encodeWidth

        self.numberOfFeatures = numberOfFeatures



    def encode(self, width=16):
        new = np.zeros((self.features.shape[0], self.features.shape[1]/width))

        for i in range(self.features.shape[0]):

            # encode each row
            
            for j in range(0, int(self.features.shape[1]/width)):
                value = 0
                shift = 0
                for k in range(width):
                    value |= ((self.features[i, j*width+k] & 1)<<shift)
                    shift += 1
                new[i, j] = value



        return new
                 

    def HashEncoder(self, features):

        for f in features:

        


    def process(self):

        header = self.inputFile.readline()

        ids = []
        self.features = []

        count = 0
        for line in self.inputFile:
            count += 1
            fields = line.split(',')

            id = fields[0]
            names = {}
            name = Kaggle_Grupo.Utils.StringNormalize(fields[1])

            for i in name.split(' '):
                names[i] = 1

            ids.append(id)
            self.features.append(names)




        featureHasher = FeatureHasher(n_features=2**12, dtype=np.uint16)

        self.features = featureHasher.transform(self.features)
        self.features = self.features.toarray()

        self.features = self.encode(width=24)


        headerFields  = ["Cliente_ID"]

        for i in range(self.features.shape[1]):
            headerFields.append('ClientName_{}'.format(i))

        headerFields = "\t".join(headerFields)


        self.outputFile.write(headerFields+'\n')

        for i in range(self.features.shape[0]):
            self.outputFile.write('{}\t{}\n'.format(ids[i], ('\t'.join(self.features[i].astype('str')).replace('False', '0').replace('True', '1'))))


        #np.savetxt(self.outputFile, features, fmt="%d", delimiter='\t', newline='\n', header=headerFields)












        

