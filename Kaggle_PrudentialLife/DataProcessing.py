
import numpy as np
import sys
import Utils
import sklearn.preprocessing.imputation



class DataProcessing:
    def __init__(self):
        self.dirName = r'D:\Data\Kaggle_PrudentialLifeInsuranceAssessment\\'
        self.trainingFile = open(self.dirName + r'train\train.csv', 'r', encoding='utf-8')
        self.testFile = open(self.dirName + r'test\test.csv', 'r', encoding='utf-8')

        self.trainX = []
        self.testX = []
        self.trainY = []
        self.header = {}
        

        isHeader = True
        for l in self.trainingFile:
            fields = l[:-1].split(',')
            if isHeader:
                for i in range(len(fields)):
                    self.header[fields[i].replace('"','')] = i
                isHeader = False
                continue

            


            self.trainX.append(fields[:-1])
            self.trainY.append(fields[-1])

        isHeader = True
        for l in self.testFile:
            fields = l[:-1].split(',')
            if isHeader:
                isHeader = False
                continue

            self.testX.append(fields[:-1])


        print(self)


    def __str__(self):
        return 'test shape {} train shape {}  target shape {} \n header {}'.format(\
            len(self.testX), len(self.trainX[0]), len(self.trainY), sorted(self.header, key=self.header.get))
            

    def preprocessorV1(self):

        Utils.RemoveColumns(self, ['Id'])
        print(self)
        Utils.NumberizeCategoryFeatures(self, ['Product_Info_2'])
        print(self)

        Utils.ReplaceMissingValuesWithNone(self)

        print(self)

        # impute values
        
        trainX = np.array(self.trainX)
        testX = np.array(self.testX)

        trainY = np.array(self.trainY)


        imputer = sklearn.preprocessing.Imputer(missing_values='NaN', strategy='mode')

        imputer.fit(trainX)

        trainX = imputer.transform(trainX)
        testX = imputer.transform(testX)


        print(self)

        






if __name__ == '__main__':
    d = DataProcessing()
    d.preprocessorV1()
