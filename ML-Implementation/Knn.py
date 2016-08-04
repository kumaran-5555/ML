from numpy import *
from collections import defaultdict



class KNN:
    def __init__(self, k):
        self.k =  k


    def fit(self, dataX, dataY):
        if dataX.shape[0] < self.k:
            raise ValueError("Should have atleast k rows to Knn at k")

        self.dataX = dataX
        self.dataY = dataY
        
        
    def predict(self, point):

        distance = self.distance(point)
        indices = distance.argsort()

        count = defaultdict(lambda : 0)
        for i in range(self.k):
            count[self.dataY[indices[i]]]+=1

        sortedClasses = sorted(count.items(), key=lambda x: x[1], reverse=True)

        return sortedClasses[0][0]


    def distance(self, point):
        '''
               distace = sqrt((x1-x2)**2)
               array of vectors 
        '''

        aMinusB = self.dataX - tile(point, (self.dataX.shape[0], 1)) 
        aMinusBSqr = aMinusB ** 2
        aMinusBSqrSum = aMinusBSqr.sum(axis=1)
        aMinusBSqrSumSqrt = aMinusBSqrSum ** 0.5


        return aMinusBSqrSumSqrt


        
if __name__ == '__main__':

    data = loadtxt(r"E:\Git\ML\Data\optdigits.train.csv", delimiter=',')
    dataX = data[:,0:-1]
    dataY = data[:,-1]

    c = KNN(10)
    c.fit(dataX, dataY)

    test = loadtxt(r"E:\Git\ML\Data\optdigits.test.csv", delimiter=',')

    testX = test[:,0:-1]
    testY = test[:,-1]

    tp = 0
    fp = 0

    for i in range(testX.shape[0]):
        predicted = c.predict(testX[i])

        if predicted == testY[i]:
            tp += 1
        else:
            fp += 1


    print("Error {} TP {} FP {}".format(tp / (tp+fp), tp, fp))







