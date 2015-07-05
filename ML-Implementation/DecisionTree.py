import numpy
import random
import pickle
import json
from sklearn import datasets
import DataSets
from sklearn import metrics



class RegressionCriterion:
    def __init__(self, start, end, n_classes, label_y, sample_weight_y, samples, total_training_weight):
        self.start = start
        self.end = end
        self.n_classes = n_classes
        self.sum_left = 0
        self.sum_right = 0
        self.sum_total = 0
        self.mean_left = 0
        self.mean_right = 0
        self.mean_total =  0
        self.sq_sum_left = 0
        self.sq_sum_right = 0
        self.sq_sum_total = 0
        self.var_left = 0
        self.var_right = 0

        
        # data[:pos] is in left, and data[pos:] is in right
        self.curr_pos = start
        self.sample_weight_y = sample_weight_y
        self.label_y = label_y
        self.total_training_weight = total_training_weight
        self.total_weight = 0
        self.total_left_weight = 0
        self.total_right_weight = 0
        self.samples = samples

        for p in range(start, end):
            i = self.samples[p]
            w = sample_weight_y[i]
            c = label_y[i]
            self.total_weight += w
            self.sum_total += (c * w)
            self.sq_sum_total += (c * c * w)



        self.total_right_weight = self.total_weight
        self.mean_total = self.sum_total / self.total_weight 

        self.total_right_weight = self.total_weight 
        self.mean_right = self.mean_total
        self.sq_sum_right = self.sq_sum_total
        self.sum_right = self.sum_total
        self.var_right = (self.sq_sum_right / self.total_right_weight) - (self.mean_right * self.mean_right)

    def reset(self):
        ''' 
            called inside a split region before doing another
            feature try

            just reset left to no samples and right to all samples
        '''
        self.total_right_weight = self.total_weight 
        self.total_left_weight = 0.0
        self.mean_right = self.mean_total
        self.mean_left = 0.0
        self.sq_sum_right = self.sq_sum_total
        self.sq_sum_left = 0.0
        self.sum_right = self.sum_total
        self.sum_left = 0.0
        self.var_right = (self.sq_sum_right / self.total_right_weight) - (self.mean_right * self.mean_right)
        self.var_left = 0.0
        self.curr_pos = self.start


    def update(self, new_position):
        diff = 0
        for p in range(self.curr_pos, new_position):
            i = self.samples[p]
            w = self.sample_weight_y[i]
            c = self.label_y[i]
                        
            self.sum_left += (w * c)
            self.sum_right -= (w * c)
            self.sq_sum_left += (w * c * c)
            self.sq_sum_right -= (w * c * c)
            diff += w

        self.total_left_weight += diff
        self.total_right_weight -= diff

        self.mean_left = self.sum_left / self.total_left_weight
        self.mean_right = self.sum_right / self.total_right_weight
        self.var_left = (self.sq_sum_left / self.total_left_weight) - (self.mean_left * self.mean_left)
        self.var_right = (self.sq_sum_right / self.total_right_weight) - (self.mean_right * self.mean_right)

        self.curr_pos = new_position

    def impurity(self):
        return (self.sq_sum_total / self.total_weight) - (self.mean_total * self.mean_total)

    def child_impurity(self):
        return self.var_left,self.var_right

    def improvement(self):
        diff = (self.sum_left / self.total_left_weight) - (self.sum_right / self.total_right_weight)
        return (self.total_left_weight * self.total_right_weight * diff *  diff) / \
            (self.total_left_weight + self.total_right_weight)

    def node_value(self):
        return self.mean_total
        



# test commenting
class GiniImpurity:    
    def __init__(self, start, end, n_classes, label_y, sample_weight_y, samples, total_training_weight):
        self.start = start
        self.end = end
        self.n_classes = n_classes
        self.label_count = numpy.zeros(n_classes)
        self.label_count_left = numpy.zeros(n_classes)
        self.label_count_right = numpy.zeros(n_classes)
        # data[:pos] is in left, and data[pos:] is in right
        self.curr_pos = start
        self.sample_weight_y = sample_weight_y
        self.label_y = label_y
        self.total_training_weight = total_training_weight
        self.total_weight = 0
        self.total_left_weight = 0
        self.total_right_weight = 0
        self.samples = samples

        for p in range(start, end):
            i = self.samples[p]
            w = sample_weight_y[i]
            c = label_y[i]
            self.label_count[c] += w
            self.label_count_right[c] += w
            self.total_weight += w
        self.total_right_weight = self.total_weight

    def reset(self):
        ''' 
            called inside a split region before doing another
            feature try

            just reset left to no samples and right to all samples
        '''
        self.total_right_weight = self.total_weight
        self.total_left_weight = 0.0
        self.label_count_left = numpy.zeros(self.n_classes)
        self.label_count_right = numpy.copy(self.label_count)
        self.curr_pos = self.start




    def update(self, new_position):
        diff = 0
        for p in range(self.curr_pos, new_position):
            i = self.samples[p]

            w = self.sample_weight_y[i]
            c = self.label_y[i]
            self.label_count_left[c] += w
            self.label_count_right[c] -= w
            diff += w

        self.total_right_weight -= diff
        self.total_left_weight += diff
        self.curr_pos = new_position

    
    def impurity(self):
        temp = 0
        for i in range(self.n_classes):
            temp += (self.label_count[i] * self.label_count[i])

        return 1.0 - temp / (self.total_weight * self.total_weight)


    def child_impurity(self):
        left = 0
        right = 0
        for i in range(self.n_classes):
            temp = self.label_count_left[i] 
            left += temp * temp
            temp = self.label_count_right[i]
            right += temp * temp

        v1,v2 = 1.0 - left / (self.total_left_weight * self.total_left_weight), \
            1.0 - right / (self.total_right_weight * self.total_right_weight)
        #print(self.curr_pos, v1, v2)
        return v1,v2

    def improvement(self):
        '''
        Weighted impurity improvement, i.e.
           N_t / N * (impurity - N_t_L / N_t * left impurity
                               - N_t_L / N_t * right impurity),
           where N is the total number of samples, N_t is the number of samples
           in the current node, N_t_L is the number of samples in the left
           child and N_t_R is the number of samples in the right child.
        '''
        left_impurity,right_impurity = self.child_impurity()
        imp = (self.total_weight / self.total_training_weight * self.impurity()) -  \
            ((self.total_left_weight / self.total_training_weight * left_impurity) + \
            (self.total_right_weight / self.total_training_weight * right_impurity))
        #print(self.curr_pos, imp)
        return imp

    def node_value(self):
        node_val = [None] * self.n_classes
        for i in range(self.n_classes):
            node_val[i] = self.label_count[i]
        return node_val


class Splitter2:
    def __init__(self, start, end, nClasses, dataX, dataY, sampleWeight, samples, totalTrainingWeight, criterion, minSamplesLeaf, minWeightLeaf, minSamplesSplit):
        self.start = start
        self.end = end
        self.nClasses = nClasses
        self.dataX = dataX
        self.dataY = dataY
        self.sampleWeight = sampleWeight
        self.samples = samples
        self.totalTrainingWeight = totalTrainingWeight
        self.criterion = criterion
        self.minSamplesLeaf = minSamplesLeaf
        self.minWeightLeaf = minWeightLeaf
        self.nSamples = self.end - self.start
        self.minSamplesSplit = minSamplesSplit
        self.nFeatures = self.dataX.shape[1]
        self.impurityThreshold = 1e-7
        self.featureValueThreshold = 1e-7


    def split(self):
        '''
            decides best split feature and split value, also rearrages samples vector according to split point
            returns tuple, tuple[0] is split status True - split done, False didn't split
            tuple[1] - node impurity (returned for both Splita and Non_Split)
            tuple[2] - feature index of split
            tuple[3] - feature value of split 
            tuple[4] - sample index of split i,e. [start: tuple[3]] - leaf child, [tuple[3], end] - right child
            tuple[5] - left impurity
            tuple[6] - right impurity
        '''

        # initialize criterion
        criterion = self.criterion(self.start, self.end, self.nClasses, self.dataY, self.sampleWeight, self.samples, self.totalTrainingWeight)
        bestImprovement = float('-inf')
        bestFeatureIdx = -1
        bestSplitValue = None
        bestSampleIdx = -1
        leftImpurity = None
        rightImpurity = None

        # check if the node can be split or not
        if self.nSamples <  (2 * self.minSamplesLeaf) or self.nSamples < self.minSamplesSplit or criterion.impurity() < self.impurityThreshold:
            return (False, criterion)


        # node has enough samples to split, but may be pure enought not to split
        for f in range(self.nFeatures):
            # reset the criterion
            criterion.reset()
            subSamples = self.samples[self.start:self.end]
            featureValueSlice = self.dataX.take(subSamples, axis=0)[:,f]
            sortedIndices = numpy.argsort(featureValueSlice)
            sortedSubSamples = [subSamples[i] for i in sortedIndices]
            self.samples[self.start:self.end] = sortedSubSamples
            featureValueSlice = numpy.sort(featureValueSlice)

            # choose best split point
            for i in range(1, self.nSamples):
                if (featureValueSlice[i] - featureValueSlice[i-1])  < self.featureValueThreshold:
                    continue
                # not enough samples on left
                if i < self.minSamplesLeaf:
                    continue

                criterion.update(i + self.start)

                if bestImprovement <= criterion.improvement():
                    bestImprovement = criterion.improvement()
                    bestFeatureIdx = f
                    bestSplitValue = (featureValueSlice[i] + featureValueSlice[i-1]) / 2
                    bestSampleIdx = self.start + i

        if bestFeatureIdx is None:
            return (False, criterion)

        s = self.start
        e = self.end

        while s < e:             
            if self.dataX[self.samples[s], bestFeatureIdx] <= bestSplitValue:
                # on the right side
                s += 1
                continue
            else:
                e -= 1
                # have to move to right side
                tmp = self.samples[e]
                self.samples[e] = self.samples[s]
                self.samples[s] = tmp

        return (True, criterion, bestFeatureIdx, bestSplitValue, bestSampleIdx)


class Tree():
    def __init__(self, nodeId, start, end, nodeValue, nodeImpurity, isLeaf, splitFeature, splitValue, leftId, rightId):
        self.nodeId = nodeId
        self.start = start
        self.end = end
        self.nodeValue = nodeValue
        self.nodeImpurity = nodeImpurity
        self.isLeaf = isLeaf
        self.splitFeature = splitFeature
        self.splitValue = splitValue
        self.leftId = leftId
        self.rightId = rightId

class Constants():
    Classifier = 'Classifier'
    Regression = 'Regression'

        


class DecisionTreeBuilder():
    def __init__(self, type, dataX, dataY, nClasses, sampleWeight, minSamplesLeaf, minSamplesSplit, minWeightLeaf, maxDepth):
        
        self.dataX = dataX
        self.dataY = dataY
        self.nClasses = nClasses
        self.nSamples = dataX.shape[0]
        if not sampleWeight:
            self.sampleWeight = numpy.ones((self.nSamples))
        else:
            self.sampleWeight = sampleWeight

        self.totalTrainingWeight = sum(self.sampleWeight)
        self.minSamplesLeaf = minSamplesLeaf
        self.minSamplesSplit = minSamplesSplit
        if maxDepth is None:
            self.maxDepth = float('inf')
        else:
            self.maxDepth = maxDepth

        self.minWeightLeaf = minWeightLeaf
        self.samples = [i for i in range(self.nSamples)]
        self.nodeId = 0
        if type == Constants.Classifier:
            self.impurity = GiniImpurity
            self.type = Constants.Classifier
        elif type == Constants.Regression:
            self.impurity = RegressionCriterion
            self.type = Constants.Regression
        else:
            print('Invalid tree type ',type)
            raise ValueError

        self.stack = []
        self.treeDict = {}

    def fit(self):

        # stack record (start, end, nodeId, depth)
        self.stack.append((0, self.nSamples, self.nodeId, 0))

        while len(self.stack):
            record = self.stack.pop()

                            
            splitter = Splitter2(record[0], record[1], self.nClasses, self.dataX, self.dataY, self.sampleWeight, self.samples, self.totalTrainingWeight, self.impurity, self.minSamplesLeaf, self.minWeightLeaf, self.minSamplesSplit)
            # depth is still less than max depth
            splitStatus = splitter.split()
            criterion = splitStatus[1]
            hasSplit =  splitStatus[0]
            
            # consider the split only depth allows it
            if record[3] >= self.maxDepth:
                hasSplit = False


            if hasSplit:
                # split happened
                self.treeDict[record[2]] = Tree(record[2], record[0], record[1], criterion.node_value(), criterion.impurity(), False, \
                    splitStatus[2], splitStatus[3], self.nodeId + 1, self.nodeId + 2)                

                self.stack.append((splitStatus[4], record[1], self.nodeId + 2, record[3] + 1))
                self.stack.append((record[0], splitStatus[4], self.nodeId + 1, record[3] + 1))

                self.nodeId += 2
            else:
                # current node is leaf
                self.treeDict[record[2]] = Tree(record[2], record[0], record[1], criterion.node_value(), criterion.impurity(), True, \
                    None, None, None, None)
               
        return self.treeDict


    def predict_proba(self, dataX):
        if self.type != Constants.Classifier:
            print('Invalid model type, predict_proba is only allowed for classifiers')
            raise ValueError
        nSamples = dataX.shape[0]
        predictedY = numpy.zeros((nSamples, self.nClasses))


        for i in range(nSamples):
            node = self.treeDict[0]
            while not node.isLeaf:
                if dataX[i][node.splitFeature] <= node.splitValue:
                    node = self.treeDict[node.leftId]
                else:
                    node = self.treeDict[node.rightId]

            predictedY[i] = node.nodeValue
        
        return predictedY

    def predict_value(self, dataX):
        if self.type != Constants.Regression:
            print('Invalid model type, predict_value is only allowed for Regression')
            raise ValueError
        nSamples = dataX.shape[0]
        predictedY = numpy.zeros((nSamples))


        for i in range(nSamples):
            node = self.treeDict[0]
            while not node.isLeaf:
                if dataX[i][node.splitFeature] <= node.splitValue:
                    node = self.treeDict[node.leftId]
                else:
                    node = self.treeDict[node.rightId]

            predictedY[i] = node.nodeValue
        
        return predictedY

if __name__ == '__main__':
    data = datasets.load_iris()
    dataX = data.data
    dataY = data.target
    '''
    for i in range(dataY.shape[0]):
        if dataY[i] != 0:
            dataY[i] = 1
    '''

    
    #c = Classifier(dataX, dataY, 3, None, 1, 2, 1, 5)
    #c.fit()

    dataX, dataY = DataSets.load_regression()
    d = datasets.load_boston()
    dataX = d.data
    dataY = d.target


    c = DecisionTreeBuilder(Constants.Regression, dataX, dataY, None, None, 1, 2, 1, None)
    c.fit()
    print(c.predict_value(dataX))
    print()

    print(metrics.mean_squared_error(dataY, c.predict_value(dataX)))


    






    


