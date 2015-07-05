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



        






'''



        
    
class Splitter:
    def __init__(self, start, end, n_classes, label_y, data_x, sample_weight_y, samples, total_training_weight, criterion, seed,\
        min_samples_leaf, min_weight_leaf, split_rec):
        '
        initialize for each node that has to be split, use split method to figure our best split for this node
        returns SplitRecord
        '
        self.start = start
        self.end = end
        self.n_classes = n_classes
        self.label_y = label_y
        self.data_x = data_x
        self.sample_weight_y = sample_weight_y
        self.samples = samples
        self.total_training_weight = total_training_weight
        self.n_features = data_x.shape[1]
        self.seed = seed
        self.n_samples = self.end-self.start
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.split_rec = split_rec
        self.criterion = criterion
        self.threshold = 1e-7
        self.gini = self.criterion(0, self.n_samples, self.n_classes, self.label_y, self.sample_weight_y, self.samples[self.start:self.end], self.total_training_weight)

    def split(self):
        ''
        tries each features for split and return best one with split value
        assumes node is splitable
        ''

        f_i = [i for i in range(self.n_features)]

        #split_rec = SplitRecord()
        self.split_rec.start = self.start
        self.split_rec.end = self.end
        
        ''
        going to use all features, so no need to randomize the pick
        # randomly pick a feature and check best split point
        # choose the best among all features
        random.seed(self.seed)
        random.shuffle(f_i)
        ''
        best_improvement = float('-inf')
        best_v = None
        best_f = None
        best_i = None
        x_i = numpy.empty(self.end-self.start)
        sample_i = numpy.empty(self.end-self.start)
        for f in f_i:
            #for i in range(self.start,self.end):
            #    x_i[i-self.start] = self.data_x[self.samples[i], f]
            #    sample_i[i-self.start] = self.samples[i]

            sample_i = self.samples[self.start:self.end]
            x_i = self.data_x.take(sample_i, axis=0)[:,f]
            
            sorted_samples = [sample_i[i] for i in numpy.argsort(x_i)]
            x_i = numpy.sort(x_i)

            gini = self.criterion(0, self.n_samples, self.n_classes, self.label_y, self.sample_weight_y, sorted_samples, self.total_training_weight)
            # is node clean enough
            if gini.impurity() < 0.0000007:
                self.split_rec.impurity = gini.impurity()
                self.split_rec.node_val = gini.node_value()
                return self.split_rec
            
            for i in range(1, self.n_samples):
                
                # no much movement in feature value
                if x_i[i] - x_i[i-1] <= self.threshold:
                    continue

                gini.update(i)
                if i < self.min_samples_leaf or self.n_samples < self.min_samples_leaf or \
                    gini.total_left_weight < self.min_weight_leaf or gini.total_right_weight < self.min_weight_leaf:
                    # bad split, continue
                    continue


                if best_improvement < gini.improvement():
                    best_improvement = gini.improvement()
                    best_i = i
                    best_v = (x_i[i-1] + x_i[i]) / 2
                    best_f = f
                    self.split_rec.improvement = best_improvement
                    self.split_rec.feature_idx = f
                    self.split_rec.impurity = gini.impurity()
                    self.split_rec.left_impurity,self.split_rec.right_impurity = gini.child_impurity()
                    self.split_rec.value = best_v
                    self.split_rec.position = self.start + i
                    self.split_rec.node_weight = gini.total_weight
                    self.split_rec.node_weight_left = gini.total_left_weight
                    self.split_rec.node_weight_right = gini.total_right_weight
                    self.split_rec.node_val = gini.node_value()


                    
        s = self.start
        e = self.end
        while s < e:             
            if self.data_x[self.samples[s], best_f] <= best_v:
                # on the right side
                s += 1
                continue
            else:
                e -= 1
                # have to move to right side
                tmp = self.samples[e]
                self.samples[e] = self.samples[s]
                self.samples[s] = tmp

        return self.split_rec
        #return best_improvement,best_v,best_f

def TreeBuilder():

    
    #f = open("C:\Program Files (x86)\WinPython-64bit-3.4.3.3\python-3.4.3.amd64\Lib\site-packages\sklearn\datasets\data\\iris2.csv",'r', encoding='utf-8')
    f = open("E:\scikit-learn-master\sklearn\datasets\data\iris.csv", 'r', encoding='utf-8')


    dim_x,dim_y = map(int,str(f.readline()).split(',')[:2])
    sample_weight_y =  numpy.ones(dim_x)


    data = numpy.empty((dim_x,dim_y))
    label = numpy.empty(dim_x) 
    samples = numpy.empty(dim_x, dtype=int)
    min_samples_split =  2
    min_samples_leaf = 1
    min_weight_leaf = 1
    n_classes = 3

    #next(f)
    # node id to split record
    tree = {}

    for i,d in enumerate(f):
        data[i] = numpy.asarray(d.split(',')[:-1], dtype=numpy.float)
        label[i] = numpy.asarray(d.split(',')[-1], dtype=numpy.int)
        samples[i] = i
    
    id = 0
    root_rec = SplitRecord(0, dim_x, id, sum(sample_weight_y))
    stack = [root_rec]

    while len(stack):
        record = stack.pop()
        # check if this node is split-able
        is_leaf = True
        splitter = Splitter(record.start, record.end,  n_classes, label, data, sample_weight_y,\
              samples, sum(sample_weight_y), GiniImpurity, 1, min_samples_leaf, min_weight_leaf, record)
        if record.node_count >= (min_samples_leaf * 2) or \
            record.node_count >= min_samples_split:
            # can split
            is_leaf = False
        else:
            is_leaf = True
            record.node_val = splitter.gini.node_value()

        if not is_leaf:

            splitter.split()
            is_leaf = not record.position

        if not is_leaf:
            
            left = SplitRecord(record.start, record.position, id+1, record.node_weight_left)
            right = SplitRecord(record.position, record.end, id+2, record.node_weight_right)
            record.left_id = id + 1
            record.right_id = id + 2
            tree[record.node_id] = record
            
            stack.append(right)
            stack.append(left)

            samples = numpy.copy(splitter.samples)
            id += 2

        else:
            tree[record.node_id] = record
            continue
    print(tree)
    tree_out_file = open("E:\scikit-learn-master\sklearn\datasets\data\iris.csv.tree", 'wb')
    pickle.dump(tree, tree_out_file)


    for i in range(id):
        print(tree[i])

'''

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


    






    


