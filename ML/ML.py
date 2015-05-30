
import numpy
import random



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


class SplitRecord:
    def __init__(self):
        self.start = None
        self.end = None
        self.value = None
        self.impurity = None
        self.left_impurity = None
        self.right_impurity = None
        self.improvement = None
        self.feature_idx = None

class Splitter:
    def __init__(self, start, end, n_classes, label_y, data_x, sample_weight_y, samples, total_training_weight, seed,\
        min_samples_leaf, min_weight_leaf):
        '''
        initialize for each node that has to be split, use split method to figure our best split for this node
        returns SplitRecord
        '''
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




    def split(self):
        '''
        tries each features for split and return best one with split value
        assumes node is splitable
        '''

        f_i = [i for i in range(self.n_features)]

        split_rec = SplitRecord()
        split_rec.start = self.start
        split_rec.end = self.end
        
        '''
        going to use all features, so no need to randomize the pick
        # randomly pick a feature and check best split point
        # choose the best among all features
        random.seed(self.seed)
        random.shuffle(f_i)
        '''
        best_improvement = float('-inf')
        best_v = None
        best_f = None
        for f in f_i:
            x_i = self.data_x[self.start:self.end, f]
            sorted_samples = numpy.argsort(x_i) + self.start
            gini = GiniImpurity(0, self.n_samples, self.n_classes, self.label_y, self.sample_weight_y, sorted_samples, self.total_training_weight)
            
            for i in range(1, self.n_samples):
                gini.update(i)
                if i < self.min_samples_leaf or self.n_samples < self.min_samples_leaf or \
                    gini.total_left_weight < self.min_weight_leaf or gini.total_right_weight < self.min_weight_leaf:
                    # bad split, continue
                    continue


                if best_improvement < gini.improvement():
                    best_improvement = gini.improvement()
                    best_v = (x_i[sorted_samples[i-1]] + x_i[sorted_samples[i]]) / 2
                    best_f = f

                    split_rec.feature_idx = f
                    split_rec.impurity = gini.impurity()
                    split_rec.left_impurity,split_rec.right_impurity = gini.child_impurity()
                    split_rec.value = best_v
                    
        s = self.start
        e = self.end
        while s < e:             
            if self.data_x[s, best_f] <= best_v:
                # on the right side
                s += 1
                continue
            else:
                e -= 1
                # have to move to right side
                tmp = self.samples[e]
                self.samples[e] = self.samples[s]
                self.samples[s] = tmp

        

        return split_rec
        #return best_improvement,best_v,best_f




f = open("E:\scikit-learn-master\sklearn\datasets\data\iris.csv",'r', encoding='utf-8')

dim_x,dim_y = map(int,str(f.readline()).split(',')[:2])
sample_weight_y =  numpy.ones(dim_x)


data = numpy.empty((dim_x,dim_y))
label = numpy.empty(dim_x)
samples = numpy.empty(dim_x)
#next(f)

for i,d in enumerate(f):
    data[i] = numpy.asarray(d.split(',')[:-1], dtype=numpy.float)
    label[i] = numpy.asarray(d.split(',')[-1], dtype=numpy.int)
    samples[i] = i

s = Splitter(0, dim_x, 3, label, data, sample_weight_y, samples, dim_x, 1, 1, 1)
print(s.split())










    




