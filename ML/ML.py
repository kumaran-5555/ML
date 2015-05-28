
import numpy


class GiniImpurity:    
    def __init__(self, start, end, n_classes, label_y, sample_weight_y, samples):
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

        return 1.0 - left / (self.total_left_weight * self.total_left_weight), \
            1.0 - right / (self.total_right_weight * self.total_right_weight)
    



f = open("E:\scikit-learn-master\sklearn\datasets\data\iris.csv",'r', encoding='utf-8')

dim_x,dim_y = map(int,str(f.readline()).split(',')[:2])
sample_weight_y =  numpy.ones(dim_x)


data = numpy.empty((dim_x,dim_y))
label = numpy.empty(dim_x)
#next(f)

for i,d in enumerate(f):
    data[i] = numpy.asarray(d.split(',')[:-1], dtype=numpy.float)
    label[i] = numpy.asarray(d.split(',')[-1], dtype=numpy.int)

s = numpy.argsort(data[:,3])
print(data[:,3])
print(s)
#print(sorted(s))

gini = GiniImpurity(0,dim_x, 3, label, sample_weight_y, s)



print(gini.impurity())

for i in range(dim_x):
    gini.update(i)
    print(i, gini.child_impurity())














    




