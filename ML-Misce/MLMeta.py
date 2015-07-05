from sklearn import tree
from sklearn import ensemble
import numpy
from sklearn import metrics
from sklearn import linear_model



train_file = open('E:\\tools\\NeuralNetEval\BadLocal\\train_meta_rating.tsv', 'r',  encoding='utf-8')

test_file = open('E:\\tools\\NeuralNetEval\BadLocal\\measurement_meta_rating.tsv', 'r', encoding='utf-8')


num_features = 2
train_rows = 20159
test_rows = 16527

train_x = numpy.empty((train_rows, 2))
test_x = numpy.empty((test_rows, 2))
train_y = numpy.empty((train_rows))
test_y = numpy.empty((test_rows))


i = 0
for l in train_file:
    fields = list(map(float, l[:-1].split('\t')[:3]))
    train_x[i][0] = fields[0]
    train_x[i][1] = fields[1]
    train_y[i] = fields[2]
    i += 1
i = 0
for l in test_file:
    fields = list(map(float, l[:-1].split('\t')[:3]))
    test_x[i][0] = fields[0]
    test_x[i][1] = fields[1]
    test_y[i] = fields[2]
    i += 1


#e = ensemble.GradientBoostingRegressor(n_estimators =200, min_samples_split = 2, min_samples_leaf = 5, max_depth = 5 )
e = linear_model.LinearRegression()
e.fit(train_x, train_y)
#print(e.predict_proba(test_x))

p,r,t = metrics.precision_recall_curve(test_y, e.predict(test_x))

for i,j,k in zip(p,r,t):
    print(i,j,k)






