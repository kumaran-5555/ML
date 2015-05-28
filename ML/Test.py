import sklearn
import sklearn.datasets
import sklearn.tree

d = sklearn.datasets.load_iris()

t = sklearn.tree.DecisionTreeClassifier()
model = t.fit(d.data, d.target)
sklearn.tree.export_graphviz(model, out_file="E:\scikit-learn-master\sklearn\datasets\data\iris.csv.model.ps")

