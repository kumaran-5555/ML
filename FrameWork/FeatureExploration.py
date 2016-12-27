import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import metrics
import pickle


class FeatureEvaluator:
    def __init__(self, dataFrame, featureName, label='label'):
        self.df = dataFrame
        self.label = label
        self.feature = featureName

    def evaluate(self):
        '''
            computes goodness of the feature
            and call metric for the final score dict
        '''
        raise NotImplementedError

    def metric(self):
        '''
            uses evaluations done by evaluate and computes various metrics
            return dict of score['metricName'] = value
        '''
        raise NotImplementedError


class XGBEvaluator(FeatureEvaluator):

    def evaluate(self):
        self.model = xgb.XGBClassifier(max_depth=7, learning_rate=0.01, objective='binary:logistic', n_estimators=50)
        self.model.fit(pd.DataFrame(self.df[self.feature]).iloc[:500000,:].values, self.df[self.label][:500000].values, verbose=True,
                       eval_metric='auc', eval_set=[tuple((pd.DataFrame(self.df[self.feature]).iloc[500000:,:].values, self.df[self.label][500000:].values))],
                       early_stopping_rounds=10)

    def metric(self):
        scores = {}
        pred = self.model.predict_proba(pd.DataFrame(self.df[self.feature]).iloc[500000:,:].values)[:,1]
        scores['xgb_auc'] = metrics.roc_auc_score(self.df[self.label][500000:].values, pred)
        scores['xgb_logloss'] = metrics.log_loss(self.df[self.label][500000:].values, pred)        

        return scores


class GiniEvaluator(FeatureEvaluator):
    def evaluate(self):
        pass

    def metric(self):
        pass




class FeatureTesting:
    def __init__(self, dataFrame, outputFile, listOfEvals=None):
        self.df = dataFrame
        self.evals = listOfEvals
        self.outputFile = open(outputFile, 'w')


    def Test(self):
        cols = [f for f in self.df.columns.values if f != 'label']
        for c in cols:
            output = c
            for eval in self.evals:
                e = eval(self.df, c, label='label')
                e.evaluate()
                scores = e.metric()

                for k,v in scores.items():
                    output += '\t{}\t{}'.format(k, v)

            self.outputFile.write(output+'\n')
            self.outputFile.flush()

        self.outputFile.close()



if __name__ == '__main__':
    df = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\WithInter1\train.pkl', 'rb'))

    ft = FeatureTesting(df, r'E:\Git\ML\Kaggle_Bosch\Data\FeatTest2.tsv', [XGBEvaluator])
    ft.Test()








