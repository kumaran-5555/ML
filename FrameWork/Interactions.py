import scipy.stats
import pandas as pd
import numpy as np
import pickle

#train = pickle.load(open(r'E:\Git\ML\Kaggle_Bosch\Data\WithInter1\train.pkl', 'rb'))
train = pd.read_csv(r'E:\Git\ML\Kaggle_Bosch\Data\train_numeric.csv', nrows=500000)

train.rename(columns={'Response': 'label'}, inplace=True)

cols = [f for f in train.columns.values if f != 'label']

train[cols] += 2
train.fillna(0, inplace=True)
eps = 0.0
output = open(r'E:\Git\ML\Kaggle_Bosch\Data\InteractionsAll.tsv', 'w')
Pairs = [l[:-1].split('\t') for l in open(r'E:\Git\ML\Kaggle_Bosch\Data\Pairs.tsv', 'r')]

def getInteractions(pairs, output):
    
    output = open(output, 'w')
    count = 0
    for c1,c2 in pairs:
        count+=1
        if count %1000 == 0:
            print(count)

        # plus
        corr = abs(scipy.stats.pearsonr(train[c1] + train[c2], train['label'])[0])
        #print('{} {} {} plus\n'.format(c1, c2, corr))
        if corr > eps:
            output.write('{} {} {} plus\n'.format(c1, c2, corr))
            output.flush()

        # minus
        corr = abs(scipy.stats.pearsonr(train[c1] - train[c2], train['label'])[0])
        if corr > eps:
            output.write('{} {} {} minus\n'.format(c1, c2, corr))
            output.flush()

        # multiply
        corr = abs(scipy.stats.pearsonr(train[c1] * train[c2], train['label'])[0])
        if corr > eps:
            output.write('{} {} {} multiply\n'.format(c1, c2, corr))
            output.flush()

        # divide
        try:
            corr = abs(scipy.stats.pearsonr(train[c1] / train[c2], train['label'])[0])
            if corr > eps:
                output.write('{} {} {} divide\n'.format(c1, c2, corr))
                output.flush()
            corr = abs(scipy.stats.pearsonr(train[c2] / train[c1], train['label'])[0])
            if corr > eps:
                output.write('{} {} {} divide\n'.format(c2, c1, corr))
                output.flush()

        except:
            continue

        
    

    output.close()
if __name__ == '__main__':   
    from multiprocessing import Process



    split = int(len(Pairs)/5)
    process = []

    for i in range(5):
        p = Process(target=getInteractions, args=(Pairs[i*split:(i+1)*split], r'E:\Git\ML\Kaggle_Bosch\Data\InteractionAll.tsv'+str(i)))
        process.append(p)
        p.start()
    
    for p in process:
        p.join()

