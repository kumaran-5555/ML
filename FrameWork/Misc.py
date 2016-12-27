from collections import defaultdict
'''
timeToVal = defaultdict(list)

final = {}

for line in open(r'E:\Git\ML\Kaggle_Bosch\Data\all_min_id.csv','r'):
    if 'index' in line:
        continue

    fields = line[:-1].split(',')

    timeToVal[fields[1]].append(fields)


 
    gets a segment of rows with consecutive ids
    update the global dict

def lag(rows):
    if len(rows) == 1:
        final[rows[0][2]] = ',,'
        return
           

    for i in range(0,len(rows)):        
        final[rows[i][2]] ='{},{},{}'.format( len(rows),i,len(rows)-1)
        

def testMeanResponse(rows):
    trainRows = []
    testRows = []
    for r in rows:
        if r[3] == '0':
            trainRows.append(r)
        else:
            testRows.append(r)

    if len(trainRows) == 0:
        # store default response
        for r in testRows:
            final[r[2]]=''

    else:
        meanResp = sum([float(r[-1]) for r in trainRows])/ len(trainRows)

        for r in testRows:
            final[r[2]] = str(meanResp)

def trainFold(rows, fold):
    trainRows = []
    testRows = []
    for r in rows:
        if r[3] == '1':
            continue
        if int(r[2]) % 5 == fold:
            testRows.append(r)
        else:
            trainRows.append(r)
        
    if len(trainRows) == 0:
        # store default response        
        for r in testRows:
            final[r[2]]=''
    else:
        meanResp = sum([float(r[-1]) for r in trainRows])/ len(trainRows)
        for r in testRows:
            final[r[2]] = str(meanResp)



for k in timeToVal.keys():
    prevId = None
    segment = []
    for r in sorted(timeToVal[k], key=lambda x: int(x[2])):
        if prevId == None:
            prevId = int(r[2])
            segment.append(r)
            continue

        # segment continues    
        if int(r[2]) == prevId+1:
            prevId = int(r[2])
            segment.append(r)
        # segment breaks
        else:
            lag(segment)
            #testMeanResponse(segment)                    
            #trainFold(segment, 0) 
            #trainFold(segment, 1)
            #trainFold(segment, 2)
            #trainFold(segment, 3)
            #trainFold(segment, 4)

            
            segment = []
            segment.append(r)
            prevId = int(r[2])
    
    # left over    
    lag(segment)
    #testMeanResponse(segment)                    
    #trainFold(segment, 0) 
    #trainFold(segment, 1)
    #trainFold(segment, 2)
    #trainFold(segment, 3)
    #trainFold(segment, 4)
                
    
print(len(final))                    
with open(r'E:\Git\ML\Kaggle_Bosch\Data\magic2.csv', 'w') as out:                
    out.write('Id,count,rank,rankReverse\n')
    for k,v in final.items():
        out.write('{},{}\n'.format(k, v))

'''
import pandas as pd
import pickle
import numpy as np


train = pickle.load(open(r'E:/Git/ML/Kaggle_Bosch/Data/train_minmax.2.pkl', 'rb'))
test = pickle.load(open(r'E:/Git/ML/Kaggle_Bosch/Data/test_minmax.2.pkl', 'rb'))

output = pd.DataFrame()

temp = pd.concat((train, test))

output['Id'] = temp['Id']

x = pd.DataFrame(temp[['S30_Max', 'Id']].groupby(by='S30_Max').size().rename('COunt').reset_index())
x['Window'] = 0
x = x.sort('S30_Max')

windows = []
windowSize = 0.1
count = 0
c =  'S30_Max'
for i in range(x.shape[0]):
    windows.append((x.iloc[i][c], x.iloc[i]['COunt']))
    count += x.iloc[i]['COunt']

    # if windows not change    
    while x.iloc[i][c] - windows[0][0] > windowSize:
        count-=windows[0][0]
        del windows[0]
            

    x['Window'].iloc[i] = count
    
