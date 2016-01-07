import sys

def RemoveColumns(dataObj, listOfCols):
    '''
        removes the given column in the dataObject and updates corresponding header
    '''

    newTrainX = []
    newTestX = []


    indicesToBeRemoved = set()

    for c in listOfCols:
        indicesToBeRemoved.add(dataObj.header[c])

    # start removing from training data

    for row in dataObj.trainX:
        newRow = []
        for i in range(len(row)):
            if i in indicesToBeRemoved:
                continue
            newRow.append(row[i])

        newTrainX.append(newRow)


    # start removing from test data

    for row in dataObj.testX:
        newRow = []
        for i in range(len(row)):
            if i in indicesToBeRemoved:
                continue
            newRow.append(row[i])

        newTestX.append(newRow)



    # update header
    h = sorted(dataObj.header, key=dataObj.header.get)
    for i in indicesToBeRemoved:
        h.pop(i)

    newHeader = {}
    for i in range(len(h)):
        newHeader[h[i]] = i


    # update object
    dataObj.trainX = newTrainX
    dataObj.testX = newTestX
    dataObj.header = newHeader

    return



def _NumberizeCategoryFeatures(dataObj, columnIndex):
    count = 0
    values = {}
    for row in dataObj.trainX:
        if row[columnIndex] in values:
            row[columnIndex] = values[row[columnIndex]]
            continue

        values[row[columnIndex]] = str(count)
        row[columnIndex] = values[row[columnIndex]]
        count += 1

    for row in dataObj.testX:
        if row[columnIndex] in values:
            row[columnIndex] = values[row[columnIndex]]
            continue

        # we want all test values to be seen in train values
        raise ValueError



        







def NumberizeCategoryFeatures(dataObj, listOfCols):
    for c in listOfCols:
        _NumberizeCategoryFeatures(dataObj, dataObj.header[c])




def ReplaceMissingValuesWithNone(dataObj):
    for row in dataObj.trainX:
        for i in range(len(row)):
            if row[i] == '':
                row[i] = None

    for row in dataObj.testX:
        for i in range(len(row)):
            if row[i] == '':
                row[i] = None




