
import os.path
import numpy


def load_regression():
    dataX = []
    dataY = []
    with open(os.path.join(os.path.dirname(__file__), 'RegessionTestData.txt'), 'r') as file:
        count = 0 
        for l in file:
            count += 1
            fields = l.split('\t')
            dataX.append(fields[:3])
            dataY.append(fields[3])
            if count >= 2000:
                break

    dataX = numpy.array(dataX, dtype=float)
    dataY = numpy.array(dataY, dtype=float)

    return dataX, dataY





