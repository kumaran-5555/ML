import sklearn.feature_extraction


features = open(r'D:\Hack\OneMLDisplayAdsHackathon\auto_researcher\train_category.dat', 'r')
hasher = sklearn.feature_extraction.FeatureHasher(input_type='dict')


def generateRows():
    tempDict = {}
    prevKey = ''
    features.readline()

    for l in features:
        fields = l[:-1].split('\t')
        if prevKey != fields[0] and prevKey != '':
            yield tempDict
            tempDict = {}
            tempDict[fields[1]] = float(fields[2])
            prevKey = fields[0]
        else:
            if prevKey == '':
                prevKey = fields[0]

            tempDict[fields[1]] = float(fields[2])


#for i in generateRows():
#    print(i)

final = hasher.transform(generateRows())
pass






