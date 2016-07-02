from sklearn.feature_extraction import DictVectorizer

import Kaggle_Grupo
import numpy





class ProductFeaturize:
    def __init__(self, inputFile, outputFile):

        self.inputFile = open(inputFile, 'r', encoding='utf-8')
        self.outputFile = open(outputFile, 'wb')
        self.rows = []
        
         


    def process(self):

        self.inputFile.readline()

        for line in self.inputFile:
            


            try:
                id = line[:-1].split(',')[0]

                name = Kaggle_Grupo.Utils.StringNormalize(line[:-1].split(',')[1])

                d = {}            

                for t in name.split(' '):
                    d['p_'+t] = 1

                d['id'] = int(id)

                d['p_weight'] = Kaggle_Grupo.Utils.ExtractProductWeight(name)

                d['p_piece'] = Kaggle_Grupo.Utils.ExtractProductPiece(name)

            
                self.rows.append(d)
            except TypeError:
                continue


        vectorizer = DictVectorizer(sparse=False)

        vec = vectorizer.fit_transform(self.rows)



        headerFields = [ i[0] for i in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1])]



        numpy.savetxt(self.outputFile, vec, fmt="%d", delimiter='\t', newline='\n', header='\t'.join(headerFields))





        


        


        