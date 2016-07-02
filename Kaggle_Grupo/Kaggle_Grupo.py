import re
import ProductFeaturizer


class Utils:
    def StringNormalize(string):
        out = string.lower()
        out = re.sub('[^A-Za-z0-9 ]+', '', out)
        out = re.sub('[ ]+', ' ', out)

        return out



    def ExtractProductWeight(string):
        grams = re.compile(r'.*\b([0-9]+)g\b.*').match(string)

        
        kilos = re.compile(r'.*\b([0-9]+)kg\b.*').match(string)

        if kilos and len(kilos.groups()) > 0 :
            return int(kilos.groups()[0])*1000

        if grams and len(grams.groups()) > 0:
            return int(grams.groups()[0])

        return 0

    def ExtractProductPiece(string):
        piece = re.compile(r'.*\b([0-9]+)p\b.*').match(string)

        if piece and len(piece.groups()) > 0:
            return int(piece.groups()[0])

        return 0


if __name__ == '__main__':
    c = ProductFeaturizer.ProductFeaturize(r"E:\Git\ML\Kaggle_Grupo\Data\producto_tabla.csv", r"E:\Git\ML\Kaggle_Grupo\Data\producto_tabla.tsv")
    c.process()



    