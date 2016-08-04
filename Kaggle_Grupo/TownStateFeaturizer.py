import sys

from collections import defaultdict





class TownStateFeaturizer:
    def __init__(self, inputFile, outputFile):
        self.inputFile = open(inputFile, 'r', encoding='utf-8')
        self.outputFile = open(outputFile, 'w', encoding='utf-8')

        self.town = defaultdict(lambda : 0)
        self.state = defaultdict(lambda : 0)



    def process(self):
        for line in self.inputFile:
            fields = line[:-1].split(',')

            id = fields[0]
            town = fields[1]
            state = fields[2]

            if town not in self.town:
                self.town[town] = len(self.town) + 1


            if state not in self.state:
                self.state[state] = len(self.state) + 1




        self.inputFile.seek(0,0)

        stateVec = ['0'] * (len(self.state) + 1)
        townVec = ['0'] * (len(self.town) + 1)


        for line in self.inputFile:
            fields = line[:-1].split(',')

            id = fields[0]
            town = fields[1]
            state = fields[2]


            stateVec[self.state[state]]= '1'

            townVec[self.town[town]] = '1'

            self.outputFile.write("{}\t{},{}\n".format(id, ','.join(stateVec), ','.join(townVec)))

            stateVec[self.state[state]]= '0'

            townVec[self.town[town]] = '0'




            