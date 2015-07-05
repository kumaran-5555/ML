#!/usr/bin/python3
import sys
from collections import defaultdict

if len(sys.argv) != 2:
    print('Usage: {0} <inputFile>'.format(sys.argv[0]))
    sys.exit(1)



inputFile = open(sys.argv[1], 'r', encoding='utf-8')

uidDict = dict()

for l in inputFile:
    fields = l[:-1].split('\t')
    if not fields[1].startswith('PageViewUrl'):
        continue
    
    try:
        hostName = fields[1].split('/')[2]
    except IndexError:
        continue
    if fields[0] not in uidDict:
        uidDict[fields[0]] = defaultdict(int)
    uidDict[fields[0]][hostName] += 1



for uid in uidDict:
    for host in uidDict[uid]:
        print('{0}\t{1}\t{2}'.format(uid, host, uidDict[uid][host]))
