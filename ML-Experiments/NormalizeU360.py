from collections import defaultdict
import hashlib

u360File = open(r'D:\Hack\OneMLDisplayAdsHackathon\u360_demodata.tsv', 'r')
outputFile = open(r'D:\Hack\OneMLDisplayAdsHackathon\auto_researcher\intermediate\u360_demodata.norm.tsv', 'w')



for line in u360File:
    fields = line[:-1].split('\t')
    '''
    0-UserId
    1-RegisteredCountry
    2-RegisteredBirthday
    3-RegisteredGender
    4-PredictedGender
    5-PredictedGenderProbability
    6-RegisteredAgeGroup
    7-PredictedAgeGroup
    8-PredictedAgeGroupProbability
    '''
    # country hashing
    countryCode = int(hashlib.sha256(fields[1].encode('utf-8')).hexdigest(), 16) % 200
       

    uid = fields[0]
    if float(fields[5]) > 0.6:
        gender = fields[4]
    else:
        gender = fields[3]

    if float(fields[8]) > 0.6:
        ageGroup = fields[7]
    else:
        ageGroup = fields[6]

    if gender == 'male':
        genderCode = 1
    elif gender == 'female':
        genderCode = 2
    else:
        genderCode = 0


    if ageGroup == '<18':
        ageCode = 1
    elif ageGroup == '[18,25)':
        ageCode = 2
    elif ageGroup == '[25,35)':
        ageCode = 3       
    elif ageGroup == '[35,50)':
        ageCode = 4
    elif ageGroup == '>=50':
        ageCode = 5
    else:
        ageCode = 0

    outputFile.write('{0}\t{1}\t{2}\n'.format(uid, 'CountryCode' + str(countryCode), '1'))
    outputFile.write('{0}\t{1}\t{2}\n'.format(uid, 'AgeCode' + str(ageCode), '1'))
    outputFile.write('{0}\t{1}\t{2}\n'.format(uid, 'GenderCode' + str(genderCode), '1'))

outputFile.close()






    