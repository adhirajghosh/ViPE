import pandas as pd
from zipfile import ZipFile

with ZipFile('Illustrative Language Modelling.csv.zip') as zf:
    responses=zf.open('Illustrative Language Modelling.csv')

responses= pd.read_csv(responses)
columns=responses.columns[1:]

resutls={}
for question in columns:
    for subject_response in responses[question]:
        if subject_response in resutls:
            resutls[subject_response] +=1
        else:
            resutls[subject_response]=1

for k,v in resutls.items():
    print('{}:{} %'.format(k,round(v/sum(resutls.values()),4)))