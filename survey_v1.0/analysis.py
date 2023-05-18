import pandas as pd

responses= pd.read_csv('Illustrative Language Modelling.csv')
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