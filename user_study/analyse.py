import zipfile
import pandas as pd
import numpy as np

# Path to the zipped Excel file
zip_file_path = './Illustrative Language Modelling_ A Case Study .csv.zip'
csv_file_name='Illustrative Language Modelling: A Case Study .csv'
# Open the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Read the CSV file from the zip archive
    with zip_ref.open(csv_file_name) as csv_file:
        # Convert the bytes data to a pandas DataFrame
        csv_data = pd.read_csv(csv_file)

results=csv_data.values[:,2:]

A_count=0
B_count=0
C_count=0

for index in range(results.shape[1]):

    if index not in [0, 31,62]:
        A_count += np.count_nonzero(results[:,index]=='A')
        B_count += np.count_nonzero(results[:, index] == 'B')
        C_count += np.count_nonzero(results[:, index] == 'C')


sum_all=A_count + B_count + C_count
print('Haivment rate ', round(A_count/sum_all,4))
print('Chatgpt rate ', round(B_count/sum_all, 4))
print('ViPE rate', round(C_count/sum_all, 4))
