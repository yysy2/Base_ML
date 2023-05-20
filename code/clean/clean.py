# Import libraries
import os
import argparse
import pandas as pd
from azureml.core import Run
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import pickle

import os
import argparse
import itertools
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from azureml.core import Dataset, Run
run = Run.get_context()

def titanic_clean(data):
    raw_row_count = (len(data))

    DropColoums = ['Name','Ticket','Cabin']
    data.drop(columns=DropColoums,inplace=True)

    le = preprocessing.LabelEncoder()
    le.fit(data['Sex'])
    data['Sex'] = le.transform(data['Sex'])
    le2 = preprocessing.LabelEncoder()
    le2.fit(data['Embarked'])
    data['Embarked'] = le2.transform(data['Embarked'])

    if data['Age'].isnull().any() == True:
        data['Age'] = data['Age'].fillna(data['Age'].median())

    if data['Fare'].isnull().any() == True:
        data['Fare'] = data['Fare'].fillna(data['Fare'].median())

    # remove nulls
    data = data.dropna()

    # Log processed rows
    row_count = (len(data))
    
    return(raw_row_count,row_count,data,le,le2)

def main(args):
    # create the outputs folder
    os.makedirs('outputs', exist_ok=True)

    # Get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepped-data', type=str, dest='prepped_data')
    args = parser.parse_args()
    save_folder = args.prepped_data

    # Get the experiment run context
    run = Run.get_context()
    
    ws = run.experiment.workspace
    titanic_df = Dataset.get_by_name(ws, name='Titanic dataset new train').to_pandas_dataframe()
    run.log('len of data', len(titanic_df))

    raw_row_count,row_count,titanic_df_cleaned,le,le2 = titanic_clean(titanic_df)

    os.makedirs(save_folder, exist_ok=True)
    model_path = os.path.join(save_folder, 'le.pkl')
    fileObj = open(model_path, 'wb')
    pickle.dump(le,fileObj)
    fileObj.close()

    model_path = os.path.join(save_folder, 'le2.pkl')
    fileObj = open(model_path, 'wb')
    pickle.dump(le2,fileObj)
    fileObj.close()

    run.log('raw_rows', raw_row_count)
    run.log('processed_rows', row_count)

    # Save the prepped data
    print("Saving Data")
    save_path = os.path.join(save_folder,'data_1303.csv')
    titanic_df_cleaned.to_csv(save_path, index=False, header=True)

    # End the run
    #run.complete()
    #1

if __name__ == '__main__':
    main()
