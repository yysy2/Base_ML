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

def main(args):
    # create the outputs folder
    os.makedirs('outputs', exist_ok=True)

    # Log arguments
    run.log('Kernel type', np.str(args.kernel))
    run.log('Penalty', np.float(args.penalty))

    # Load iris dataset
    X, y = datasets.load_iris(return_X_y=True)

    # dividing X,y into train and test data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    data = {'train': {'X': x_train, 'y': y_train},
            'test': {'X': x_test, 'y': y_test}}

    # train a SVM classifier
    svm_model = SVC(kernel=args.kernel, C=args.penalty, gamma='scale').fit(data['train']['X'], data['train']['y'])
    svm_predictions = svm_model.predict(data['test']['X'])

    # accuracy for X_test
    accuracy = svm_model.score(data['test']['X'], data['test']['y'])
    print('Accuracy of SVM classifier on test set: {:.2f}'.format(accuracy))
    run.log('Accuracy', np.float(accuracy))

    # precision for X_test
    precision = precision_score(svm_predictions, data["test"]["y"], average='weighted')
    print('Precision of SVM classifier on test set: {:.2f}'.format(precision))
    run.log('precision', precision)

    # recall for X_test
    recall = recall_score(svm_predictions, data["test"]["y"], average='weighted')
    print('Recall of SVM classifier on test set: {:.2f}'.format(recall))
    run.log('recall', recall)

    # f1-score for X_test
    f1 = f1_score(svm_predictions, data["test"]["y"], average='weighted')
    print('F1-Score of SVM classifier on test set: {:.2f}'.format(f1))
    run.log('f1-score', f1)

    # files saved in the "outputs" folder are automatically uploaded into run history
    model_file_name = "model.pkl"
    joblib.dump(svm_model, os.path.join('outputs', model_file_name))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', type=str, default='rbf', help='Kernel type to be used in the algorithm')
    parser.add_argument('--penalty', type=float, default=1.0, help='Penalty parameter of the error term')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    ws = run.experiment.workspace
    titanic_df_dataset = Dataset.get_by_name(ws, name='Titanic dataset new train')
    data = dataset.to_pandas_dataframe()
    run.log('len of data', len(data))
    main(args=args)
