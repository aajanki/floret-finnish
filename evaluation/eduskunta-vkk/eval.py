import fasttext
import floret
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

data_dir = Path('evaluation/data/eduskunta-vkk')


def main():
    model_name = sys.argv[1]
    if model_name.endswith('cc.fi.300.bin'):
        model = fasttext.load_model(model_name)
    else:
        model = floret.load_model(model_name)
    train_X, train_labels, dev_X, dev_labels, test_X, test_labels = load_data(model)

    print(f'train shape: {train_X.shape}')
    print(f'dev shape: {dev_X.shape}')
    print(f'test shape: {test_X.shape}')
    
    pipe = get_pipeline(C=0.1)
    pipe.fit(train_X, train_labels)
    
    print('## train ##')
    train_pred = pipe.predict(train_X)
    print(classification_report(train_labels, train_pred))

    print('## dev ##')
    dev_pred = pipe.predict(dev_X)
    print(classification_report(dev_labels, dev_pred))

    print('## test ##')
    test_pred = pipe.predict(test_X)
    print(classification_report(test_labels, test_pred))

    
def load_vectors(filename, model):
    data = []
    with open(filename) as f:
        for text in f:
            data.append(model.get_sentence_vector(text.strip()))

    return np.asarray(data)


def load_labels(filename):
    return pd.read_csv(filename, header=0)['ministry']


def load_data(model):
    train_labels = load_labels(data_dir / 'train.csv.bz2')
    dev_labels = load_labels(data_dir / 'dev.csv.bz2')
    test_labels = load_labels(data_dir / 'test.csv.bz2')

    train_vectors = load_vectors(data_dir / 'train.txt', model)
    dev_vectors = load_vectors(data_dir / 'dev.txt', model)
    test_vectors = load_vectors(data_dir / 'test.txt', model)
    
    return train_vectors, train_labels, dev_vectors, dev_labels, test_vectors, test_labels


def get_pipeline(C=0.1):
    steps = []
    steps.append(('scaler', MaxAbsScaler()))
    steps.append(('classifier', LinearSVC(loss='hinge', C=C, intercept_scaling=5.0,
                                          max_iter=100000, multi_class='ovr')))
    return Pipeline(steps)


if __name__ == '__main__':
    main()
