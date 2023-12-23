import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import argparse
import pickle

def pad_sequences(sequences, max_len):
    return [np.pad(seq, (0, max_len - len(seq)), 'constant') for seq in sequences]


def _load_raw_KPI(train_filename, test_filename):
    train_data = pd.read_csv(train_filename)
    train_data = train_data.set_index(['KPI ID', 'timestamp']).sort_index()
    x_train = {}
    y_train = {}
    timestamp_train = {}
    scaler = {}
    for name, df in train_data.groupby(level=0):
        x_train[name] = df['value'].to_numpy()
        y_train[name] = df['label'].to_numpy()
        meanv = df['value'].mean()
        stdv = df['value'].std()
        scaler[name] = (meanv, stdv)
        x_train[name] = (x_train[name] - meanv) / stdv
        timestamp_train[name] = df.index.get_level_values(1)
    
    test_data = pd.read_hdf(test_filename)
    test_data['KPI ID'] = test_data['KPI ID'].apply(str)
    test_data = test_data.set_index(['KPI ID', 'timestamp']).sort_index()
    x_test = {}
    y_test = {}
    timestamp_test = {}
    for name, df in test_data.groupby(level=0):
        x_test[name] = df['value'].to_numpy()
        y_test[name] = df['label'].to_numpy()
        x_test[name] = (x_test[name] - scaler[name][0]) / scaler[name][1]
        timestamp_test[name] = df.index.get_level_values(1)
    return x_train, y_train, timestamp_train, x_test, y_test, timestamp_test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default='data/KPI_data/phase2_train.csv')
    parser.add_argument('--test-file', type=str, default='data/KPI_data/phase2_ground_truth.hdf')
    parser.add_argument('-o', '--output', type=str, default='kpi.pkl')
    args = parser.parse_args()

    data1, labels1, timestamps1, data2, labels2, timestamps2 = _load_raw_KPI(args.train_file, args.test_file)
    
    all_train_data = []
    all_train_labels = []
    all_test_data = []
    all_test_labels = []
    
    for k in data1:
        data = data1[k]
        labels = labels1[k]
        l = len(data) // 2

        all_train_data.append(data[:l])
        all_train_labels.append(labels[:l])  # Keep as numpy array
        all_test_data.append(data[l:])
        all_test_labels.append(labels[l:])  # Keep as numpy array

    for k in data2:
        data = data2[k]
        labels = labels2[k]
        l = len(data) // 2

        all_train_data.append(data[:l])
        all_train_labels.append(labels[:l])  # Keep as numpy array
        all_test_data.append(data[l:])
        all_test_labels.append(labels[l:])  # Keep as numpy array

    max_length = max(max(len(seq) for seq in all_train_data), 
                     max(len(seq) for seq in all_test_data))
    

    all_train_data = pad_sequences(all_train_data, max_length)
    all_train_labels = pad_sequences(all_train_labels, max_length)
    all_test_data = pad_sequences(all_test_data, max_length)
    all_test_labels = pad_sequences(all_test_labels, max_length)



    # Save the data into separate files
    with open('KPI_x_train.pkl', 'wb') as f:
        pickle.dump(all_train_data, f)
    with open('KPI_y_train.pkl', 'wb') as f:
        pickle.dump(all_train_labels, f)
    with open('KPI_x_test.pkl', 'wb') as f:
        pickle.dump(all_test_data, f)
    with open('KPI_y_test.pkl', 'wb') as f:
        pickle.dump(all_test_labels, f)