import numpy as np
import pandas as pd
import pickle
import os
import glob

def preprocess_yahoo_dataset(path, output_prefix):
    files_a1 = glob.glob(os.path.join(path, 'A1Benchmark/real_*.csv'))
    files_a1.sort()
    files_a2 = glob.glob(os.path.join(path, 'A2Benchmark/synthetic_*.csv'))
    files_a2.sort()
    files_a3 = glob.glob(os.path.join(path, 'A3Benchmark/A3Benchmark-TS*.csv'))
    files_a3.sort()
    files_a4 = glob.glob(os.path.join(path, 'A4Benchmark/A4Benchmark-TS*.csv'))
    files_a4.sort()

    all_files = files_a1 + files_a2 + files_a3 + files_a4

    series_data = []
    series_labels = []

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for file in all_files:
        df = pd.read_csv(file)
        values = df['value'].tolist()
        if 'is_anomaly' in df.columns:
            labels = df['is_anomaly'].tolist()
        elif 'anomaly' in df.columns:
            labels = df['anomaly'].tolist()
        else:
            raise ValueError("No label column found in file: " + file)

        # Split each series into training and testing sets
        split_index = len(values) // 2
        x_train.append(values[:split_index])
        y_train.append(labels[:split_index])
        x_test.append(values[split_index:])
        y_test.append(labels[split_index:])


    '''
    x_train = [np.array(series, dtype=float) for series in x_train]
    y_train = [np.array(series, dtype=float) for series in y_train]
    x_test = [np.array(series, dtype=float) for series in x_test]
    y_test = [np.array(series, dtype=float) for series in y_test]
    '''

    # Find the maximum length of series in the training and testing sets
    max_length_train = max(len(series) for series in x_train)
    max_length_test = max(len(series) for series in x_test)
    max_length = max(max_length_train, max_length_test)

    # Zero-padding each series in the training set to the maximum length
    padded_x_train = [np.pad(series, (0, max_length - len(series)), 'constant') for series in x_train]
    padded_y_train = [np.pad(series, (0, max_length - len(series)), 'constant') for series in y_train]

    # Zero-padding each series in the testing set to the maximum length
    padded_x_test = [np.pad(series, (0, max_length - len(series)), 'constant') for series in x_test]
    padded_y_test = [np.pad(series, (0, max_length - len(series)), 'constant') for series in y_test]

    # Save data into separate files
    with open(output_prefix + '_x_train.pkl', 'wb') as f:
        pickle.dump(padded_x_train, f)
    with open(output_prefix + '_y_train.pkl', 'wb') as f:
        pickle.dump(padded_y_train, f)
    with open(output_prefix + '_x_test.pkl', 'wb') as f:
        pickle.dump(padded_x_test, f)
    with open(output_prefix + '_y_test.pkl', 'wb') as f:
        pickle.dump(padded_y_test, f)

if __name__ == '__main__':
    path = "/Users/aleg2/Downloads/ydata-labeled-time-series-anomalies-v1_0"  # Update this path to the location of your Yahoo dataset
    output_prefix = 'yahoo_as_ts2vec'  # The prefix for the output file names
    preprocess_yahoo_dataset(path, output_prefix)
