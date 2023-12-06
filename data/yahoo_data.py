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

    for file in all_files:
        df = pd.read_csv(file)
        values = df['value'].tolist()
        if 'is_anomaly' in df.columns:
            labels = df['is_anomaly'].tolist()
        elif 'anomaly' in df.columns:
            labels = df['anomaly'].tolist()
        else:
            raise ValueError("No label column found in file: " + file)

        series_data.append(values)
        series_labels.append(labels)

    # Find the maximum length of series in the dataset
    max_length = max(len(series) for series in series_data)

    # Zero-padding each series to the maximum length
    padded_x = [np.pad(series, (0, max_length - len(series)), 'constant') for series in series_data]
    padded_y = [np.pad(labels, (0, max_length - len(labels)), 'constant') for labels in series_labels]


    # Split the list of series into training and testing sets
    split_index = len(padded_x) // 2
    x_train = padded_x[:split_index]
    y_train = padded_y[:split_index]
    x_test = padded_x[split_index:]
    y_test = padded_y[split_index:]

    # Save data into separate files
    with open(output_prefix + '_x_train.pkl', 'wb') as f:
        pickle.dump(x_train, f)
    with open(output_prefix + '_y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open(output_prefix + '_x_test.pkl', 'wb') as f:
        pickle.dump(x_test, f)
    with open(output_prefix + '_y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)

if __name__ == '__main__':
    path = "/Users/aleg2/Downloads/ydata-labeled-time-series-anomalies-v1_0"  # Update this path to the location of your Yahoo dataset
    output_prefix = 'yahoo'  # The prefix for the output file names
    preprocess_yahoo_dataset(path, output_prefix)
