import numpy as np
import pandas as pd
import pickle
import os
import glob

'''
The matrix profile data are computed with the matlab code officially provided by the authors of the paper.
This code contains the functions to preprocess matrix profile data referred to yahoo anomaly detection dataset.

Input: it takes as input the matrix profile data inside a csv, with a column named "score" and a column that contains the relative timestamp index.

The input should contain NaN/negative values for the training set used during the matrix profile computation.

The output is a pickle file containing the matrix profile data and the relative timestamp index. 
It should be organized specifically with the exact same structure of the pkl file containing training and test data.

So the output is a list of lists, where each list contains the matrix profile data for a specific time series.

Each output refers indipendentely to the training and test set, together with the corresponding timestamp index, which is inside a 
different pkl, as for yahoo data.

Because it is necessary to work with sequences of the same length, the matrix profile data are padded with zeros to the
maximum length of the time series. This should also help to avoid problems with the matrix profile data that are not computed. 

From previous analisys, good parameter for the MP are Subsequence length = 432 and CurrentIndex=4032.

'''


def preprocess_matrix_profile_yahoo(path, matrix_profile_path, output_prefix):
    files_a1 = glob.glob(os.path.join(path, 'A1Benchmark/real_*.csv'))
    files_a1.sort()
    files_a2 = glob.glob(os.path.join(path, 'A2Benchmark/synthetic_*.csv'))
    files_a2.sort()
    files_a3 = glob.glob(os.path.join(path, 'A3Benchmark/A3Benchmark-TS*.csv'))
    files_a3.sort()
    files_a4 = glob.glob(os.path.join(path, 'A4Benchmark/A4Benchmark-TS*.csv'))
    files_a4.sort()

    # Load the matrix profile data
    mp_df = pd.read_csv(matrix_profile_path, delimiter=';')
    matrix_profile = mp_df['score'].tolist()  

    all_files = files_a1 + files_a2 + files_a3 + files_a4

    series_data = []
    series_labels = []
    series_mp = []

    mp_index = 0

    for file in all_files:
        df = pd.read_csv(file)
        values = df['value'].tolist()
        if 'is_anomaly' in df.columns:
            labels = df['is_anomaly'].tolist()
        elif 'anomaly' in df.columns:
            labels = df['anomaly'].tolist()
        else:
            raise ValueError("No label column found in file: " + file)

        series_length = len(values)
        
        series_mp_segment = matrix_profile[mp_index:mp_index + series_length]
        mp_index += series_length

        series_data.append(values)
        series_labels.append(labels)
        series_mp.append(series_mp_segment)

    #Note, maybe it is possible/useful to look for the shortest one and use it as a window size to resize the code

    # Find the maximum length of series in the dataset
    max_length = max(len(series) for series in series_data)

    # Zero-padding each series to the maximum length
    padded_x = [np.pad(series, (0, max_length - len(series)), 'constant') for series in series_data]
    padded_y = [np.pad(labels, (0, max_length - len(labels)), 'constant') for labels in series_labels]
    padded_mp = [np.pad(mp, (0, max_length - len(mp)), 'constant') for mp in series_mp]

    # Split the list of series into training and testing sets
    split_index = len(padded_x) // 2
    x_train = padded_x[:split_index]
    y_train = padded_y[:split_index]
    x_test = padded_x[split_index:]
    y_test = padded_y[split_index:]

    mp_train = padded_mp[:split_index]
    mp_test = padded_mp[split_index:]

    # Save data into separate files
    with open(output_prefix + '_x_train.pkl', 'wb') as f:
        pickle.dump(x_train, f)
    with open(output_prefix + '_y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open(output_prefix + '_x_test.pkl', 'wb') as f:
        pickle.dump(x_test, f)
    with open(output_prefix + '_y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    with open(output_prefix + '_mp_train.pkl', 'wb') as f:
        pickle.dump(mp_train, f)
    with open(output_prefix + '_mp_test.pkl', 'wb') as f:
        pickle.dump(mp_test, f)


if __name__ == '__main__':
    path = "/Users/aleg2/Downloads/ydata-labeled-time-series-anomalies-v1_0"  # Update this path to the location of your Yahoo dataset
    mp_path = "/Users/aleg2/Desktop/TNC_MP/data/DAMP_yahoo_672_4032_timestamp.csv"
    output_prefix = 'yahoo'  # The prefix for the output file names
    preprocess_matrix_profile_yahoo(path,mp_path, output_prefix)
