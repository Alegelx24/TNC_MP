import torch
import os
import pickle
import numpy as np
import random
from tnc.models import RnnEncoder 
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import column_or_1d
import numpy as np
import time
from sklearn.metrics import f1_score, precision_score, recall_score
import bottleneck as bn
import pandas as pd


# consider delay threshold and missing segments
def get_range_proba(predict, label, delay=7):
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict


# set missing = 0
def reconstruct_label(timestamp, label):
    timestamp = np.asarray(timestamp, np.int64)
    index = np.argsort(timestamp)

    timestamp_sorted = np.asarray(timestamp[index])
    interval = np.min(np.diff(timestamp_sorted))

    label = np.asarray(label, np.int64)
    label = np.asarray(label[index])

    idx = (timestamp_sorted - timestamp_sorted[0]) // interval

    new_label = np.zeros(shape=((timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1,), dtype=np.int)
    new_label[idx] = label

    return new_label



def eval_ad_result(test_pred_list, test_labels_list, test_timestamps_list, delay):
    labels = []
    pred = []
    for test_pred, test_labels, test_timestamps in zip(test_pred_list, test_labels_list, test_timestamps_list):
        assert test_pred.shape == test_labels.shape == test_timestamps.shape
        test_labels = reconstruct_label(test_timestamps, test_labels)
        test_pred = reconstruct_label(test_timestamps, test_pred)
        test_pred = get_range_proba(test_pred, test_labels, delay)
        labels.append(test_labels)
        pred.append(test_pred)
    labels = np.concatenate(labels)
    pred = np.concatenate(pred)

    #combined_data = np.column_stack((labels, pred))
    #results_df = pd.DataFrame(combined_data)
    #results_df.to_csv("prediction_random_encoder_00.csv", index=False)
    
    return {
        'f1': f1_score(labels, pred),
        'precision': precision_score(labels, pred),
        'recall': recall_score(labels, pred)
    }


def np_shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

'''
def extract_sliding_window_repr(encoder, data, sliding_padding=100, mask="all"):
    # Convert data to a PyTorch tensor and add necessary dimensions
    data_tensor = torch.Tensor(data).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_timestamps)
    
    n_timestamps = data_tensor.shape[2]
    representations = []

    # Process each timestamp with sliding window
    for i in range(n_timestamps):
        # Define the window boundaries
        start = max(i - sliding_padding, 0)
        end = i + 1  # window size is 1

        # Extract the window
        window = data_tensor[:, :, start:end]

        # Get the representation for this window
        window_repr = encoder(window, mask=mask)


        # Store the representation (squeeze to remove batch and sequence dimensions)
        representations.append(window_repr.squeeze(0).squeeze(0))

    # Concatenate all representations along a new dimension
    full_repr = torch.stack(representations, dim=0)

    return full_repr
'''

import torch

def extract_sliding_window_repr(encoder, data, sliding_padding=100, mask="all", batch_size=10):
    # Convert data to a PyTorch tensor and add necessary dimensions
    data_tensor = torch.Tensor(data).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_timestamps)
    
    n_timestamps = data_tensor.shape[2]
    representations = []

    # Process in batches
    for i in range(0, n_timestamps, batch_size):
        batch_reprs = []
        for j in range(i, min(i + batch_size, n_timestamps)):
            # Define the window boundaries
            start = max(j - sliding_padding, 0)
            end = j + 1  # window size is 1

            # Extract the window
            window = data_tensor[:, :, start:end]

            # Get the representation for this window
            window_repr = encoder(window, mask=mask)

            # Store the representation (squeeze to remove batch and sequence dimensions)
            batch_reprs.append(window_repr.squeeze(0).squeeze(0))

        # Concatenate batch representations and add to main list
        if batch_reprs:
            batch_reprs = torch.stack(batch_reprs, dim=0)
            representations.append(batch_reprs)

    # Concatenate all representations along a new dimension
    full_repr = torch.cat(representations, dim=0)

    return full_repr


def eval_anomaly_detection(encoder, all_train_data, all_train_labels, all_test_data, all_test_labels, all_test_timestamps, delay):
    t = time.time()
    
    all_train_repr = {}
    all_test_repr = {}
    all_train_repr_wom = {}
    all_test_repr_wom = {}
    
    for k in range(all_train_data.size(0)):
        train_data = all_train_data[k]
        test_data = all_test_data[k].squeeze(0)

        #Note: train data tensor shape ([1680])

        '''
        full_repr = encoder(
            np.concatenate([train_data, test_data]).reshape(1, -1, 1),
            mask='mask_last'
        ).squeeze()
        '''

        torch.no_grad()

        full_repr_train = extract_sliding_window_repr(encoder, train_data, sliding_padding=50, mask="mask_last",)
        full_repr_test = extract_sliding_window_repr(encoder, test_data, sliding_padding=50, mask="mask_last")

        #full_repr_test = torch.Tensor(np.random.rand(*full_repr_train.shape))
      
        '''
        full_repr_train = encoder(
            torch.Tensor(train_data).unsqueeze(0).unsqueeze(0),
            mask='mask_last'
        )

        full_repr_test = encoder(
            torch.Tensor(test_data).unsqueeze(0).unsqueeze(0),
            mask='mask_last'
        )

        full_repr_train_wom = encoder(
           torch.Tensor(train_data).unsqueeze(0).unsqueeze(0),
           mask='all'
        )

        full_repr_test_wom = encoder(
            torch.Tensor(test_data).unsqueeze(0).unsqueeze(0),
            mask='all'
        )
        '''
               
        all_train_repr[k] = full_repr_train
        all_test_repr[k] = full_repr_test

        full_repr_train_wom = extract_sliding_window_repr(encoder, train_data, sliding_padding=50, mask="all")
        full_repr_test_wom = extract_sliding_window_repr(encoder, test_data, sliding_padding=50, mask="all")
        
        #full_repr_train_wom = np.random.rand(*full_repr_train.shape)
        #full_repr_test_wom = np.random.rand(*full_repr_test.shape)      

        all_train_repr_wom[k] = torch.Tensor(full_repr_train_wom)
        all_test_repr_wom[k] = torch.Tensor(full_repr_test_wom)
        
    res_log = []
    labels_log = []
    timestamps_log = []
    
    for k in range(all_train_data.size(0)): #iterate over all datasets subsequences, its ok to do it on training set because it 50% of the data
        train_data = all_train_data[k]

        test_data = all_test_data[k]
        test_labels = all_test_labels[k]
        test_timestamps = all_test_timestamps[k]

        print("test_data", test_data.shape, "train shape", train_data.shape, "test_timestamps", test_timestamps.shape)

        train_err = np.abs(all_train_repr_wom[k].detach().cpu() - all_train_repr[k].detach().cpu()).sum(axis=1)
        print("train_err", train_err.sum())
        test_err = np.abs(all_test_repr_wom[k].detach().cpu() - all_test_repr[k].detach().cpu()).sum(axis=1)

        ma = np_shift(bn.move_mean(np.concatenate([train_err, test_err]), 21), 1)
        train_err_adj = (train_err - ma[:len(train_err)]) / ma[:len(train_err)]
        test_err_adj = (test_err - ma[len(train_err):]) / ma[len(train_err):]
        train_err_adj = train_err_adj[22:]

        thr = torch.mean(train_err_adj) + 4 * torch.std(train_err_adj)

        test_res = (test_err_adj > thr) * 1

        for i in range(len(test_res)):
            if i >= delay and test_res[i-delay:i].sum() >= 1:
                test_res[i] = 0

        #print("test_res", test_res.shape, )

        for i in range(len(test_res)):
            if i >= delay and test_res[i-delay:i].sum() >= 1:
                test_res[i] = 0

        res_log.append(test_res)
        labels_log.append(test_labels)
        timestamps_log.append(test_timestamps)

    t = time.time() - t
    
    eval_res = eval_ad_result(res_log, labels_log, timestamps_log, delay)
    eval_res['infer_time'] = t
    return res_log, eval_res



if __name__ == "__main__":
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = RnnEncoder(hidden_size=100, in_channel=1, encoding_size=10, device=device)
    tcl_checkpoint = torch.load('./ckpt/yahoo/checkpoint_size10.pth.tar')
    # tcl_checkpoint = torch.load('./ckpt/waveform_trip/checkpoint.pth.tar')
    encoder.load_state_dict(tcl_checkpoint['encoder_state_dict'])
    encoder.eval()
    encoder.to(device)

    window_size = 30
    path = './data/yahoo_data/'

    with open(os.path.join(path, 'yahoo_x_test.pkl'), 'rb') as f:
        x_test = pickle.load(f)
    with open(os.path.join(path, 'yahoo_y_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    
    with open(os.path.join(path, 'yahoo_x_train.pkl'), 'rb') as f:
        x_train = pickle.load(f)
    with open(os.path.join(path, 'yahoo_y_train.pkl'), 'rb') as f:
        y_train = pickle.load(f)
    

    x_test=torch.Tensor(x_test)
    y_test=torch.Tensor(y_test)
    timestamp_test = torch.arange(y_test.shape[0] * y_test.shape[1]).view(y_test.shape)


    x_train=torch.Tensor(x_train)

    y_train=torch.Tensor(y_train)

    T = x_test.shape[-1]

    x_test = x_test.unsqueeze(1)

    x_window = np.split(torch.Tensor(x_test)[ : ,:,:window_size * (T // window_size)], (T // window_size), -1)
    x_window = np.concatenate(x_window, 0)

    #simple splitting in window length sequence 
    y_window = np.concatenate(np.split((y_test)[:, :window_size * (T // window_size)], (T // window_size), -1),0).astype(int) 
    #y_window = np.array([np.bincount(yy).argmax() for yy in y_window])

    #all sequence is considered as a single sample
    y_window = np.array([1 if np.any(yy == 1) else 0 for yy in y_window])

    shuffled_inds_test = list(range(len(x_window)))
    random.shuffle(shuffled_inds_test)

    testset = torch.utils.data.TensorDataset(torch.Tensor(x_window), torch.Tensor(y_window))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True)

    is_anomaly = y_window.copy()

    #Encoding phase
    encodings_train = []
    for x,_ in test_loader:
        encodings_train.append(encoder(x.to(device)).detach().cpu().numpy())
    encodings_train = np.concatenate(encodings_train, 0)


    out, eval_res = eval_anomaly_detection(encoder, x_train, y_train, x_test, y_test, timestamp_test, delay=50)

    print('Evaluation result:', eval_res)












