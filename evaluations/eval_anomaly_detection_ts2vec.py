import torch
import os
import pickle
import numpy as np
import random

from tnc.models import RnnEncoder 
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import column_or_1d

device = 'cuda' if torch.cuda.is_available() else 'cpu'

encoder = RnnEncoder(encoding_size=10, hidden_size=100, in_channel=1)
tcl_checkpoint = torch.load('./ckpt/yahoo/checkpoint_0.pth.tar')
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

# with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
#     x_train = pickle.load(f)
# with open(os.path.join(path, 'state_train.pkl'), 'rb') as f:
#     y_train = pickle.load(f)

T = torch.Tensor(x_test).shape[-1]
x_window = np.split(torch.Tensor(x_test)[ :window_size * (T // window_size)], (T // window_size), -1)
x_window = np.concatenate(x_window, 0)
y_window = np.concatenate(np.split(torch.Tensor(y_test)[:, :window_size * (T // window_size)], (T // window_size), -1),0).astype(int)
y_window = np.array([np.bincount(yy).argmax() for yy in y_window])

T = torch.Tensor(x_test).shape[-1]
# x_window_train = np.split(x_train[:, :, :window_size * (T // window_size)], (T // window_size), -1)
# x_window_train = np.concatenate(x_window_train, 0)
# y_window_train = np.concatenate(np.split(y_train[:, :window_size * (T // window_size)], (T // window_size), -1),0).astype(int)
# y_window_train = np.array([np.bincount(yy).argmax() for yy in y_window_train])
# shuffled_inds_train = list(range(len(x_window_train)))
# random.shuffle(shuffled_inds_train)
shuffled_inds_test = list(range(len(x_window)))
random.shuffle(shuffled_inds_test)

# print(x_window.shape, y_window.shape)
testset = torch.utils.data.TensorDataset(torch.Tensor(x_window), torch.Tensor(y_window))
test_loader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True)
# trainset = torch.utils.data.TensorDataset(torch.Tensor(x_window_train), torch.Tensor(y_window_train))
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)


is_anomaly = y_window.copy()
is_anomaly = np.logical_or(is_anomaly==1, is_anomaly==2).astype(int)

encodings_train = []
############################################################################################ iNTERVENIRE QUI!!!!!!!!!
for x,_ in test_loader:
    encodings_train.append(encoder(x.to(device)).detach().cpu().numpy())
encodings_train = np.concatenate(encodings_train, 0)
print(encodings_train.shape)

# train the KNN detector
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.mcd import MCD
from pyod.models.lscp import LSCP
# from pyod.models.auto_encoder import AutoEncoder

clf_knn = KNN()
clf_pca = PCA()
clf_mcd = MCD()
clf_lof = LOF()
clf_cblof = CBLOF()
# clf_lscp = LSCP([clf_knn, clf_pca, clf_mcd ])
# clf_ae = AutoEncoder(epochs=50)

clf_mcd.fit(encodings_train)
clf_pca.fit(encodings_train)
clf_knn.fit(encodings_train)
clf_lof.fit(encodings_train)
clf_cblof.fit(encodings_train)
# clf_lscp.fit(encodings_train)
# clf_ae.fit(encodings_train)

anomaly_scores_mcd = clf_mcd.decision_function(encodings_train)
anomaly_scores_pca = clf_pca.decision_function(encodings_train)
anomaly_scores_knn = clf_knn.decision_function(encodings_train)
anomaly_scores_lof = clf_lof.decision_function(encodings_train)
anomaly_scores_cblof = clf_cblof.decision_function(encodings_train)
# anomaly_scores_lscp = clf_lscp.decision_function(encodings_train)
# anomaly_scores_ae = clf_ae.predict_proba(encodings_train)


# y_test_scores = []
# for x,_ in test_loader:
#     encodings_test = encoder(torch.Tensor(x).to(device))
#     probs = clf.predict_proba(encodings_test.detach().cpu().numpy())
#     y_test_scores.extend(probs[:,0])
# y_test_scores = np.array(y_test_scores)

y_ind_1 = np.argwhere(y_window.reshape(-1, ) == 1)
y_ind_3 = np.argwhere(y_window.reshape(-1, ) == 3)

for i, anomaly_scores in enumerate([anomaly_scores_knn, anomaly_scores_lof, anomaly_scores_cblof, anomaly_scores_mcd, anomaly_scores_pca]):
    method = ['KNN', 'LOF', 'CBLOF', 'MCD', 'PCA'][i]
    print('********** Results for ', method)
    auc = roc_auc_score(column_or_1d(is_anomaly), column_or_1d(anomaly_scores))#[:,0])
    auprc = average_precision_score(column_or_1d(is_anomaly), column_or_1d(anomaly_scores))#[:,0])
    print('Anomaly detection AUC: ', auc)
    print('Anomaly detection AUPRC: ', auprc)
    print('Label 1: ', np.mean(anomaly_scores[y_ind_1.reshape(-1,)]), '+-',
          np.std(anomaly_scores[y_ind_1.reshape(-1,)]))
    print('Label 3: ', np.mean(anomaly_scores[y_ind_3.reshape(-1,)]), '+-',
          np.std(anomaly_scores[y_ind_3.reshape(-1,)]))


