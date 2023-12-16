
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import argparse
import math
import seaborn as sns; sns.set()
import sys
import numpy as np
import pickle
import os
import random
from tnc.models import RnnEncoder, WFEncoder
from tnc.utils import plot_distribution, track_encoding
from tnc.evaluations import WFClassificationExperiment, ClassificationPerformanceExperiment
from statsmodels.tsa.stattools import adfuller

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TNCDataset_MP_contrastive(data.Dataset): #dataset class to model the set for TNC
    """
    A custom dataset class for TNC (Time Series Nearest Class) dataset.

    Args:
        x (numpy.ndarray): The input time series data.
        mc_sample_size (int): The number of Monte Carlo samples.
        window_size (int): The size of the sliding window.
        augmentation (int): The number of times to augment each sample.
        epsilon (int, optional): The epsilon value for ADF (Augmented Dickey-Fuller) test. Defaults to 3.
        state (torch.Tensor, optional): The state tensor. Defaults to None.
        adf (bool, optional): Whether to use ADF for epsilon calculation. Defaults to False.
    """

    def __init__(self, x, mc_sample_size, window_size, augmentation, epsilon=3, state=None, adf=False, mp=None):
        super(TNCDataset_MP_contrastive, self).__init__()
        self.time_series = x
        self.T = x.shape[-1]
        self.window_size = window_size
        self.sliding_gap = int(window_size*25.2)
        self.window_per_sample = (self.T-2*self.window_size)//self.sliding_gap
        self.mc_sample_size = mc_sample_size
        self.state = state
        self.augmentation = augmentation
        self.adf = adf
        self.mp = mp
        if not self.adf:
            self.epsilon = epsilon #it is for the adf test
            self.delta = 5*window_size*epsilon 

    def __len__(self):
        return len(self.time_series)*self.augmentation

    def __getitem__(self, ind):
        ind = ind%len(self.time_series)
        t = np.random.randint(2*self.window_size, self.T-2*self.window_size) #time series index
        x_t = self.time_series[ind][t-self.window_size//2:t+self.window_size//2]

        #retrieval of positive and negative samples
        X_close = self._find_neighours(self.time_series[ind], t) #these are the positive samples, x_p
        X_distant = self._find_non_neighours(self.time_series[ind], t) # these are the negative samples, x_n 
        if self.state is None:
            y_t = -1
        else:
            y_t = torch.round(torch.mean(self.state[ind][t-self.window_size//2:t+self.window_size//2]))
        return x_t, X_close, X_distant, y_t

    def _find_neighours(self, x, t):
        """
        Find neighboring samples around a given time point.
        Args: x (torch.Tensor): Input time series data.
              t (int): Time index.
        Returns: torch.Tensor: Neighboring samples around the given time point.
        """
        T = self.time_series.shape[-1]
        if self.adf:
            gap = self.window_size
            corr = []
            for w_t in range(self.window_size, 4*self.window_size, gap):
                try:
                    p_val = 0
                    for f in range(x.shape[-2]):
                        p = adfuller(np.array(x[f, max(0, t - w_t):min(x.shape[-1], t + w_t)].reshape(-1, )))[1]
                        p_val += 0.01 if math.isnan(p) else p
                    corr.append(p_val/x.shape[-2])
                except:
                    corr.append(0.6)
            self.epsilon = len(corr) if len(np.where(np.array(corr) >= 0.01)[0]) == 0 else (np.where(np.array(corr) >= 0.01)[0][0] + 1)
            self.delta = 5*self.epsilon*self.window_size
        ## Random from a Gaussian
        t_p = [int(t+np.random.randn()*self.epsilon*self.window_size) for _ in range(self.mc_sample_size)]
        t_p = [max(self.window_size//2+1, min(t_pp, T-self.window_size//2)) for t_pp in t_p]
        #x_p = torch.stack([x[:, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_p])
        x_p = torch.stack([x[ t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_p])

        return x_p

    def _find_non_neighours(self, x, t): #find non-neighbors returns negative samples
        T = self.time_series.shape[-1]
        if t>T/2:
            t_n = np.random.randint(self.window_size//2, max((t - self.delta + 1), self.window_size//2+1), self.mc_sample_size)
        else:
            t_n = np.random.randint(min((t + self.delta), (T - self.window_size-1)), (T - self.window_size//2), self.mc_sample_size)
        #x_n = torch.stack([x[:, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_n])
        x_n = torch.stack([x[ t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_n])

        if len(x_n)==0:
            rand_t = np.random.randint(0,self.window_size//5)
            if t > T / 2:
                x_n = x[:,rand_t:rand_t+self.window_size].unsqueeze(0)
            else:
                x_n = x[:, T - rand_t - self.window_size:T - rand_t].unsqueeze(0)
        return x_n
#end of tnc dataset class
    



def epoch_run_MP_contrastive(loader, disc_model, encoder, device, w=0, optimizer=None, train=True, alpha=1):
    if train:
        encoder.train()
        disc_model.train()
    else:
        encoder.eval()
        disc_model.eval()
    # loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.BCEWithLogitsLoss() #loss function!
    
    #mp loss definition
    mp_loss = 0

    encoder.to(device)
    disc_model.to(device)
    epoch_loss = 0
    epoch_acc = 0
    batch_count = 0

    for x_t, x_p, x_n, _ in loader: #iterate over batches
        #mc_sample = x_p.shape[0]
        mc_sample = x_p.shape[1]

        batch_size, len_size = x_t.shape

        f_size = 1
        x_t = x_t.reshape((-1, f_size, len_size))

        x_p = x_p.reshape((-1, f_size, len_size))
        x_n = x_n.reshape((-1, f_size, len_size))
        x_t = np.repeat(x_t, mc_sample, axis=0)#create multiple samples for each time point for monte carlo sampling
        neighbors = torch.ones((len(x_p))).to(device)
        non_neighbors = torch.zeros((len(x_n))).to(device)
        x_t, x_p, x_n = x_t.to(device), x_p.to(device), x_n.to(device)
        #x_t shape [200, 1, 30]

        z_t = encoder(x_t)
        z_p = encoder(x_p)
        z_n = encoder(x_n)

        d_p = disc_model(z_t, z_p) #output of the discriminator, if close to 1, then the two inputs are close
        d_n = disc_model(z_t, z_n)

        p_loss = loss_fn(d_p, neighbors)
        n_loss = loss_fn(d_n, non_neighbors)
        n_loss_u = loss_fn(d_n, neighbors)
        loss = (p_loss + w*n_loss_u + (1-w)*n_loss)/2

        #hybrid loss
        loss =  loss 

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        p_acc = torch.sum(torch.nn.Sigmoid()(d_p) > 0.5).item() / len(z_p)
        n_acc = torch.sum(torch.nn.Sigmoid()(d_n) < 0.5).item() / len(z_n)
        epoch_acc = epoch_acc + (p_acc+n_acc)/2
        epoch_loss += loss.item()
        batch_count += 1
    return epoch_loss/batch_count, epoch_acc/batch_count #returns the average loss and accuracy for the epoch

