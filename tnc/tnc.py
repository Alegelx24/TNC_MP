"""
Temporal Neighborhood Coding (TNC) for unsupervised learning representation of non-stationary time series
"""
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
from tnc.tnc_contrastive import TNCDataset_MP_contrastive, epoch_run_MP_contrastive

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Discriminator(torch.nn.Module):
    def __init__(self, input_size, device):

        super(Discriminator, self).__init__()
        self.device = device
        self.input_size = input_size

        self.model = torch.nn.Sequential(torch.nn.Linear(2*self.input_size, 4*self.input_size),
                                         torch.nn.ReLU(inplace=True),
                                         torch.nn.Dropout(0.5),
                                         torch.nn.Linear(4*self.input_size, 1))

        torch.nn.init.xavier_uniform_(self.model[0].weight)
        torch.nn.init.xavier_uniform_(self.model[3].weight)

    def forward(self, x, x_tild):
        """
        Predict the probability of the two inputs belonging to the same neighbourhood.
        """
        x_all = torch.cat([x, x_tild], -1)
        p = self.model(x_all)
        return p.view((-1,))


class TNCDataset(data.Dataset): #dataset class to model the set for TNC
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
        super(TNCDataset, self).__init__()
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
            self.epsilon = epsilon
            self.delta = 5*window_size*epsilon

    def __len__(self):
        return len(self.time_series)*self.augmentation

    def __getitem__(self, ind):
        ind = ind%len(self.time_series)
        t = np.random.randint(2*self.window_size, self.T-2*self.window_size)
        #x_t = self.time_series[ind][:,t-self.window_size//2:t+self.window_size//2]
        x_t = self.time_series[ind][t-self.window_size//2:t+self.window_size//2]

        #plt.savefig('./plots/%s_seasonal.png'%ind)#save the plot of the time series
        X_close = self._find_neighours(self.time_series[ind], t) #these are the positive samples, x_p
        X_distant = self._find_non_neighours(self.time_series[ind], t) # these are the negative samples, x_n 
        if self.state is None:
            y_t = -1
        else:
            y_t = torch.round(torch.mean(self.state[ind][t-self.window_size//2:t+self.window_size//2]))


        if self.mp is not None:
            Mp_close = self._find_mp_neighours(self.mp[ind], t) 
            Mp_distant = self._find_mp_non_neighours(self.mp[ind], t) 
        else:
            Mp_close= torch.zeros(X_close.shape)
            Mp_distant = torch.zeros(X_distant.shape)
        
 
        #matrix profile
        if self.mp is not None:
            matrix_profile_window = self.mp[ind][t-self.window_size//2:t+self.window_size//2] #window mp values
            matrix_profile_value = self.mp[ind][t] #single sample mp value
            matrix_profile_window_mean = self.mp[ind][t-self.window_size//2:t+self.window_size//2].mean() #window mp values

        else:
            matrix_profile_window = torch.zeros(self.window_size)
            matrix_profile_value = torch.zeros(1)
            matrix_profile_window_mean = torch.zeros(1)



        return x_t, X_close, X_distant, y_t, matrix_profile_window, matrix_profile_window_mean, matrix_profile_value, Mp_close, Mp_distant

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
    
    def _find_mp_neighours(self, mp, t):
        T = self.time_series.shape[-1]
        if self.adf:
            gap = self.window_size
            corr = []
            for w_t in range(self.window_size, 4*self.window_size, gap):
                try:
                    p_val = 0
                    for f in range(mp.shape[-2]):
                        p = adfuller(np.array(mp[f, max(0, t - w_t):min(mp.shape[-1], t + w_t)].reshape(-1, )))[1]
                        p_val += 0.01 if math.isnan(p) else p
                    corr.append(p_val/mp.shape[-2])
                except:
                    corr.append(0.6)
            self.epsilon = len(corr) if len(np.where(np.array(corr) >= 0.01)[0]) == 0 else (np.where(np.array(corr) >= 0.01)[0][0] + 1)
            self.delta = 5*self.epsilon*self.window_size
        ## Random from a Gaussian
        t_p = [int(t+np.random.randn()*self.epsilon*self.window_size) for _ in range(self.mc_sample_size)]
        t_p = [max(self.window_size//2+1, min(t_pp, T-self.window_size//2)) for t_pp in t_p]
        #x_p = torch.stack([x[:, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_p])
        mp_p = torch.stack([mp[ t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_p])

        return mp_p
    
    def _find_mp_non_neighours(self, mp, t): 
        T = self.time_series.shape[-1]
        if t>T/2:
            t_n = np.random.randint(self.window_size//2, max((t - self.delta + 1), self.window_size//2+1), self.mc_sample_size)
        else:
            t_n = np.random.randint(min((t + self.delta), (T - self.window_size-1)), (T - self.window_size//2), self.mc_sample_size)
        #x_n = torch.stack([x[:, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_n])
        mp_n = torch.stack([mp[ t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_n])

        if len(mp_n)==0:
            rand_t = np.random.randint(0,self.window_size//5)
            if t > T / 2:
                mp_n = mp[:,rand_t:rand_t+self.window_size].unsqueeze(0)
            else:
                mp_n = mp[:, T - rand_t - self.window_size:T - rand_t].unsqueeze(0)
        return mp_n
    
#end of tnc dataset class


def epoch_run(loader, disc_model, encoder, device, w=0, optimizer=None, train=True, alpha=1):
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
    mp_on_encoding=False

    encoder.to(device)
    disc_model.to(device)
    epoch_loss = 0
    epoch_acc = 0
    batch_count = 0

    for x_t, x_p, x_n, _, matrix_profile_window, matrix_profile_window_mean, _ , Mp_close, Mp_distant in loader: #iterate over batches
        #mc_sample = x_p.shape[0]
        mc_sample = x_p.shape[1]
        
        #single timestamp
        mp_loss_single_timestamp = matrix_profile_window.float() #should be of shape([10,30])

        #sum over the window
        mp_loss_sum = matrix_profile_window.float().sum() #should be of shape([1]) 
        
        #mp loss over window mean
        mp_loss_window_mean = matrix_profile_window_mean.float() #should be of shape([10])      
        
        #topK contribute
        k=1
        mp_loss_topK = torch.topk(matrix_profile_window, k, dim=1, largest=True, sorted=True, out=None)[0] #should be of shape([10,1])

        #mean also over the batches
        flattened_matrix_profile = torch.flatten(matrix_profile_window)
        mp_loss_over_window_mean_batches = flattened_matrix_profile.float().mean() #should be of shape([1])

        #mean threshold
        threshold = mp_loss_window_mean.mean()
        std = mp_loss_window_mean.std()
        mask = mp_loss_single_timestamp > (threshold + std)
        elements_greater_than_threshold = mp_loss_single_timestamp[mask]
        mp_loss_sum_threshold = elements_greater_than_threshold.sum()

        '''
        OTHER SOLUTIONS FOR MP LOSS
        threshold = mp_loss_window_mean
        mp_loss_exp = torch.exp(matrix_profile_value - threshold) - 1

        anomaly_loss = torch.mean((torch.nn.functional.relu(matrix_profile_value - threshold)), dim=0) # Only penalizes values above the threshold
        '''

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
 
        if mp_on_encoding == True:# MP with encoding inside the loss or not
                
            #matrix profile encoding loss inside the discriminator
            Mp_close = Mp_close.reshape((-1, f_size, len_size))
            Mp_distant = Mp_distant.reshape((-1, f_size, len_size))
            Mp_close, Mp_distant = Mp_close.to(device), Mp_distant.to(device)
            z_mp_close = encoder(Mp_close)
            z_mp_distant = encoder(Mp_distant)
            d_mp= disc_model(z_mp_close, z_mp_distant)
            mp_loss_encoding = loss_fn(d_mp, non_neighbors)

            z_t = encoder(x_t)
            z_p = encoder(x_p)
            z_n = encoder(x_n)

            d_p = disc_model(z_t, z_p) #output of the discriminator, if close to 1, then the two inputs are close
            d_n = disc_model(z_t, z_n)

            p_loss = loss_fn(d_p, neighbors)
            n_loss = loss_fn(d_n, non_neighbors)
            n_loss_u = loss_fn(d_n, neighbors)
            loss = (p_loss + w*n_loss_u + (1-w)*n_loss + mp_loss_encoding)/3

        else:
            z_t = encoder(x_t)
            z_p = encoder(x_p)
            z_n = encoder(x_n)

            d_p = disc_model(z_t, z_p) #output of the discriminator, if close to 1, then the two inputs are close
            d_n = disc_model(z_t, z_n)

            p_loss = loss_fn(d_p, neighbors)
            n_loss = loss_fn(d_n, non_neighbors)
            n_loss_u = loss_fn(d_n, neighbors)
            loss = (p_loss + w*n_loss_u + (1-w)*n_loss)/2

            #window mean loss
            #mp_loss = mp_loss_window_mean.mean()
            
            # window sum loss
            #mp_loss = mp_loss_sum

            #topK loss mean
            mp_loss = mp_loss_topK.sum()

            # sum of discord over the threshold mean
            #mp_loss = mp_loss_sum_threshold

            #hybrid loss
            loss = (alpha)* loss + (1-alpha)*(mp_loss)

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






#training loop function
def learn_encoder(x, encoder, window_size, w, lr=0.001, decay=0.005, mc_sample_size=20,
                  n_epochs=100, path='simulation', device='cpu', augmentation=1, n_cross_val=1,
                  cont=False, encoding_size=180, mp=None, alpha=1, mp_contrastive=False, model_name='tnc'):
    accuracies, losses = [], []
    for cv in range(n_cross_val): #cross validation loop over n cv folds
        if 'waveform' in path:
            encoder = WFEncoder(encoding_size=64).to(device)
            batch_size = 5
        elif 'simulation' in path:
            encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=10, device=device)
            batch_size = 10
        elif 'har' in path: #har dataset, 561 features, RNN encoder with 100 hidden units
            encoder = RnnEncoder(hidden_size=100, in_channel=561, encoding_size=10, device=device)
            batch_size = 10
        elif 'yahoo' in path: #yahoo dataset, 1 feature, RNN encoder with 100 hidden units
            encoder = RnnEncoder(hidden_size=100, in_channel=1, encoding_size=encoding_size, device=device)
            batch_size = 10 

        if not os.path.exists('./ckpt/%s'%path):#create checkpoint folder
            os.mkdir('./ckpt/%s'%path)
        if cont:#continue training from a checkpoint
            checkpoint = torch.load('./ckpt/%s/checkpoint_%d.pth.tar'%(path, cv))
            encoder.load_state_dict(checkpoint['encoder_state_dict'])

        disc_model = Discriminator(encoder.encoding_size, device)
        params = list(disc_model.parameters()) + list(encoder.parameters())#set of parameters to optimize
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=decay)
        inds = list(range(len(x)))
        random.shuffle(inds)
        x = x[inds]# take a random subsequence
        n_train = int(0.8*len(x))#80% of the data for training
        performance = []
        best_acc = 0
        best_loss = np.inf

        #Matrix profile data, take random subsequence
        if mp is not None:
            mp = mp[inds]

        for epoch in range(n_epochs+1):
            # Create the dataset and dataloader
            if mp is not None and mp_contrastive == False:
                trainset = TNCDataset(x=torch.Tensor(x[:n_train]), mc_sample_size=mc_sample_size,
                                      window_size=window_size, augmentation=augmentation, adf=True, mp = torch.Tensor(mp[:n_train]))
            # x_p and x_n are inside a 2D array, each row is a sample of 40 timestamps, we have 20 elements  
                
            
            elif mp is not None and mp_contrastive == True : #Contrastive loss integration with matrix profile
                trainset = TNCDataset_MP_contrastive(x=torch.Tensor(x[:n_train]), mc_sample_size=mc_sample_size,
                                     window_size=window_size, augmentation=augmentation, adf=True, mp = torch.Tensor(mp[:n_train])) 
            
            else:
                trainset = TNCDataset(x=torch.Tensor(x[:n_train]), mc_sample_size=mc_sample_size,
                                      window_size=window_size, augmentation=augmentation, adf=True)              
                

            train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
                
            # Create the validation set (20% of the data)
            validset = TNCDataset(x=torch.Tensor(x[n_train:]), mc_sample_size=mc_sample_size,
                                  window_size=window_size, augmentation=augmentation, adf=True)
            valid_loader = data.DataLoader(validset, batch_size=batch_size, shuffle=True)

            # Run the training    
            if mp_contrastive == True:
                epoch_loss, epoch_acc = epoch_run_MP_contrastive(train_loader, disc_model, encoder, optimizer=optimizer,
                                              w=w, train=True, device=device, alpha= alpha)
            else:
                epoch_loss, epoch_acc = epoch_run(train_loader, disc_model, encoder, optimizer=optimizer,
                                              w=w, train=True, device=device, alpha= alpha)
            # Run the validation
            test_loss, test_acc = epoch_run(valid_loader, disc_model, encoder, train=False, w=w, device=device, alpha= alpha)
            performance.append((epoch_loss, test_loss, epoch_acc, test_acc))

            if epoch%10 == 0:
                print('(cv:%s)Epoch %d Loss =====> Training Loss: %.5f \t Training Accuracy: %.5f \t Test Loss: %.5f \t Test Accuracy: %.5f'
                      % (cv, epoch, epoch_loss, epoch_acc, test_loss, test_acc))
                
            # Save the best model inside the checkpoint folder
            if best_loss > test_loss or path=='har':
                best_acc = test_acc
                best_loss = test_loss
                state = {
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'discriminator_state_dict': disc_model.state_dict(),
                    'best_accuracy': test_acc
                }
                torch.save(state, './ckpt/%s/checkpoint_%s.pth.tar'%(path,model_name))

        accuracies.append(best_acc)
        losses.append(best_loss)

        # Save performance plots
        if not os.path.exists('./plots/%s'%path):
            os.mkdir('./plots/%s'%path)
        train_loss = [t[0] for t in performance]
        test_loss = [t[1] for t in performance]
        train_acc = [t[2] for t in performance]
        test_acc = [t[3] for t in performance]
        plt.figure()
        plt.plot(np.arange(n_epochs+1), train_loss, label="Train")
        plt.plot(np.arange(n_epochs+1), test_loss, label="Test")
        plt.title("Loss")
        plt.legend()
        plt.savefig(os.path.join("./plots/%s"%path, "loss_%s.pdf"%model_name))
        plt.figure()
        plt.plot(np.arange(n_epochs+1), train_acc, label="Train")
        plt.plot(np.arange(n_epochs+1), test_acc, label="Test")
        plt.title("Accuracy")
        plt.legend()
        plt.savefig(os.path.join("./plots/%s"%path, "accuracy_%s.pdf"%model_name))

    print('=======> Performance Summary:')
    print('Accuracy: %.2f +- %.2f'%(100*np.mean(accuracies), 100*np.std(accuracies)))
    print('Loss: %.4f +- %.4f'%(np.mean(losses), np.std(losses)))
    return encoder

# Main function
def main(is_train, data_type, cv, w, cont, epochs, encoding_size, matrix_profile, alpha=1, mp_contrastive=False, model_name='tnc'):
    if not os.path.exists("./plots"):
        os.mkdir("./plots")
    if not os.path.exists("./ckpt/"):
        os.mkdir("./ckpt/")

    # Yahoo data
    if data_type == 'yahoo':
        #set window size 
        window_size = 120
        path = './data/yahoo_data/'
        #initialization of encoder
        encoder = RnnEncoder(hidden_size=100, in_channel=1, encoding_size=encoding_size, device=device)

        if is_train: #train the Rnn encoder on the training set

            with open(os.path.join(path, 'yahoo_as_ts2vec_x_train.pkl'), 'rb') as f:
                x = pickle.load(f)
            if matrix_profile is True:    
                with open(os.path.join(path, 'yahoo_as_ts2vec_mp_train.pkl'), 'rb') as f:
                    mp = pickle.load(f)
                    mp_tensor= torch.Tensor(mp)
                    print("mp_tensor shape: ", mp_tensor.shape)
            else:
                mp_tensor = None

            learn_encoder(torch.Tensor(x), encoder, w=w, lr=1e-3, decay=1e-5, n_epochs=epochs, window_size=window_size,
                        path='yahoo', mc_sample_size=20, device=device, augmentation=5, n_cross_val=cv, encoding_size=encoding_size,
                        mp=mp_tensor, alpha=alpha, mp_contrastive=mp_contrastive, model_name=model_name)
            
        else: #test the encoder on the test set
            with open(os.path.join(path, 'yahoo_as_ts2vec_x_test.pkl'), 'rb') as f:
                x_test = pickle.load(f)
            with open(os.path.join(path, 'yahoo_as_ts2vec_y_test.pkl'), 'rb') as f:
                y_test = pickle.load(f)
            checkpoint = torch.load('./ckpt/%s/checkpoint_0.pth.tar' % (data_type))
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            encoder = encoder.to(device)
            #track_encoding(x_test[0,:,:], y_test[0,:], encoder, window_size, 'har') #used to plot the encoding
            for cv_ind in range(cv):
                plot_distribution(x_test, y_test, encoder, window_size=window_size, path='yahoo', device=device,
                                augment=100, cv=cv_ind, title='TNC')
                
                #set up the classification experiment
                exp = ClassificationPerformanceExperiment(n_states=2, encoding_size=10, path='yahoo', hidden_size=100,
                                                        in_channel=1, window_size=30, cv=cv_ind)
                # Run cross validation for classification
                for lr in [0.001, 0.01, 0.1]:
                    print('===> lr: ', lr)
                    tnc_acc, tnc_auc, e2e_acc, e2e_auc = exp.run(data='yahoo', n_epochs=50, lr_e2e=lr, lr_cls=lr)
                    print('TNC acc: %.2f \t TNC auc: %.2f \t E2E acc: %.2f \t E2E auc: %.2f'%(tnc_acc, tnc_auc, e2e_acc, e2e_auc))



if __name__ == '__main__':
    random.seed(1234)
    parser = argparse.ArgumentParser(description='Run TNC')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--w', type=float, default=0.05)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--encoding_size', type=int, default=180)
    parser.add_argument('--mp',  action='store_true')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--mp_contrastive', action='store_true')
    parser.add_argument('--model_name', type=str, default='tnc')
    args = parser.parse_args()
    print('TNC model with w=%f'%args.w)
    main(args.train, args.data, args.cv, args.w, args.cont, args.epochs, args.encoding_size, args.mp, args.alpha, args.mp_contrastive, args.model_name)


