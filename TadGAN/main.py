#!/usr/bin/env python
# coding: utf-8
import os, csv
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

from scipy.signal import butter, filtfilt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

from sklearn.preprocessing import StandardScaler

import model
import anomaly_detection

logging.basicConfig(filename='train.log', level=logging.DEBUG)

class SignalDataset(Dataset):
    def __init__(self, path:str, is_test:bool = False):
        self.signal_df = pd.read_csv(path)
        # self.signal_df = self.make_rms()
        self.signal_df = self.make_abnormal_label(is_test)
        # self.normalization()
        self.signal_columns = self.make_signal_list()
        self.make_rolling_signals()

        
    def make_abnormal_label(self, is_test):
        """
        make abnormal label as 0
        * this function is only for no fault data

        Args:
            data (list): signal dataset
        """
        # print(self.signal_df.head, self.signal_df.columns)
        # if is_test:
        #     normal = [0 for _ in range(600)]
        #     abnormal = [1 for _ in range(600, len(self.signal_df))]
        #     anomaly = normal.extend(abnormal)
        # else:
        #     anomaly = 0
        # self.signal_df = self.signal_df.drop(['skew'], axis=1)
        self.signal_df['anomaly'] = 0
        print(self.signal_df.head)
        self.signal_df.columns = ['signal', 'anomaly']
        # print(self.signal_df.head)
        # self.signal_df = self.signal_df.drop([''], axis=1)
        # print(data.head)
        return self.signal_df
    
    def normalization(self):
        stder = StandardScaler()
        signal = np.array(self.signal_df['signal']).reshape(self.signal_df['signal'].shape[0], 1)
        stder.fit(signal)
        sig = stder.transform(signal)
        self.signal_df['signal'] = sig.reshape(sig.shape[0])

    def make_signal_list(self) -> list:
        """
        Making signal column, range of signal-50 ~ signal49

        Returns:
            list: string list
        """
        signal_list = list()
        for i in range(-50, 50):
            signal_list.append('signal'+str(i))
        return signal_list

    def make_rolling_signals(self) -> None:
        """
        Making dataset index as cycle
        """
        for i in range(-50, 50):
            self.signal_df['signal'+str(i)] = np.roll(self.signal_df['signal'], shift=i)# self.signal_df['signal'].shift(i)
        # drop NaN value and reset index
        self.signal_df = self.signal_df.dropna()
        self.signal_df = self.signal_df.reset_index(drop=True)

    def __len__(self):
        return len(self.signal_df)

    def __getitem__(self, idx):
        row = self.signal_df.loc[idx]
        x = row[self.signal_columns].values.astype(float)
        x = torch.from_numpy(x)
        x.cuda()
        # print(row.keys())
        return {'signal':x, 'anomaly':row['anomaly']}

def critic_x_iteration(sample:list) -> list:
    """
    

    Args:
        sample (numpy array): _description_

    Returns:
        list: _description_
    """
    # Adam optimizer
    optim_cx.zero_grad()

    # Calculate Critic score X - original, fake
    x = sample['signal'].view(1, batch_size, signal_shape).to(device) # x.shape = (1, 64, 100)
    valid_x = critic_x(x)
    valid_x = torch.squeeze(valid_x)
    critic_score_valid_x = torch.mean(torch.ones(valid_x.shape).to(device) * valid_x) # Wasserstein Loss

    # The sampled z are the anomalous points - points deviating from actual distribution of z (obtained through encoding x)
    z = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1) # 1, 64, 100, latent space
    x_ = decoder(z.to(device)) # time series - Generator G 
    fake_x = critic_x(x_)
    fake_x = torch.squeeze(fake_x)
    critic_score_fake_x = torch.mean(torch.ones(fake_x.shape).to(device) * fake_x)  #Wasserstein Loss


    # Gradient
    alpha = torch.rand(x.shape).to(device)
    ix = Variable(alpha * x + (1 - alpha) * x_) #Random Weighted Average
    ix.requires_grad_(True)
    v_ix = critic_x(ix)
    v_ix.mean().backward()
    gradients = ix.grad

    #Gradient Penalty Loss
    gp_loss = torch.sqrt(torch.sum(torch.square(gradients).view(-1)))

    # Detect original one / weight update
    #Critic has to maximize Cx(Valid X) - Cx(Fake X).
    #Maximizing the above is same as minimizing the negative.
    wl = critic_score_fake_x - critic_score_valid_x
    loss = wl + gp_loss
    loss.backward()
    optim_cx.step()

    return loss

def critic_z_iteration(sample):
    # Adam optimizer
    optim_cz.zero_grad()

    # Calculate Critic score Z - original, fake    
    x = sample['signal'].view(1, batch_size, signal_shape).to(device) # x.shape = (1, batch size, signal shape)
    z = encoder(x) # latent space, Generator E
    valid_z = critic_z(z)
    valid_z = torch.squeeze(valid_z)
    critic_score_valid_z = torch.mean(torch.ones(valid_z.shape).to(device) * valid_z)

    z_ = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1).to(device) # fake latent space
    fake_z = critic_z(z_)
    fake_z = torch.squeeze(fake_z)
    critic_score_fake_z = torch.mean(torch.ones(fake_z.shape).to(device) * fake_z) #Wasserstein Loss

    # Gradient
    alpha = torch.rand(z.shape).to(device)
    iz = Variable(alpha * z + (1 - alpha) * z_) #Random Weighted Average
    iz.requires_grad_(True)
    v_iz = critic_z(iz)
    v_iz.mean().backward()
    gradients = iz.grad

    gp_loss = torch.sqrt(torch.sum(torch.square(gradients).view(-1)))

    # Detect original one / weight update
    #Critic has to maximize Cx(Valid X) - Cx(Fake X).
    #Maximizing the above is same as minimizing the negative.
    wl = critic_score_fake_z - critic_score_valid_z
    loss = wl + gp_loss
    loss.backward()
    optim_cz.step()

    return loss

def encoder_iteration(sample):
    optim_enc.zero_grad()

    x = sample['signal'].view(1, batch_size, signal_shape).to(device)
    valid_x = critic_x(x)
    valid_x = torch.squeeze(valid_x)
    critic_score_valid_x = torch.mean(torch.ones(valid_x.shape).to(device) * valid_x) #Wasserstein Loss

    z = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1).to(device) # fake latent space
    x_ = decoder(z) # Generator G, fake time series
    fake_x = critic_x(x_)
    fake_x = torch.squeeze(fake_x)
    critic_score_fake_x = torch.mean(torch.ones(fake_x.shape).to(device) * fake_x)

    enc_z = encoder(x) # real latent space
    gen_x = decoder(enc_z) # real time series

    mse = mse_loss(x.float(), gen_x.float()) # compare real data - reconstruction data
    loss_enc = mse + critic_score_valid_x - critic_score_fake_x
    loss_enc.backward(retain_graph=True)
    optim_enc.step()

    return loss_enc

def decoder_iteration(sample):
    optim_dec.zero_grad()

    x = sample['signal'].view(1, batch_size, signal_shape).to(device)
    z = encoder(x) # latent space 
    valid_z = critic_z(z)
    valid_z = torch.squeeze(valid_z)
    critic_score_valid_z = torch.mean(torch.ones(valid_z.shape).to(device) * valid_z)

    z_ = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1).to(device) # fake latent space
    fake_z = critic_z(z_)
    fake_z = torch.squeeze(fake_z)
    critic_score_fake_z = torch.mean(torch.ones(fake_z.shape).to(device) * fake_z)

    enc_z = encoder(x) # = z, latent space
    gen_x = decoder(enc_z) # time series

    mse = mse_loss(x.float(), gen_x.float())
    loss_dec = mse + critic_score_valid_z - critic_score_fake_z
    loss_dec.backward(retain_graph=True)
    optim_dec.step()

    return loss_dec


def train(n_epochs=2000):
    logging.debug('Starting training')
    cx_epoch_loss = list()
    cz_epoch_loss = list()
    encoder_epoch_loss = list()
    decoder_epoch_loss = list()

    for epoch in range(n_epochs):
        print('Epoch : ', epoch)
        logging.debug('Epoch {}'.format(epoch))
        n_critics = 10

        cx_nc_loss = list()
        cz_nc_loss = list()

        for i in range(n_critics):
            cx_loss = list()
            cz_loss = list()

            for batch, sample in tqdm(enumerate(train_loader)):
                # print(sample.shape)
                sample['signal'].to(device)
                loss = critic_x_iteration(sample)
                cx_loss.append(loss)

                loss = critic_z_iteration(sample)
                cz_loss.append(loss)

            cx_nc_loss.append(torch.mean(torch.tensor(cx_loss)))
            cz_nc_loss.append(torch.mean(torch.tensor(cz_loss)))

        logging.debug('Critic training done in epoch {}'.format(epoch))
        encoder_loss = list()
        decoder_loss = list()

        for batch, sample in tqdm(enumerate(train_loader)):
            sample['signal'].to(device)
            enc_loss = encoder_iteration(sample)
            dec_loss = decoder_iteration(sample)
            encoder_loss.append(enc_loss)
            decoder_loss.append(dec_loss)

        cx_epoch_loss.append(torch.mean(torch.tensor(cx_nc_loss)))
        cz_epoch_loss.append(torch.mean(torch.tensor(cz_nc_loss)))
        encoder_epoch_loss.append(torch.mean(torch.tensor(encoder_loss)))
        decoder_epoch_loss.append(torch.mean(torch.tensor(decoder_loss)))

        # scheduler.step()

        print('Encoder decoder training done in epoch {}'.format(epoch))
        print('critic x loss {:.3f} critic z loss {:.3f} \nencoder loss {:.3f} decoder loss {:.3f}\n'.format(cx_epoch_loss[-1], cz_epoch_loss[-1], encoder_epoch_loss[-1], decoder_epoch_loss[-1]))
        logging.debug('Encoder decoder training done in epoch {}'.format(epoch))
        logging.debug('critic x loss {:.3f} critic z loss {:.3f} \nencoder loss {:.3f} decoder loss {:.3f}\n'.format(cx_epoch_loss[-1], cz_epoch_loss[-1], encoder_epoch_loss[-1], decoder_epoch_loss[-1]))

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            torch.save(encoder.state_dict(), encoder.encoder_path + '_' + str(epoch) + '.pt')
            torch.save(decoder.state_dict(), decoder.decoder_path + '_' + str(epoch) + '.pt')
            torch.save(critic_x.state_dict(), critic_x.critic_x_path + '_' + str(epoch) + '.pt')
            torch.save(critic_z.state_dict(), critic_z.critic_z_path + '_' + str(epoch) + '.pt')

if __name__ == "__main__":

    # path = 'C:/Users/dk866/Desktop/monitoring_system/Data/nothing_90/'

    # sig = []
    # for fn in os.listdir(path):
    #     file = open(path + fn, 'r')
    #     rea = csv.reader(file)

    #     for r in rea:
    #         sig.extend(list(map(float, r)))

    # dataset = pd.DataFrame(sig)

    dataset = pd.read_csv('../Monitoring_System/Data/gearhead_time_feature_normalized.csv')['rms']
    device = torch.device("cuda:0")
    #Splitting intro train and test
    #TODO could be done in a more pythonic way
    train_len = int(0.7 * dataset.shape[0])
    dataset[0:train_len].to_csv('train_dataset.csv', index=False)
    dataset[train_len:].to_csv('test_dataset.csv', index=False)

    train_dataset = SignalDataset(path='train_dataset.csv')
    test_dataset = SignalDataset(path='test_dataset.csv', is_test=True)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True) #, collate_fn=lambda x: default_collate(x).to(device))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True) #, collate_fn=lambda x: default_collate(x).to(device))
    print("     train dataset length: ", len(train_loader))
    # print(len(train_dataset.signal_df))
    logging.info('Number of train datapoints is {}'.format(len(train_dataset)))
    logging.info('Number of samples in train dataset {}'.format(len(train_dataset)))

    lr = 0.0005

    signal_shape = 100
    latent_space_dim = 40
    encoder_path = "TadGAN/models/2000_skew/encoder"
    decoder_path = "TadGAN/models/2000_skew/decoder"
    critic_x_path = "TadGAN/models/2000_skew/critic_x"
    critic_z_path = "TadGAN/models/2000_skew/critic_z"
    
    # Generator E
    encoder = model.Encoder(encoder_path, signal_shape).to(device)
    # Generator G
    decoder = model.Decoder(decoder_path, signal_shape).to(device)
    # Critic X
    critic_x = model.CriticX(critic_x_path, signal_shape).to(device)
    # Critic Z
    critic_z = model.CriticZ(critic_z_path).to(device)

    # encoder.load_state_dict(torch.load('TadGAN/models/encoder_299.pt'))
    # decoder.load_state_dict(torch.load('TadGAN/models/decoder_299.pt'))
    # critic_x.load_state_dict(torch.load('TadGAN/models/critic_x_299.pt'))
    # critic_z.load_state_dict(torch.load('TadGAN/models/critic_z_299.pt'))
    mse_loss = torch.nn.MSELoss()

    optim_enc = optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_dec = optim.Adam(decoder.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_cx = optim.Adam(critic_x.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_cz = optim.Adam(critic_z.parameters(), lr=lr, betas=(0.5, 0.999))

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optim_dec, T_max=300, eta_min=1e-6)

    train(n_epochs=2000)

    anomaly_detection.test(test_loader, encoder, decoder, critic_x)
