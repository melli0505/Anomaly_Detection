import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import csv
import os

from model import Encoder, Decoder, CriticX, CriticZ
from main import SignalDataset
import math

def reconstruction_errors(y, y_hat, step_size=1, score_window=10, smoothing_window=0.01,
                          smooth=True, rec_error_type='point'):

    if isinstance(smoothing_window, float):
        smoothing_window = min(math.trunc(len(y) * smoothing_window), 200)

    true = [item[0] for item in y.reshape((y.shape[0], -1))]
    for item in y[-1][1:]:
        true.extend(item)

    predictions = []
    predictions_vs = []

    pred_length = y_hat.shape[1]
    num_errors = y_hat.shape[1] + step_size * (y_hat.shape[0] - 1)

    for i in range(num_errors):
        intermediate = []
        for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
            intermediate.append(y_hat[i - j, j])
        if intermediate:
            predictions.append(np.median(np.asarray(intermediate)))

            predictions_vs.append([[
                np.min(np.asarray(intermediate)),
                np.percentile(np.asarray(intermediate), 25),
                np.percentile(np.asarray(intermediate), 50),
                np.percentile(np.asarray(intermediate), 75),
                np.max(np.asarray(intermediate))
            ]])

    true = np.asarray(true)
    predictions = np.asarray(predictions)
    predictions_vs = np.asarray(predictions_vs)

    # Compute reconstruction errors
    if rec_error_type.lower() == "point":
        errors = _point_wise_error(true, predictions)

    elif rec_error_type.lower() == "area":
        errors = _area_error(true, predictions, score_window)

    elif rec_error_type.lower() == "dtw":
        errors = _dtw_error(true, predictions, score_window)

    # Apply smoothing
    if smooth:
        errors = pd.Series(errors).rolling(
            smoothing_window, center=True, min_periods=smoothing_window // 2).mean().values

    return errors, predictions_vs


def _point_wise_error(y, y_hat):
    return abs(y - y_hat)

def _area_error(y, y_hat, score_window=10):
    smooth_y = pd.Series(y).rolling(
        score_window, center=True, min_periods=score_window // 2).apply(integrate.trapz)
    smooth_y_hat = pd.Series(y_hat).rolling(
        score_window, center=True, min_periods=score_window // 2).apply(integrate.trapz)

    errors = abs(smooth_y - smooth_y_hat)

    return errors


def _dtw_error(y, y_hat, score_window=10):
    length_dtw = (score_window // 2) * 2 + 1
    half_length_dtw = length_dtw // 2

    # add padding
    y_pad = np.pad(y, (half_length_dtw, half_length_dtw),
                   'constant', constant_values=(0, 0))
    y_hat_pad = np.pad(y_hat, (half_length_dtw, half_length_dtw),
                       'constant', constant_values=(0, 0))

    i = 0
    similarity_dtw = list()
    while i < len(y) - length_dtw:
        true_data = y_pad[i:i + length_dtw]
        true_data = true_data.flatten()

        pred_data = y_hat_pad[i:i + length_dtw]
        pred_data = pred_data.flatten()

        dist = dtw(true_data, pred_data)
        similarity_dtw.append(dist)
        i += 1

    errors = ([0] * half_length_dtw + similarity_dtw +
              [0] * (len(y) - len(similarity_dtw) - half_length_dtw))

    return errors


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset = self.make_anomaly_label()
        self.columns = self.make_signal_list()
        self.make_rolling_signals()

    def make_anomaly_label(self):
        dataset = pd.DataFrame(self.dataset)
        dataset['anomaly'] = 0
        dataset.columns = ['signal', 'anomaly']
        dataset = dataset.reset_index()
        return dataset

    def make_signal_list(self):
        signal_list = list()
        for i in range(-50, 50):
            signal_list.append('signal'+str(i))
        return signal_list
        
    def make_rolling_signals(self) -> None:
        for i in range(-50, 50):
            self.dataset['signal'+ str(i)] = np.roll(self.dataset['signal'], shift=i)
        self.dataset = self.dataset.reset_index(drop=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.loc[idx]
        x = row[self.columns].values.astype(float)
        x = torch.from_numpy(x)
        return {'signal': x, 'anomaly': row['anomaly']}


def make_rms(sr, data):
    temp = []
    for i in range(len(data) // int(sr)):
        temp.append(data[i * int(sr) : (i + 1) * int(sr)])
    print(len(temp), len(temp[0]))
    rms = []
    for l in temp:
        rms.append(np.sqrt((l ** 2).sum()  / sr)) 
    print(len(rms))
    return rms

def normalization(dataset):
    stder = MinMaxScaler()
    # signal = np.array(self.dataset['signal']).reshape(self.dataset['signal'].shape[0], 1)
    stder.fit(np.array(dataset).reshape(-1, 1))
    sig = stder.transform(np.array(dataset).reshape(-1, 1))
    print(sig.reshape(sig.shape[0]).shape)
    return sig.reshape(sig.shape[0])

if __name__ == '__main__':
    
    # merge data
    filepath = input('Enter filepath or dir: ')
    origin_signal = []

    if not os.path.isdir(filepath):
        column_name = input("Enter column name: ")
        origin_signal = pd.read_csv(filepath)[column_name]
    else:
        limit = 3
        for filename in os.listdir(filepath):
            file = open(filepath + '/' + filename, 'r')
            reader = csv.reader(file)
            for row in reader:
                origin_signal.extend(list(map(float, row)))
            # limit -= 1
            # if limit < 0: break

    # feature extraction
    is_row = input('If above data is row data, enter Y: ')

    if is_row in ['Y', 'y']:
        sampling_rate = float(input('Enter the sampling rate: '))
        signal = make_rms(sampling_rate, np.array(origin_signal))

    signal = normalization(signal)

    # make data as trainable shape
    dataset = Dataset(signal)
    
    # print(dataset.dataset['signal'][:5])
    # gpu accelerator 
    device = torch.device("cuda:0")

    # pytorch dataloader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, drop_last=True)

    # load models
    encoder = Encoder('./TadGAN/models/encoder.pt').to('cuda:0')
    encoder.load_state_dict(torch.load('./TadGAN/models/2000_rms/encoder_1999.pt'))

    decoder = Decoder('./TadGAN/models/decoder.pt').to('cuda:0')
    decoder.load_state_dict(torch.load('./TadGAN/models/2000_rms/decoder_1999.pt'))

    criticz = CriticZ('./TadGAN/models/critic_z.pt').to('cuda:0')
    criticz.load_state_dict(torch.load('./TadGAN/models/2000_rms/critic_z_1999.pt'))

    criticx = CriticX('./TadGAN/models/critic_x.pt').to('cuda:0')
    criticx.load_state_dict(torch.load('./TadGAN/models/2000_rms/critic_x_1999.pt'))

    # reconstruction
    recon_result = list()
    
    for batch, sample in enumerate(data_loader):
        recon_signal = decoder(encoder(sample['signal']))
        recon_signal = torch.squeeze(recon_signal)
        recon_result.extend(recon_signal.detach().cpu().numpy())

    recon_result = pd.DataFrame(recon_result)

    error, _ = reconstruction_errors(np.array(signal[:len(recon_result[0])]).reshape(-1, 1), np.array(recon_result[0]).reshape(-1, 1))

    
    plt.figure(figsize=[20, 10])
    plt.subplot(311)
    plt.plot(signal, color='orange')

    plt.subplot(312)
    plt.plot(dataset.dataset['signal'][:len(recon_result[0])], color='b')
    plt.plot(recon_result[0], color='g')
    
    plt.subplot(313)
    plt.plot(error, color='r')
    plt.show()